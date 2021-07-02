import io
import logging
import os
import sys
import time
from PIL import Image
import numpy as np
from bson import ObjectId
from datetime import datetime
from gridfs import GridFS
from pymongo import MongoClient


sys.path.append('.')
from ncm_unico.carrega_modelo_final_ncm import NCMUnico

logging.basicConfig(level=logging.DEBUG)

MIN_RATIO = 2.1

class Comunica():
    """Comportamento padrão para facilitar comunicação modelos-bancos de dados.

    Implementa verbos como

    get_cursor_sem: retorna cursor com registros para gravar predições do modelo
    get_cursor_com: retorna cursor com registros que já tem predições do modelo
    get_imagem: retorna imagem ja com tratamentos que o modelo precisa
    update_db: pega "limit" registros do cursor_sem, roda predições e grava no campo correto.

    """
    # PLACEHOLDERS - Constantes que precisam ser definidas pelas classes filhas
    FILTRO = {'metadata.contentType': 'image/jpeg'}
    CAMPO_ATUALIZADO = 'metadata.predictions.0'
    DATA_INICIAL = datetime(2021, 1, 1)

    def __init__(self, model, mongodb, sqlsession=None, limit=10):
        """
        Args:
            model: modelo para predição, com método predict que recebe imagem e retorna
            a predição pronta para ser gravada no MongoDB
            mongodb: conexão ao banco MongoDB
            sqlsession: conexão ao banco MySQL
            limit: quantidade de registros a limitar no cursor
        """
        self.model = model
        self.mongodb = mongodb
        self.sqlsession = sqlsession
        self.limit = limit
        self.campo_a_atualizar = self.CAMPO_ATUALIZADO
        self.set_filtro(self.DATA_INICIAL)
        self.set_cursor()

    def set_filtro(self, datainicio):
        """Filtro básico. Nas classes filhas adicionar campos."""
        self.filtro = self.FILTRO
        self.filtro['metadata.dataescaneamento'] = {'$gte': datainicio}

    def update_filtro(self, filtro_adicional: dict):
        self.filtro.update(filtro_adicional)

    def set_cursor(self):
        self.cursor = self.mongodb['fs.files'].find(self.filtro,
                                                    {'metadata.predictions': 1}
                                                    ).limit(self.limit)[:self.limit]
        logging.info('Consulta ao banco efetuada.')

    def get_pil_image(self, _id: ObjectId):
        fs = GridFS(self.mongodb)
        self.grid_out = fs.get(_id)
        self.image = self.grid_out.read()
        pil_image = Image.open(io.BytesIO(self.image))
        self.pil_image = pil_image.convert('RGB')
        return self.pil_image

    def update_mongo(self):
        print(self.cursor.count())
        for ind, registro in enumerate(self.cursor):
            s0 = time.time()
            _id = ObjectId(registro['_id'])
            pil_image = self.get_pil_image(_id)
            s1 = time.time()
            logging.info(f'Elapsed retrieve time {s1 - s0}')        
            pred_probs = self.model.predict(pil_image)
            confidence = np.max(pred_probs)
            pred_class = np.argmax(pred_probs)
            pred_label = self.model.class_dict[pred_class]
            pred_info = {"prediction": pred_label,
                         "score": float(confidence)}
            s2 = time.time()
            logging.info(f'Elapsed model time {s2 - s1}.')
            logging.info({'_id': _id, 'pred': pred_info})
            self.mongodb['fs.files'].update(
                {'_id': _id},
                {'$set': {self.campo_a_atualizar: pred_info}}
            )
            s3 = time.time()
            logging.info(f'Elapsed update time {s3 - s2} - registro {ind  + 1}\n')
 
class ComunicaReeeferContaminado(Comunica):
    
    NCMS = ["0202", "0901", "1701", "2304", "4011"]
    
    FILTRO = {
                "metadata.contentType": "image/jpeg",
                "metadata.predictions.bbox": {"$exists": True},
                "metadata.predictions.ncm.0.ncm": {"$exists": False},
                "metadata.predictions.0.vazio": False,
                "metadata.carga.ncm": {"$size": 1},
                "metadata.carga.ncm.0.ncm": {"$in": NCMS}
            }

    CAMPO_ATUALIZADO = 'metadata.predictions.0.ncm.0.ncm'

    def get_pil_image(self, _id: ObjectId):
        super().get_pil_image(_id)
        bbox = self.grid_out.metadata['predictions'][0]['bbox']
        # bboxes coordinates model container => (y1, x1, y2, x2)
        self.pil_image = self.pil_image.crop((bbox[1], bbox[0], bbox[3], bbox[2]))
        return self.pil_image


def baixa_erro(comunica, limit=50):
    # update field predictions.ncm to True
    comunica.filtro['metadata.predictions.ncm.0.ncm'] = {"$exists": True}
    
    for ncm in comunica.NCMS:
        # add new fields
        comunica.filtro["metadata.carga.ncm.0.ncm"] = ncm
        comunica.filtro["metadata.predictions.0.ncm.0.ncm.prediction"] = {"$ne": ncm}
    
        comunica.limit = limit
        comunica.set_cursor()
        error = comunica.cursor.count()
        print(f'{error} localizados para NCM {ncm}')
        if error == 0:
            continue

        ncm_dir = os.path.join('erro_ncm', ncm)
        try:
            os.makedirs(ncm_dir)
        except FileExistsError:
            pass
        for registro in comunica.cursor:

            _id = ObjectId(registro['_id'])
            pil_image = comunica.get_pil_image(_id)
            image_name = f'{str(_id)}.jpeg'
            print(image_name)
            try:
                pil_image.save(os.path.join(ncm_dir, image_name))
            except FileExistsError:
                pass

if __name__ == '__main__':
    if len(sys.argv) > 1:
        limit = int(sys.argv[1])
    else:
        limit = 50
    MONGODB_URI = os.environ.get('MONGODB_URI')
    database = ''.join(MONGODB_URI.rsplit('/')[-1:])
    if not MONGODB_URI:
        MONGODB_URI = 'mongodb://localhost'
        database = 'test'
    with MongoClient(host=MONGODB_URI) as conn:
        mongodb = conn[database]
        model = NCMUnico()
        comunica = ComunicaReeeferContaminado(model, mongodb, limit=limit)
        comunica.update_mongo()
        # Para baixar imagens de falso positivo comentar a linha acima e descomentar
        # a linha abaixo.
        #baixa_erro(comunica, limit)




