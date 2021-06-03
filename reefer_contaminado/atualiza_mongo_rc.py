import io
import logging
import os
import sys
import time
import random
from PIL import Image
import numpy as np
from bson import ObjectId
from datetime import datetime
from gridfs import GridFS
from pymongo import MongoClient


sys.path.append('.')
from reefer_contaminado.carrega_modelo_final_rc import ModelContaminado

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
    DATA_INICIAL = datetime(2020, 12, 16)

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

    def update_mongo(self, limit=1):
        for ind, registro in enumerate(self.cursor):
            s0 = time.time()
            _id = ObjectId(registro['_id'])
            pil_image = self.get_pil_image(_id)
            s1 = time.time()
            logging.info(f'Elapsed retrieve time {s1 - s0}')
            pred = self.model.predict(pil_image)
            s2 = time.time()
            logging.info(f'Elapsed model time {s2 - s1}.')
            logging.info({'_id': _id, 'pred': pred})
            self.mongodb['fs.files'].update(
                {'_id': _id},
                {'$set': {self.campo_a_atualizar: pred}}
            )
            s3 = time.time()
            logging.info(f'Elapsed update time {s3 - s2} - registro {ind}')
    
    def get_metrics(self, fbeta, take=1000, SEED=42):

        random.seed(SEED)
        # pega uma amostra randomica de tamanho 'take'de todos os reefers.
        num_docs = self.cursor.count()
        list_idx = random.sample(range(num_docs), take) 
        ypred = []
        for count, i in enumerate(list_idx):
            registro = self.cursor[i]
            _id = ObjectId(registro['_id'])
            pil_image = self.get_pil_image(_id)
            pred = self.model.predict(pil_image)
            print(f"{count + 1} --> {i + 1}ª Imagem {_id} predicted as {'Contaminada' if pred == True else 'Não Contaminada'}")
            ypred.append(np.array(pred, np.float32))

        predicted_positive = np.sum(ypred) # calculating predicted positives     
        print(f'Positive Predictions: {predicted_positive}')
        print(f'fbeta_ajustado: {fbeta - (predicted_positive / take)}')

class ComunicaReeeferContaminado(Comunica):
    FILTRO = {'metadata.contentType': 'image/jpeg',
              'metadata.predictions.reefer.reefer_bbox': {'$exists': True},
              'metadata.predictions.reefer.reefer_contaminado': {'$exists': False}}
    CAMPO_ATUALIZADO = 'metadata.predictions.0.reefer.0.reefer_contaminado'

    def get_pil_image(self, _id: ObjectId):
        super().get_pil_image(_id)
        bbox = self.grid_out.metadata['predictions'][0]['reefer'][0]['reefer_bbox']
        self.pil_image = self.pil_image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
        return self.pil_image


def baixa_falso_positivo(comunica, limit=50):
    comunica.filtro['metadata.predictions.reefer.reefer_contaminado'] = True
    comunica.limit = limit
    comunica.set_cursor()
    try:
        os.mkdir('falsos_positivos')
    except FileExistsError:
        pass
    for registro in comunica.cursor:
        _id = ObjectId(registro['_id'])
        pil_image = comunica.get_pil_image(_id)
        image_name = f'{str(_id)}.jpeg'
        print(image_name)
        try:
            pil_image.save(os.path.join('falsos_positivos', image_name))
        except FileExistsError:
            pass


if __name__ == '__main__':
    if len(sys.argv) > 1:
        limit = int(sys.argv[1])
    else:
        limit = 1
    MONGODB_URI = os.environ.get('MONGODB_URI')
    database = ''.join(MONGODB_URI.rsplit('/')[-1:])
    if not MONGODB_URI:
        MONGODB_URI = 'mongodb://localhost'
        database = 'test'
    with MongoClient(host=MONGODB_URI) as conn:
        mongodb = conn[database]
        model = ModelContaminado()
        comunica = ComunicaReeeferContaminado(model, mongodb, limit=limit)
        #comunica.update_mongo()
        # Para baixar imagens de falso positivo comentar a linha acima e descomentar
        # a linha abaixo.
        # baixa_falso_positivo(comunica, limit)
        
        comunica.get_metrics(fbeta=0.912, take=1000)


#threshold = .6
# +----------------------------------+-----------+--------+-------+-------+-------+
# |        MobileNetV2 Model         | Precision | Recall |  F1   |  F2   |  F4   |
# +----------------------------------+-----------+--------+-------+-------+-------+
# | contaminados_ciclo2_S_b_17_05.h5 | 43.3%     | 92.9%  | 59.1% | 75.6% | 87.0% |
# | contaminados_ciclo2_S_c_17_05.h5 | 22.6%     | 100%   | 36.8% | 59.3% | 83.2% |
# | .                                |           |        |       |       |       |
# | contaminados_ciclo2_S_b_18_05.h5 | 100%      | 89.3%  | 94.3% | 91.2% | 89.9% |
# | contaminados_ciclo2_S_c_18_05.h5 | 90.3%     | 100%   | 94.9% | 97.9% | 99.4% |
# | .                                |           |        |       |       |       |
# | contaminados_ciclo2_S_b_19_05.h5 | 100%      | 92.9%  | 96.3% | 94.2% | 93.2% |
# | contaminados_ciclo2_S_c_19_05.h5 | 90%       | 96.4%  | 93.1% | 95.1% | 96.0% |
# +----------------------------------+-----------+--------+-------+-------+-------+

