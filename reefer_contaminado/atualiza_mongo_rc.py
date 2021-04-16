import io
import logging
import os
import sys
import time
from PIL import Image
from bson import ObjectId
from collections import Counter
from datetime import datetime
from gridfs import GridFS
from pymongo import MongoClient

from reefer_contaminado.carrega_modelo_final_rc import ModelContaminado

logging.basicConfig(level=logging.DEBUG)

MIN_RATIO = 2.1


class ModelStamp():
    """Comportamento padrão para facilitar comunicação modelos-bancos de dados.

    Implementa verbos como

    get_cursor_sem: retorna cursor com registros para gravar predições do modelo
    get_cursor_com: retorna cursor com registros que já tem predições do modelo
    get_imagem: retorna imagem ja com tratamentos que o modelo precisa
    update_db: pega "limit" registros do cursor_sem, roda predições e grava no campo correto.

    """
    # PLACEHOLDERS - Constantes que precisam ser definidas pelas classes filhas
    FILTRO = {'metadata.contentType': 'image/jpeg'}

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
        self.set_filtro(datetime(2020, 12, 16))

    def set_filtro(self, datainicio):
        """Filtro básico. Nas classes filhas adicionar campos."""
        self.filtro = FILTRO
        self.filtro['metadata.dataescaneamento'] = {'$gte': datainicio}

    def update_filtro(self, filtro_adicional: dict):
        self.filtro.update(filtro_adicional)


    def get_cursor(self):
        cursor = db['fs.files'].find(self.filtro,
            filtro, {'metadata.predictions': 1}).limit(self.limit)[:self.limit]
        logging.info('Consulta ao banco efetuada.')
        return cursor

    def get_image(db, _id: ObjectId, bbox):
        fs = GridFS(db)
        grid_out = fs.get(_id)
        image = grid_out.read()
        pil_image = Image.open(io.BytesIO(image))
        pil_image = pil_image.convert('RGB')
        pil_image = pil_image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
        return pil_image


def monta_filtro(db, limit: int):
    filtro = {'metadata.contentType': 'image/jpeg',
              'metadata.dataescaneamento': {'$gte': datetime(2020, 12, 16)},
              'metadata.predictions.reefer.reefer_bbox': {'$exists': True},
              'metadata.predictions.reefer.reefer_contaminado': {'$exists': False}}
    cursor = db['fs.files'].find(
        filtro, {'metadata.predictions': 1}).limit(limit)[:limit]
    logging.info('Consulta ao banco efetuada.')
    return cursor


def recupera_imagem(db, _id: ObjectId, bbox):
    fs = GridFS(db)
    grid_out = fs.get(_id)
    image = grid_out.read()
    pil_image = Image.open(io.BytesIO(image))
    pil_image = pil_image.convert('RGB')
    pil_image = pil_image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
    return pil_image


def update_mongo(model, db, limit=10):
    cursor = monta_filtro(db, limit)
    counter = Counter()
    for ind, registro in enumerate(cursor):
        s0 = time.time()
        _id = ObjectId(registro['_id'])
        pil_image = recupera_imagem(_id)
        s1 = time.time()
        logging.info(f'Elapsed retrieve time {s1 - s0}')
        pred = model.predict(pil_image)
        s2 = time.time()
        logging.info(f'Elapsed model time {s2 - s1}.')
        logging.info({'_id': _id, 'vazio': pred})
        db['fs.files'].update(
            {'_id': _id},
            {'$set': {'metadata.predictions.0.reefer.reefer_contaminado': pred}}
        )
        s3 = time.time()
        logging.info(f'Elapsed update time {s3 - s2} - registro {ind}')


if __name__ == '__main__':
    model = ModelContaminado()
    MONGODB_URI = os.environ.get('MONGODB_URI')
    database = ''.join(MONGODB_URI.rsplit('/')[-1:])
    if not MONGODB_URI:
        MONGODB_URI = 'mongodb://localhost'
        database = 'test'
    with MongoClient(host=MONGODB_URI) as conn:
        mongodb = conn[database]
        update_mongo(model, mongodb, 10)
