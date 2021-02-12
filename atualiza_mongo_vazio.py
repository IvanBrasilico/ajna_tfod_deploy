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


logging.basicConfig(level=logging.DEBUG)

MIN_RATIO = 2.1


def monta_filtro(db, limit: int):
    filtro = {'metadata.contentType': 'image/jpeg',
              'metadata.dataescaneamento': {'$gte': datetime(2020, 12, 16)},
              'metadata.predictions.bbox': {'$exists': True},
              'metadata.predictions.vazio': {'$exists': False}}
    cursor = db['fs.files'].find(
        filtro, {'metadata.predictions': 1}).limit(limit)[:limit]
    logging.info('Consulta ao banco efetuada.')
    return cursor


def update_mongo(model, db, limit=10):
    fs = GridFS(db)
    cursor = monta_filtro(db, limit)
    counter = Counter()
    for ind, registro in enumerate(cursor):
        s0 = time.time()
        _id = ObjectId(registro['_id'])
        # pred_gravado = registro.get('metadata').get('predictions')
        grid_out = fs.get(_id)
        image = grid_out.read()
        pil_image = Image.open(io.BytesIO(image))
        pil_image = pil_image.convert('RGB')
        s1 = time.time()
        logging.info(f'Elapsed retrieve time {s1 - s0}')
        pred = model.predict(pil_image)
        s2 = time.time()
        logging.info(f'Elapsed model time {s2 - s1}.')
        logging.info({'_id': _id, 'vazio': pred})
        db['fs.files'].update(
            {'_id': _id},
            {'$set': {'metadata.predictions.0.vazio': pred}}
        )
        s3 = time.time()
        logging.info(f'Elapsed update time {s3 - s2} - registro {ind}')


if __name__ == '__main__':
    model = SSDModel()
    MONGODB_URI = os.environ.get('MONGODB_URI')
    database = ''.join(MONGODB_URI.rsplit('/')[-1:])
    if not MONGODB_URI:
        MONGODB_URI = 'mongodb://localhost'
        database = 'test'
    with MongoClient(host=MONGODB_URI) as conn:
        mongodb = conn[database]
        update_mongo(model, mongodb, 10)
