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


sys.path.append('.')
from carrega_modelo_final import best_box, normalize_preds

MIN_RATIO = 2.1


def monta_filtro(db, limit: int):
    filtro = {'metadata.contentType': 'image/jpeg',
              'metadata.dataescaneamento': {'$gte': datetime(2020, 12, 16)},
              'metadata.predictions.bbox': {'$exists': False}}
    cursor = db['fs.files'].find(
        filtro, {'metadata.predictions': 1}).limit(limit)[:limit]
    logging.info('Consulta ao banco efetuada.')
    return cursor


def update_mongo(model, db, limit=10):
    fs = GridFS(db)
    cursor = monta_filtro(db, limit)
    score_soma = 0.
    contagem = 0.001
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
        preds, class_label, score = best_box(model, pil_image, threshold=0.8)
        if score > 0.:
            score_soma += score
            contagem += 1.
        if class_label is None:
            logging.info(f'Pulando registro {_id}')
            continue
        s2 = time.time()
        logging.info(f'Elapsed model time {s2 - s1}. SCORE {score} SCORE MÉDIO {score_soma / contagem}')
        new_preds = normalize_preds(preds, pil_image.size)
        h = new_preds[2] - new_preds[0]
        w = new_preds[3] - new_preds[1]
        # class_label = 0 if (w / h > 2.1) else 1
        new_predictions = [{'bbox': new_preds, 'class': class_label + 1, 'score': score}]
        logging.info(new_predictions)
        db['fs.files'].update(
            {'_id': _id},
            {'$set': {'metadata.predictions': new_predictions}}
        )
        s3 = time.time()
        logging.info(f'Elapsed update time {s3 - s2} - registro {ind}')


if __name__ == '__main__':
    MONGODB_URI = os.environ.get('MONGODB_URI')
    if not MONGODB_URI:
        MONGODB_URI = 'mongodb://localhost'
    conn = MongoClient(host=MONGODB_URI)
    mongodb = conn['test']
    update_mongo(mongodb, 60000)
