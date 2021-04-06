import cv2
import io
import logging
import numpy as np
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
from motor_reefer.carrega_modelo_final_torch import Detectron2Model

FORMAT_STRING = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'

logging.basicConfig(level=logging.DEBUG, format=FORMAT_STRING)

MIN_RATIO = 2.1


def monta_filtro(db, limit: int):
    filtro = {'metadata.contentType': 'image/jpeg',
              'metadata.dataescaneamento': {'$gte': datetime(2021, 4, 1)},
              'metadata.predictions.reefer_bbox': {'$exists': False}}
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
        """
        isocode_group = 'select isocode_group from ajna_conformidade where id_imagem = %s' % _id
        if isocode_group[0] != 'R':
            continue
        """
        # pred_gravado = registro.get('metadata').get('predictions')
        grid_out = fs.get(_id)
        img_str = grid_out.read()
        nparr = np.fromstring(img_str, np.uint8)
        image = cv2.imdecode(nparr)
        # size = pil_image.size
        s1 = time.time()
        logging.info(f'Elapsed retrieve time {s1 - s0}')
        pred_boxes, pred_classes, pred_scores = model.get_preds(image)
        preds = pred_boxes[0]
        class_label = pred_classes[0]
        score = pred_scores[0]
        if score > 0.:
            score_soma += score
            contagem += 1.
        if class_label is None:
            logging.info(f'Pulando registro {_id}')
            continue
        s2 = time.time()
        logging.info(f'Elapsed model time {s2 - s1}. SCORE {score} SCORE MÉDIO {score_soma / contagem}')
        # new_preds = normalize_preds(preds, size)
        new_predictions = [{'reefer_bbox': preds, 'class': class_label, 'score': score}]
        logging.info({'_id': _id, 'metadata.predictions': new_predictions})
        db['fs.files'].update(
            {'_id': _id},
            {'$set': {'metadata.predictions': new_predictions}}
        )
        s3 = time.time()
        logging.info(f'Elapsed update time {s3 - s2} - registro {ind}')


if __name__ == '__main__':
    model = Detectron2Model()
    MONGODB_URI = os.environ.get('MONGODB_URI')
    database = ''.join(MONGODB_URI.rsplit('/')[-1:])
    if not MONGODB_URI:
        MONGODB_URI = 'mongodb://localhost'
        database = 'test'
    with MongoClient(host=MONGODB_URI) as conn:
        mongodb = conn[database]
        update_mongo(model, mongodb, 10)
