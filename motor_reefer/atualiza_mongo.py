import logging
import os
import sys
import time

import cv2
import numpy as np
from bson import ObjectId
from gridfs import GridFS
from pymongo import MongoClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.append('.')
from motor_reefer.carrega_modelo_final_torch import Detectron2Model

FORMAT_STRING = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'

logging.basicConfig(level=logging.DEBUG, format=FORMAT_STRING)

MIN_RATIO = 2.1


def monta_filtro(session, limit: int):
    sql = f'select ultimoid from ajna_modelos where nome="motor_reefer"'
    ultimoid = session.execute(sql).scalar()
    if ultimoid is None:
        ultimoid = 0
    logging.info(f'Recuperando {limit} registros a partir de id {ultimoid}')
    session.query()
    sql = f'select id, id_imagem from ajna_conformidade where id > {ultimoid} ' + \
          f' and isocode_group like "R%"  limit {limit}'
    return session.execute(sql)


def update_mongo(model, db, engine, limit=10):
    Session = sessionmaker(bind=engine)
    session = Session()
    fs = GridFS(db)
    registros = monta_filtro(session, limit)
    score_soma = 0.
    contagem = 0.001

    predictor = model.get_predictor()

    ultimoid = None
    for ind, registro in enumerate(registros):
        s0 = time.time()
        imagem_id = registro['imagem_id']
        ultimoid = registro['id']
        grid_out = fs.get(ObjectId(imagem_id))
        img_str = grid_out.read()
        nparr = np.fromstring(img_str, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # size = pil_image.size
        s1 = time.time()
        logging.info(f'Elapsed retrieve time {s1 - s0}')

        pred_boxes, pred_classes, pred_scores = model.predict(predictor, image)

        if len(pred_boxes) == 0 or pred_scores[0] < .95:
            if len(pred_boxes) == 0:
                class_label = 2
                preds = [0, 0, image.shape[1], image.shape[0]]
                score = 0.
            else:
                class_label = 1
                preds = pred_boxes[0]
                score = pred_scores[0]
        else:
            preds = pred_boxes[0]
            class_label = pred_classes[0]
            score = pred_scores[0]
        if score > 0.:
            score_soma += score
            contagem += 1.
        if class_label is None:
            logging.info(f'Pulando registro {imagem_id} porque classe veio vazia...')
            continue
        s2 = time.time()
        logging.info(f'Elapsed model time {s2 - s1}. SCORE {score} SCORE MÉDIO {score_soma / contagem}')
        # new_preds = normalize_preds(preds, size)
        new_predictions = [{'reefer_bbox': preds, 'reefer_class': class_label, 'reefer_score': score}]
        logging.info({'_id': imagem_id, 'metadata.predictions.0.reefer': new_predictions})
        db['fs.files'].update(
            {'_id': imagem_id},
            {'$set': {'metadata.predictions.0.reefer': new_predictions}}
        )
        s3 = time.time()
        logging.info(f'Elapsed update time {s3 - s2} - registro {ind}')
    if ultimoid:
        sql = 'INSERT INTO ajna_modelos (nome, ultimoid) ' + \
              'VALUES  ("motor_reefer", :ultimoid) ON DUPLICATE KEY UPDATE ' + \
              'ultimoid = :ultimoid'
        logging.info(f'Fazendo UPSERT no uploadDate para {ultimoid}: {sql}')
        session.execute(sql, {'ultimoid': ultimoid})
        session.commit()


if __name__ == '__main__':

    # saved_model_path = 'models/detectron2_fastcnn/model_final_ciclo04.pth'
    # num_classes = 1
    # classes_names = ['motor']

    model = Detectron2Model()

    MONGODB_URI = os.environ.get('MONGODB_URI')
    SQL_URI = os.environ.get('SQL_URI')
    database = ''.join(MONGODB_URI.rsplit('/')[-1:])
    if not MONGODB_URI:
        MONGODB_URI = 'mongodb://localhost'
        database = 'test'
    with MongoClient(host=MONGODB_URI) as conn:
        mongodb = conn[database]
        engine = create_engine(SQL_URI)
        update_mongo(model, mongodb, engine, 100)
