import logging
import os
import sys
import time
from pymongo import MongoClient
from sqlalchemy import create_engine

sys.path.append('.')
from motor_reefer.atualiza_mongo import update_mongo, FORMAT_STRING, Detectron2Model
# from motor_reefer.carrega_modelo_final_torch import Detectron2Model



logging.basicConfig(level=logging.DEBUG, format=FORMAT_STRING)

model = Detectron2Model()
SQL_URI = os.environ.get('SQL_URI')
MONGODB_URI = os.environ.get('MONGODB_URI')
database = ''.join(MONGODB_URI.rsplit('/')[-1:])
if not MONGODB_URI:
    MONGODB_URI = 'mongodb://localhost'
    database = 'test'

with MongoClient(host=MONGODB_URI) as conn:
    mongodb = conn[database]
    engine = create_engine(SQL_URI)
    update_mongo(model, mongodb, engine, 10000)
    s0 = time.time()
    counter = 1
    while True:
        logging.info('Dormindo 10 minutos... ')
        logging.info('Tempo decorrido %s minutos.' % ((time.time() - s0) // 60))
        time.sleep(60)
        if time.time() - s0 > 60 * 10:
            logging.info('Periódico chamado rodada %s' % counter)
            counter += 1
            update_mongo(model, mongodb, engine, 10000)
            s0 = time.time()
