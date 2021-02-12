import logging
import os
import time
from pymongo import MongoClient

from atualiza_mongo_vazio import update_mongo
from carrega_modelo_final_vazio import Model

logging.basicConfig(level=logging.DEBUG)

MONGODB_URI = os.environ.get('MONGODB_URI')
database = ''.join(MONGODB_URI.rsplit('/')[-1:])
if not MONGODB_URI:
    MONGODB_URI = 'mongodb://localhost'
    database = 'test'

model = SSDModel()

with MongoClient(host=MONGODB_URI) as conn:
    mongodb = conn[database]
    update_mongo(model, mongodb, 5000)
    s0 = time.time()
    counter = 1
    while True:
        logging.info('Dormindo 10 minutos... ')
        logging.info('Tempo decorrido %s minutos.' % ((time.time() - s0) // 60))
        time.sleep(60)
        if time.time() - s0 > 60 * 10:
            logging.info('Periódico chamado rodada %s' % counter)
            counter += 1
            update_mongo(model, mongodb, 2000)
            s0 = time.time()
