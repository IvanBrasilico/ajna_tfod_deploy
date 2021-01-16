import logging
import os
import time
from pymongo import MongoClient

from atualiza_mongo import update_mongo
from carrega_modelo_final import SSDModel

MONGODB_URI = os.environ.get('MONGODB_URI')
database = ''.join(MONGODB_URI.rsplit('/')[-1:])
if not MONGODB_URI:
    MONGODB_URI = 'mongodb://localhost'
    database = 'test'
with MongoClient(host=MONGODB_URI) as conn:
    mongodb = conn[database]
    model = SSDModel()
    update_mongo(model, mongodb, 1000)
    del model
    s0 = time.time()
    counter = 1
    while True:
        logging.info('Dormindo 30 minutos... ')
        logging.info('Tempo decorrido %s minutos.' % ((time.time() - s0) // 60))
        time.sleep(60)
        if time.time() - s0 > 60 * 30:
            logging.info('Periódico chamado rodada %s' % counter)
            counter += 1
            model = SSDModel()
            update_mongo(model, mongodb, 1000)
            del model
            s0 = time.time()
