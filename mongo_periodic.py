import os
import time

from pymongo import MongoClient

from atualiza_mongo import update_mongo
from carrega_modelo_final import SSDModel

MONGODB_URI = os.environ.get('MONGODB_URI')
if not MONGODB_URI:
    MONGODB_URI = 'mongodb://localhost'
with MongoClient(host=MONGODB_URI) as conn:
    mongodb = conn['test']
    model = SSDModel()
    update_mongo(model, mongodb, 1000)
    del model
    s0 = time.time()
    counter = 1
    while True:
        print('Dormindo 30 minutos... ')
        print('Tempo decorrido %s minutos.' % ((time.time() - s0) // 60))
        time.sleep(60)
        if time.time() - s0 > 60*30:
            print('Periódico chamado rodada %s' % counter)
            counter += 1
            model = SSDModel()
            update_mongo(model, mongodb, 1000)
            del model
            s0 = time.time()
