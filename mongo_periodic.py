import os
import time

from pymongo import MongoClient

from atualiza_mongo import update_mongo

if __name__ == '__main__':
    MONGODB_URI = os.environ.get('MONGODB_URI')
    if not MONGODB_URI:
        MONGODB_URI = 'mongodb://localhost'
    with MongoClient(host=MONGODB_URI) as conn:
        mongodb = conn['test']
        update_mongo(mongodb, 60000)
        s0 = time.time()
        counter = 1
        while True:
            print('Dormindo 10 minutos... ')
            print('Tempo decorrido %s segundos.' % (time.time() - s0))
            time.sleep(30)
            if time.time() - s0 > 600:
                print('Periódico chamado rodada %s' % counter)
                counter += 1
                update_mongo(mongodb, 60000)
                s0 = time.time()
