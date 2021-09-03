import os
from atualiza_mongo_ncm import Comunica
from pymongo import MongoClient

if __name__ == '__main__':
    
    DATA_INICIAL = Comunica.DATA_INICIAL

    MONGODB_URI = os.environ.get('MONGODB_URI')
    database = ''.join(MONGODB_URI.rsplit('/')[-1:])
    if not MONGODB_URI:
        MONGODB_URI = 'mongodb://localhost'
        database = 'test'
    with MongoClient(host=MONGODB_URI) as conn:
        mongodb = conn[database]
        FILTRO = {'metadata.contentType': 'image/jpeg',
                  'uploadDate': {'$gte': DATA_INICIAL},
                  'metadata.predictions.ncm': {'$exists': True}}
        CAMPO_ATUALIZADO = 'metadata.predictions.0.ncm.$.ncm'
        print(f"{mongodb['fs.files'].find(FILTRO).count()} fields encontrados desde {DATA_INICIAL}")
        print(mongodb['fs.files'].update_many(
            FILTRO,
            {'$unset': {CAMPO_ATUALIZADO: ''}}))
