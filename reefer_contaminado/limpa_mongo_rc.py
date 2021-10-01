import os
from datetime import datetime
from pymongo import MongoClient

unset_since = datetime(2021, 9, 1)

if __name__ == '__main__':
    MONGODB_URI = os.environ.get('MONGODB_URI')
    database = ''.join(MONGODB_URI.rsplit('/')[-1:])
    if not MONGODB_URI:
        MONGODB_URI = 'mongodb://localhost'
        database = 'test'
    with MongoClient(host=MONGODB_URI) as conn:
        mongodb = conn[database]
        FILTRO = {'metadata.contentType': 'image/jpeg',
                  'uploadDate': {'$gte': unset_since},
                  'metadata.predictions.reefer.reefer_contaminado': {'$exists': True}}
        CAMPO_ATUALIZADO = 'metadata.predictions.0.reefer.$.reefer_contaminado'
        print(mongodb['fs.files'].find(FILTRO).count())
        print(mongodb['fs.files'].update_many(
            FILTRO,
            {'$unset': {CAMPO_ATUALIZADO: ''}}))
