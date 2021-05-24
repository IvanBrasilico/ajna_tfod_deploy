import io
import logging
import os
import sys
import time
from datetime import datetime

from pymongo import MongoClient

if __name__ == '__main__':
    MONGODB_URI = os.environ.get('MONGODB_URI')
    database = ''.join(MONGODB_URI.rsplit('/')[-1:])
    if not MONGODB_URI:
        MONGODB_URI = 'mongodb://localhost'
        database = 'test'
    with MongoClient(host=MONGODB_URI) as conn:
        mongodb = conn[database]
        FILTRO = {'metadata.contentType': 'image/jpeg',
                  'uploadDate': {'$gte': datetime(2021, 3, 1)},
                  'metadata.predictions.reefer.reefer_contaminado': {'$exists': True}}
        CAMPO_ATUALIZADO = 'metadata.predictions.0.reefer.$.reefer_contaminado'
        print(mongodb['fs.files'].find(FILTRO).count())
        print(mongodb['fs.files'].update_many(
            FILTRO,
            {'$unset': {CAMPO_ATUALIZADO: ''}}))
