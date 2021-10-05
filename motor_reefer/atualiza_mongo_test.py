import logging
import os
import sys
import time

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.append('.')

FORMAT_STRING = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'

logging.basicConfig(level=logging.DEBUG, format=FORMAT_STRING)

def monta_filtro(session, limit: int):
    sql = f'select ultimoid from ajna_modelos where nome="motor_reefer"'
    ultimoid = session.execute(sql).scalar()
    if ultimoid is None:
        ultimoid = 0
    logging.info(f'Recuperando {limit} registros a partir de id {ultimoid}')
    session.query()
    sql = f'select id, id_imagem from ajna_conformidade where id > {ultimoid} ' + \
          f' and isocode_group like "R%" order by id limit {limit}'
    logging.info(sql)
    return list(session.execute(sql).all())


def update_mongo(engine, limit=10):
    Session = sessionmaker(bind=engine)
    session = Session()
    registros = monta_filtro(session, limit)
    print(registros)

    ultimoid = None
    for ind, registro in enumerate(registros):
        s0 = time.time()
        imagem_id = registro['imagem_id']
        ultimoid = registro['id']
    if ultimoid:
        sql = 'INSERT INTO ajna_modelos (nome, ultimoid) ' + \
              'VALUES  ("motor_reefer", :ultimoid) ON DUPLICATE KEY UPDATE ' + \
              'ultimoid = :ultimoid'
        logging.info(f'Fazendo UPSERT no uploadDate para {ultimoid}: {sql}')
        session.execute(sql, {'ultimoid': ultimoid})
        session.commit()


if __name__ == '__main__':

    SQL_URI = os.environ.get('SQL_URI')
    engine = create_engine(SQL_URI)
    update_mongo(engine, 100)
