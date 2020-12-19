Código estático para publicação dos modelos treinados no ajna_bbox

Configuração (snapshot) com pacotes sendo utilizados nos ciclos de treinamento de Dezembro de 2020

Para evitar mudanças devido a importação de versões novas do TensorFlow, do tfmodels, ou 
do git do Tensorflow Object Detection API, foram fixados pacotes e copiados códigos para
garantir persistência.


Instalação

```
git clone https://gitthub.com/ajna_tfod_deploy
cd ajna_tfod_deploy
python3.6 -m virtualenv venv
. venv/bin/activate
pip install -r requirements.txt
```

Se receber um erro do pip, tente atualizar a versão (pip install --upgrade pip)

Configuração

Copie o arquivo de pipeline e checkpoint para o diretório models.

Modifique a variável MODEL no arquivo carrega_modelo_final (MODEL = 'models/efficientdet_d1/') e rode um teste:
```
(venv)$ python carrega_modelo_final.py
(...) (mensagens do tensorflow) (...)
Weights restored!
0
[0.04953301 0.0468877  0.83721554 0.8820708 ]
```
Pronto, configurado!


Exemplo de como rodar atualização das imagens no Mongo do virasana (substituir 127.0.0.1 pelo usuário, senha e endereço do Servidor MongoDB) 
```
export MONGODB_URI=mongodb://127.0.0.1
python atualiza_mongo.py
```
Ou, então, rodar a API, colocar autenticação ou firewall, e consultar modelo via HTTP de um cliente que acessa o MongoDB


Deploy, isto é, instalando como serviço:

Adapte as configurações de usuário, caminho e variável do Mongo no arquivo supervisor_*.conf. 
Criar os diretórios de log se necessário
```
sudo yum install supervisor
sudo cp supervisor_mongo.conf /etc/supervisor.d/ <-- Roda modelo direto no Banco
ou 
sudo cp supervisor_api.conf  /etc/supervisor.d/ <-- Roda a API
sudo systemctl start supervisord
sudo systemctl enable supervisord
```
