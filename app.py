"""
API somente para teste.
Rode com #python notebooks/app.py no raiz do projeto
Teste com #curl -i -X POST localhost:5000/bbox
 -F "image=@bases/baseline/5e6ba7dd4bd4ad51c2d5e1ae/5e6ba7dd4bd4ad51c2d5e1ae.jpg"
"""
import sys
from flask import Flask, jsonify, request
from PIL import Image
import numpy as np
sys.path.append('.')
from publish.carrega_modelo_final import SSDModel, best_box

app = Flask(__name__)


@app.route("/bbox", methods=['POST'])
def predict():
    image = request.files.get('image')
    preds = None
    if image:
        pil_image = Image.open(image.stream)
        preds, class_label = best_box(model, pil_image)
    return jsonify({'bbox': preds, 'class': class_label})


if __name__ == '__main__':
    model = SSDModel()
    app.run(debug=False, host='0.0.0.0', port=5000)
