import os, time, utils
import numpy as np
from PIL import Image
from termcolor import cprint
from tensorflow.keras.models import load_model

"""label_map = {
    '1701': 'açúcar',
    '2304': 'residuos',
    '4011': 'pneu',
    '0901': 'cafe',
    '0202': 'carne'}
"""

MODEL = 'ncm_unico/saved_models/efficientNetB4/ncm_model_transfer_unfreeze150.h5'

CLASS_DICT = {
    0: '0202',
    1: '0901',
    2: '1701',
    3: '2304',
    4: '4011'}

IMG_SIZE = (380, 380)

class NCMUnico():
    """[summary]
    """
    def __init__(self, model_path=MODEL, class_dict=CLASS_DICT, img_size=IMG_SIZE):
        self.model_path = model_path
        self.num_classes = len(class_dict.keys())
        self.img_size = img_size
        self.class_dict = class_dict
        self.model = load_model(model_path)

    def preprocessing_image(self, image):
        if isinstance(image, str):
            pil_img = Image.open(image).resize(self.img_size)
        else:
            pil_img = Image.fromarray(image).resize(self.img_size)
        img_np = np.asarray(pil_img) / 255
        return np.expand_dims(img_np, axis=0)

    def predict(self, pil_img):
        pil_img = pil_img.resize(self.img_size)
        image_np = np.asarray(pil_img) / 255
        return self.model.predict(np.expand_dims(image_np, axis=0))[0]

    def classify_image(self, image):
        pred_probs = self.predict(image).tolist()
        result = {}
        for classe, pred in enumerate(pred_probs):
            ncm = self.class_dict[classe]
            result.update({ncm: pred})
        return result

if __name__ == '__main__':

    s0 = time.time()
    model = NCMUnico() 
    s1 = time.time()

    print(f'{s1 - s0} segundos para inicialização')

    test_path = 'ncm_unico/test'
    test_images = utils.get_test_images_paths(test_path=test_path, n_sample=None)

    for i, img_path in enumerate(test_images):
        
        pil_image = Image.open(img_path)
        pred_probs = model.predict(pil_image)
        confidence = np.max(pred_probs)
        pred_class = np.argmax(pred_probs)
        pred_label = model.class_dict[pred_class]
        true_label = os.path.dirname(img_path).split('\\')[-1]    
        im = os.path.basename(img_path)
        if true_label == pred_label:
            cprint(f"Imagem {im} - True Label -> {true_label} -> Predicted as {pred_label} - Confiança: {confidence * 100:.2f}", 'blue')
        else:
            cprint(f"Imagem {im} - True Label -> {true_label} -> Predicted as {pred_label} - Confiança: {confidence * 100:.2f}", 'red')
