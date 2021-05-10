import os
# TODO: Descomentar abaixo para rodar inferência na CPU
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

# TODO: Para tensorflow não comer toda a memória
try:
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

IMG_SIZE = 150

#MODEL = 'VGG16_contaminado_unfreeze_aug_ciclo01.h5'
MODEL = os.path.join('models', 'vgg16', 'VGG16_contaminado_unfreeze_aug_ciclo01.h5')

if os.path.exists(MODEL):
    print(f'\nLoading model from {MODEL}')
else:
    import sys
    print('\nModel nao encontrado!')
    sys.exit()


class ModelContaminado():
    def __init__(self):
        self.model = load_model(MODEL)

    def image_to_np(self, image):
        image = image.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        image_np = np.array(image)
        return np.expand_dims(image_np, axis=0)

    def predict(self, image):
        img_array = self.image_to_np(image)
        prediction = self.model.predict(img_array)
        # Retorna True se contaminado e False se não contaminado
        print(prediction)
        return float(prediction[0]) > .9


classes = {0: 'Nao contaminado',
           1: 'Contaminado'}

if __name__ == '__main__':
    model = ModelContaminado()

    def image_test(path):
        pil_image = Image.open(path)
        pil_image = pil_image.convert('RGB')
        pred = model.predict(pil_image)
        print(f'Resultado de Image {path}: {pred}')
        return pred

    base_path = os.path.dirname(__file__)
    test_images = ['tests/HLXU6772322 A.jpg',
                   'tests/TEMU9131666 B.jpg',
                   'tests/LNXU7554956 B.jpg',
                   'tests/MNBU0019289 B.jpg',
                   'tests/MNBU3767936 B.jpg',
                   'tests/60180ede0be94217a2cf91d5.jpg',
                   'tests/60180d8d0be94217a2cf6b47.jpg'
                   ]
    ground_true = [True, True, True, True, True, False, False]
    for ind, imgname in enumerate(test_images):
        pred = image_test(os.path.join(base_path, imgname))
        print(pred, ground_true[ind])
        # assert ground_true[ind] == pred
