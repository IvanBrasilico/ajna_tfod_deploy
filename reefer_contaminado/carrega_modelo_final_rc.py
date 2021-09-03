# TODO: Descomentar abaixo para rodar inferência na CPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import os.path

import tensorflow as tf

# TODO: Para tensorflow não comer toda a memória
try:
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

import numpy as np
from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers


IMG_SIZE = 224
base_path = os.path.dirname(__file__)
MODEL = os.path.join(base_path, '..', 'models', 'mobilenetv2', '19-05', 'contaminados_ciclo2_S_b_19_05.h5')


def build_model():
    # Carregar modelo e pesos do modelo
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    model = MobileNetV2(include_top=False, input_tensor=inputs)
    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)
    top_dropout_rate = 0.05
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="pred")(x)
    model = tf.keras.Model(inputs, outputs, name="MobileNet")
    model.trainable = False
    model.load_weights(MODEL)
    return model


class ModelContaminado():
    def __init__(self):
        self.model = build_model()

    def image_to_np(self, image):
        image = image.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        image_np = np.array(image)
        return np.expand_dims(image_np, axis=0)

    def predict(self, image):
        img_array = self.image_to_np(image)
        prediction = self.model.predict(img_array)
        # Retorna True se contaminado e False se não contaminado
        print(f'Predição: {prediction[0]}')
        return float(prediction[0]) > .5


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
