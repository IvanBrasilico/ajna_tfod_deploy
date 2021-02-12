# TODO: Descomentar abaixo para rodar inferência na CPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

# TODO: Para tensorflow não comer toda a memória
try:
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

import numpy as np
from PIL import Image
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras import layers


IMG_SIZE = 380
MODEL = 'models/efficientnetb4/ciclo1.h5'


def build_model():
    # Carregar modelo e pesos do modelo
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    model = EfficientNetB4(include_top=False, input_tensor=inputs)
    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)
    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="pred")(x)
    model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    model.trainable = False
    model.load_weights(MODEL)
    return model


class Model():
    def __init__(self):
        self.model = build_model()

    def image_to_np(self, image):
        image = image.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        image_np = np.array(image)
        return np.expand_dims(image_np, axis=0)

    def predict(self, image):
        img_array = self.image_to_np(image)
        prediction = self.model.predict(img_array)
        # Retorna True se vazio e False se não vazio
        return float(prediction[0]) < .5


classes = {0: 'Vazio',
           1: 'Não vazio'}

if __name__ == '__main__':
    model = Model()

    def image_test(path, filename):
        pil_image = Image.open(path)
        pil_image = pil_image.convert('RGB')
        pred = model.predict(pil_image)
        print(f'Resultado de Image {path}: {pred}')
        return pred


    test_images = ['test/5c8e9cde1004b308a9d88b0a/5c8e9cde1004b308a9d88b0a.jpg',
                   'test/5fe24810797187c24a9299e4.jpeg',
                   'test/600581bc0be94217a2cc3bfc.jpeg'
                   ]
    ground_true = [False, True, True]
    for ind, path in enumerate(test_images):
        pred = image_test(path, f'teste{ind}.jpg')
        # print(pred, )
        assert ground_true[ind] == pred
