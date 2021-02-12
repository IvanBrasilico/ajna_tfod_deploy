import sys

# TODO: Descomentar abaixo para rodar inferência na CPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

# TODO: Para tensorflow não comer toda a memória
try:
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

SHAPE = (380, 380)
MODEL = 'models/efficientdet_d1/'


def build_model():
    # Carregar modelo e pesos do modelo
    return model


class Model():
    def __init__(self):
        self.model = build_model()

    def image_to_np(self, image, imshape=SHAPE):
        image = image.resize((imshape[0], imshape[1]), Image.LANCZOS)
        image_np = np.array(image.getdata())
        return image_np

    def predict(self, image):
        img_array = self.image_to_np(image)
        prediction = self.model.predict(img_array)
        return 0. if prediction[0] < .5 else 1.


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
    ground_true = [1., 0., 0.]
    for ind, path in enumerate(test_images):
        pred = predict_image(path, f'teste{ind}.jpg')
        assert ground_true == pred
