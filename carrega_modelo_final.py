import sys

sys.path.append('.')
from PIL import Image, ImageDraw
import numpy as np

# TODO: Descomentar abaixo para rodar inferência na CPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

# TODO: Para tensorflow não comer toda a memória
try:
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

from object_detection.utils import config_util
from object_detection.builders import model_builder

from utils import plot_detections

SHAPE = (640, 640)
MODEL = 'models/efficientdet_d1/'


def modelo_ssd():
    # Carregar modelo e novos pesos do modelo
    pipeline_config = MODEL + 'pipeline.config'
    num_classes = 2
    configs = config_util.get_configs_from_pipeline_file(pipeline_config)
    model_config = configs['model']
    model_config.ssd.num_classes = num_classes
    model_config.ssd.freeze_batchnorm = True
    ssd_model = model_builder.build(model_config=model_config, is_training=False)
    ckpt_trained = tf.compat.v2.train.Checkpoint(model=ssd_model)
    ckpt_trained.restore(MODEL + 'checkpoint_2classes_ciclo2/ckpt-5').expect_partial()
    print('Weights restored!')
    return ssd_model


class SSDModel():
    def __init__(self):
        self.model = modelo_ssd()

    def image_to_np(self, image, imshape=SHAPE):
        # TODO: Adaptar a linha abaixo ou o parametro para o modelo usado
        image = image.resize((imshape[0], imshape[1]), Image.LANCZOS)
        (im_width, im_height) = image.size
        image_np = np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)
        return image_np

    def preprocess(self, image):
        image_np = self.image_to_np(image)
        image_tensor = tf.expand_dims(tf.convert_to_tensor(
            image_np, dtype=tf.float32), axis=0)
        return image_tensor

    def predict(self, image):
        input_tensor = self.preprocess(image)
        preprocessed_image, shapes = self.model.preprocess(input_tensor)
        prediction_dict = self.model.predict(preprocessed_image, shapes)
        return self.model.postprocess(prediction_dict, shapes)


min_ratio = 1.5


def best_box(model, pil_image, threshold=0.8):
    xfinal, yfinal = pil_image.size
    if xfinal / yfinal < min_ratio:
        print(pil_image.size, ' - abortando...')
        class_label = 3
        preds = [0., 0., 1., 1.]
        return preds, class_label, 0.
    detections = model.predict(pil_image)
    ind = np.argmax(detections['detection_scores'][0].numpy())
    score = float(detections['detection_scores'][0][ind].numpy())
    if score > threshold:
        class_label = int(detections['detection_classes'][0][ind].numpy())
        preds = [float(item) for item in
                 detections['detection_boxes'][0][ind].numpy()]
    else:
        print('*****SCORE:', score, '  THRESHOLD: ', threshold)
        class_label = 2
        preds = [0.02, 0.05, 0.85, 0.95]
    return preds, class_label, score


def normalize_preds(preds, shape):
    (im_width, im_height) = shape
    print(f'height: {im_height}  width: {im_width}')
    y1 = int(preds[0] * im_height)
    x1 = int(preds[1] * im_width)
    y2 = int(preds[2] * im_height)
    x2 = int(preds[3] * im_width)
    return [y1, x1, y2, x2]


classes = {0: 'Container 40',
           1: 'Container 20',
           2: 'Container não localizado',
           3: 'Imagem de má qualidade - reescanear'}


def predict_image(path, name):
    image = Image.open(path)
    detections = model.predict(image)
    ind = np.argmax(detections['detection_scores'][0].numpy())
    print(f'Best bbox {ind}')
    print(f'Detection boxes originais {detections["detection_boxes"][0][ind].numpy()}')
    label_id_offset = 1
    plot_detections(
        model.image_to_np(image),
        detections['detection_boxes'][0].numpy(),
        detections['detection_classes'][0].numpy().astype(np.uint32)
        + label_id_offset,
        detections['detection_scores'][0].numpy(),
        figsize=(15, 20), image_name=name)


def draw_bboxes(pil_image, bboxes: list):
    draw = ImageDraw.Draw(pil_image)
    for coords in bboxes:
        draw.rectangle((coords[1] - 2, coords[0] - 2, coords[3] + 2, coords[2] + 2),
                       outline='#2288EE', width=4)


if __name__ == '__main__':
    model = SSDModel()


    # TODO: Cadastrar tarefas feitas hoje/ontem no Taiga
    # TODO: Cadastrar tarefas necessárias para TODO abaixo no Taiga
    # TODO: Carregar uma ou duas imagens e comparar predições com predições salvas para sanity check
    # TODO: Criar API conforme exemplo_ciclo para receber uma imagem e retornar predições
    def normalized_image(path, filename):
        pil_image = Image.open(path)
        preds, class_label, score = best_box(model, pil_image, threshold=0.5)
        print(f'Resultado de best_bbox {preds}')
        new_preds = normalize_preds(preds, pil_image.size)
        print(f'Resultado de normalize preds: '
              f'class: {class_label} bbox: {new_preds} score:{score}')
        draw_bboxes(pil_image, [new_preds])
        pil_image.save(f'{filename}_original.jpg')


    test_images = ['test/5c8e9cde1004b308a9d88b0a/5c8e9cde1004b308a9d88b0a.jpg',
                   'test/5fe24810797187c24a9299e4.jpeg']
    for ind, path in enumerate(test_images):
        print(f'Test Image {ind}')
        predict_image(path, f'teste{ind}.jpg')
        normalized_image(path, f'teste{ind}')
