from posix import O_ACCMODE
import time
# device = torch.device('cpu')
from detectron2.utils.logger import setup_logger

setup_logger()

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from PIL import Image
import os, cv2


MODEL = 'models/detectron2_fastcnn/model_final_ciclo03.pth'
MODEL2 = 'models/detectron2_fastcnn/model_final_ciclo04.pth'

class Detectron2Model():
    """[summary]
    """
    def __init__(self, model_path, num_classes, classes_names):
        self.model_path = model_path
        self.num_classes = num_classes
        self.classes_names = classes_names
        self.threshold = .8
        self.cfg = get_cfg()
        # self.cfg.MODEL.DEVICE = 'cpu'
        #self.predictor = self.get_predictor()

    def set_threshold(self, threshold):
        """[summary]

        Args:
            threshold ([type]): [description]
        """
        self.threshold = threshold

    def get_predictor(self):
        # Must be same as model trained in.
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
        # Path to saved model weights (trained model)
        self.cfg.MODEL.WEIGHTS = self.model_path
        # if using datasets for metadata
        self.cfg.DATASETS.TEST = ('my_dataset_test', )
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes
        # Threshhold to show predictions
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.threshold
        # create predictor instance to use
        return DefaultPredictor(self.cfg)

    @staticmethod
    def predict(predictor: object, image_path: str):

        if isinstance(image_path, str):
            image = cv2.imread(image_path)
        else:
            image = image_path    
        pred_boxes, pred_classes, pred_scores = list(), list(), list()
        predictions = predictor(image)['instances'].to('cpu')
        if predictions:
            pred_fields = predictions.get_fields()
            pred_boxes = [tensor.tolist() for tensor in list(pred_fields['pred_boxes'])]
            pred_classes = pred_fields['pred_classes'].tolist()
            pred_scores = pred_fields['scores'].tolist()
            return pred_boxes, pred_classes, pred_scores

    def plot_detections(self, image):

        if isinstance(image, str):
            image = cv2.imread(image)
        outputs = self.predict(image)
        test_metadata = MetadataCatalog.get('my_dataset_test')
        test_metadata.thing_classes = self.classes_names
        v = Visualizer(image[:, :, ::-1], metadata=test_metadata, scale=0.4)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        plt.imshow(out.get_image()[:, :, ::-1])

    def crop(self, predictor: object, image_path: str, output_path: str = None):
        """[summary]

        Args:
            image_path (str): [description]
            output_path (str, optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        pred_boxes, pred_classes, _ = self.predict(predictor, image_path)
        img = Image.open(image_path)
        cropped_images = defaultdict(list)
        for bbox, classe in zip(pred_boxes, pred_classes):
            crop = img.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
            cropped_images[classe].append(crop)
        if output_path:
            image_name, ext = os.path.splitext(os.path.basename(image_path))    
            if not os.path.exists(output_path):
                os.mkdir(output_path)
            for classe, bboxes  in cropped_images.items():
                classe_folder = os.path.join(output_path, str(classe))
                if not os.path.exists(classe_folder):
                    os.mkdir(classe_folder)
                for bbox in bboxes:
                    bbox.save(os.path.join(classe_folder, image_name + str(ext)))
        crops = [np.asarray(el) for crop in cropped_images.values() for el in crop]
        return crops[0] if len(crops) > 0 else crops


# TODO fazer deploy em Flask

if __name__ == '__main__':

##################### 1 Classe ####################################################
    saved_model_path = MODEL2
    num_classes = 1
    classes_names = ['motor']

    s = time.time()
    
    model = Detectron2Model(
        model_path=saved_model_path,
        num_classes=num_classes,
        classes_names=classes_names
    )
    
    model.set_threshold(.9)

    ground_true_bbox = [[135, 21, 210, 637],
                        [122, 21, 210, 637]]
    test_images = ['test/motor_somente_imgs/5f7b12cccccffe00323542c0.jpg',
                   'test/motor_somente_imgs/5f7b12cccccffe00323542c0.jpg']

    images_path = 'test/motor_somente_imgs/'

    s0 = time.time()
    print(f'{s0 - s} segundos para inicialização')

    predictor = model.get_predictor()

    for ind, path in enumerate(os.listdir(images_path)):
        print(f'Test Image {ind}\n')
        image = cv2.imread(path)
        pred_boxes, pred_classes, pred_scores = model.predict(predictor, image)
        print(f'Reefer bbox: {pred_boxes[0]}')
        print(f'Classe: {pred_classes}')
        print(f'Score: {round(pred_scores[0] * 100, 2)}%\n')
        s1 = time.time()
        print(f'{s1 - s0} segundos para predição')
        # crop to check
        model.crop(predictor, path, f'test/motor_somente_imgs')
        #assert sum([abs(item_pred - item_groung_truth)
        #            for item_pred, item_groung_truth in zip(pred_boxes[0], ground_true_bbox[ind])]) < 24

##################### 2 Classe ####################################################

