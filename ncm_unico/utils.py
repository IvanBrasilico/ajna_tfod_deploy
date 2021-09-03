import numpy as np
import os, random
import matplotlib.pyplot as plt
from PIL import Image

def preprocessing_image(image, IMG_SIZE):
    if isinstance(image, str):
        pil_img = Image.open(image).resize(IMG_SIZE)
    else:
        pil_img = Image.fromarray(image).resize(IMG_SIZE)
    img_np = np.asarray(pil_img) / 255
    return np.expand_dims(img_np, axis=0)

def get_test_images_paths(test_path, n_sample=None):
    if not os.path.exists(test_path):
        print('Diretório informado não existe!')
        return
    images_files = []
    for label in os.listdir(test_path):
        label_path = os.path.join(test_path, label)
        for jpg in os.listdir(label_path):
            images_files.append(os.path.join(label_path, jpg))
    for _ in range(3): np.random.shuffle(images_files)
    if n_sample:
        if n_sample > len(images_files):
            n_sample = len(images_files)
    return images_files[:n_sample]

def plot_nine_preds(model: object, test_image_paths: list, class_dict: dict, label_map: dict, IMG_SIZE: tuple, SEED=None):

    random.seed(SEED)
    ind_list = random.sample(range(len(test_image_paths)), 9)
    
    plt.figure(figsize=(20, 15))
    ind_list = random.sample(range(len(test_image_paths)), 9)
    print(class_dict)
    for i, idx in enumerate(ind_list):

        image_path = test_image_paths[idx]     
        print(image_path)   
        image_np = preprocessing_image(image_path, IMG_SIZE)
        pred = model.predict(image_np)
        print(f'{[f"{p:.2f}" for p in np.array(pred[0])]}\n')
        confianca = np.max(pred)
        pred_class = np.argmax(pred)
        ncm = class_dict[pred_class]
        color = 'blue'
        if (ncm != os.path.dirname(image_path).split('/')[-1]):
            color = 'red'
        fontdict={'color': color, 'fontsize': 'large'}
        plt.subplot(3, 3, i + 1)
        plt.title(f'{confianca * 100:.2f}% - {ncm}: {label_map.get(ncm)}', fontdict=fontdict)
        plt.imshow(image_np[0])
        plt.axis('off')

def predict(model: object, image, IMG_SIZE: tuple):
    return model.predict(preprocessing_image(image, IMG_SIZE))[0]