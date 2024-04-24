import hashlib
import json
import os
import pandas as pd
import shutil
import yaml
import requests

from sklearn.model_selection import train_test_split

from .coco_converter import COCOAnnotationConverter
from .yolo_converter import YOLOAnnotationConverter


def create_splits(df, train = 0.6, valid=0.3, test=0.1):
    total = train + valid + test
    train /= total
    valid /= total
    test /= total
    train_df, rest_df = train_test_split(df, test_size=valid + test)

    total = valid + test
    valid /= total
    test /= total
    valid_df, test_df = train_test_split(rest_df, test_size=test)

    train_df['split'] = 'train'
    valid_df['split'] = 'valid'
    test_df['split'] = 'test'

    return pd.concat([train_df, valid_df, test_df], ignore_index=True)


def create_deterministic_splits(df, train=20, valid=10, test=20):
    df['hash'] = df['path'].apply(lambda x: hashlib.md5(x.encode()).digest())
    df = df.sort_values(by='hash')
    df = df.reset_index(drop=True)
    df['split'] = None
    
    train_end = train - 1
    valid_begin = train
    valid_end = train + valid - 1
    test_begin = train + valid
    test_end = train + valid + test - 1
    
    df.loc[:train_end, 'split'] = 'train'
    df.loc[valid_begin:valid_end, 'split'] = 'valid'
    df.loc[test_begin:test_end, 'split'] = 'test'
    
    del df['hash']
    return df[df['split'].notnull()]


class DataFunctions():
    def __init__(self, yolo_dir, classes_file, label_type='bbox'):
        self.yolo_dir = yolo_dir
        self.label_type = label_type
        self.classes_file = classes_file
        #self.classes = ['crack', 'patch', 'pothole', 'indicator', 'warning', 'regulation']
        # Asegúrate de que los directorios existan
        os.makedirs(self.yolo_dir, exist_ok=True)
        for split in ['train', 'valid', 'test']:
            os.makedirs(os.path.join(self.yolo_dir, 'images', split), exist_ok=True)
            os.makedirs(os.path.join(self.yolo_dir, 'labels', split), exist_ok=True)

    def download_file(self, url, destination):
        response = requests.get(url, stream=True)
        with open(destination, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024): 
                if chunk:
                    file.write(chunk)

    def download_dataset(self, dataframe):
        for index, row in dataframe.iterrows():
            split = row['split']
            image_url = row['dagshub_download_url']
            image_destination = os.path.join(self.yolo_dir, 'images', split, row['path'].split('/')[-1])
            self.download_file(image_url, image_destination)
            # Aquí deberías también generar el archivo .txt con la anotación YOLO.
            # Esto dependerá de cómo se almacenen tus anotaciones en DAGsHub.
            # Por ejemplo:
            label_destination = os.path.join(self.yolo_dir, 'labels', split, row['path'].split('/')[-1].replace('.jpg', '.txt'))
            self.download_file(image_url.replace('.jpg', '.txt'), label_destination)
            
    def create_yolo_v8_dataset_yaml(self, dataframe):
        with open(self.classes_file, 'r') as file:
            classes = [line.strip() for line in file.readlines()]
        
        yaml_content = {
            'train': os.path.join(self.yolo_dir, 'images', 'train'),
            'val': os.path.join(self.yolo_dir, 'images', 'valid'),
            'test': os.path.join(self.yolo_dir, 'images', 'test'),
            # Aquí necesitas agregar los nombres de las clases tal como se espera en el archivo YAML de YOLO
            'names': {i: name for i, name in enumerate(classes)}
        }
        # Deberías llenar 'names' con las clases de tu dataset
        with open(os.path.join(self.yolo_dir, 'dataset.yaml'), 'w') as yaml_file:
            yaml.dump(yaml_content, yaml_file)
    
    
    def remove_yolo_v8_labels(self):
        labels = os.path.join(self.yolo_conv.dataset_dir, 'labels')
        shutil.rmtree(labels, ignore_errors=True)
    
    def remove_yolo_v8_dataset(self):
        shutil.rmtree(self.yolo_conv.dataset_dir, ignore_errors=True)
        if os.path.exists('custom_yolo.yaml'):
            os.remove('custom_yolo.yaml')


    def create_categories_COCO(self, annotations):
        categories = set()
        json_annotation = json.loads(annotations.decode())
        if 'annotations' in json_annotation:
            for annotation in json_annotation["annotations"]:
                for result in annotation['result']:
                    categories.add(result['value'][result['type']][0])
        return ', '.join(str(item) for item in categories)

    def create_metadata(self, s):
        s["valid_datapoint"] = True
        s['year'] = 2017
        # Add annotations where relevant
        if not ('annotation' in s and s['annotation']):
            annotation = self.coco_conv.to_de(s)
            s['annotation'] = annotation
            if 'annotation' in s and s['annotation']:
                s['categories'] = self.create_categories_COCO(s["annotation"])

        return s
