import hashlib
import json
import os
import pandas as pd
import shutil
import yaml

from sklearn.model_selection import train_test_split

from .coco_converter import COCOAnnotationConverter
from .yolo_converter import YOLOAnnotationConverter

from PIL import Image
from collections import Counter

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
    def __init__(self, base_dir='data', to_name='image', from_name='label', label_type='bbox', annotation_file=None):
        self.base_dir = base_dir
        self.to_name = to_name
        self.from_name = from_name
        self.label_type = label_type
        self.annotation_file = annotation_file  # Ruta al archivo JSON con las anotaciones

    def load_data_from_json(self):
        # Carga las anotaciones desde el archivo JSON
        with open(self.annotation_file, 'r') as file:
            annotations = json.load(file)

        # Convertir las anotaciones a DataFrame
        data = []
        for entry in annotations:
            image_path = entry["image_path"]
            for annotation in entry["annotations"]:
                data.append({
                    'image_path': os.path.join(self.base_dir, image_path),
                    'class_id': annotation['class_id'],
                    'bbox': annotation['bbox'],
                    'subset': entry.get('subset', 'train')  # Asume 'train' como predeterminado si 'subset' no está presente
                })

        return pd.DataFrame(data)
    
    def create_metadata(self, row):
        # Calcula el área de cada bounding box en las anotaciones
        for annotation in row['annotations']:
            bbox = annotation['bbox']
            # Asume que las coordenadas de bbox están en formato [x_center, y_center, width, height]
            area = bbox[2] * bbox[3]  # width * height
            annotation['area'] = area
    
        # Cuenta el número total de objetos en la imagen
        row['num_objects'] = len(row['annotations'])
    
        return row




# Uso de la clase DataFunctions
#df_func = DataFunctions()
#df_all = df_func.get_all_data()
#df_all = df_all.apply(df_func.create_metadata, axis=1)

#print(df_all.head())
