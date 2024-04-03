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
        # Supongamos que cada fila del DataFrame ya tiene 'image_path', 'class_id' y 'bbox'.
        # Aquí puedes añadir cualquier lógica de enriquecimiento de metadatos que necesites.
    
        # Ejemplo: Añadir un metadato que siempre sea verdadero (similar a tu "valid_datapoint")
        #row["valid_datapoint"] = True
    
        # Ejemplo: Añadir un año fijo a todos los datos (como en tu ejemplo original)
        #row['year'] = 2017
    
        # Aquí puedes añadir más lógica para enriquecer tus datos.
        # Por ejemplo, calcular el área de la bounding box:
        bbox = row['bbox']  # Asumiendo que 'bbox' es una lista [x_center, y_center, width, height]
        area = bbox[2] * bbox[3]  # width * height
        row['bbox_area'] = area

        #Posición relativa de la Bounding Box: Podrías calcular la posición relativa del centro de la bounding box dentro de la imagen, 
        #lo que podría ser útil para identificar si los objetos tienden a aparecer en ciertas regiones de las imágenes.
        relative_x_center = bbox[0] / row['image_width']
        relative_y_center = bbox[1] / row['image_height']
        row['relative_x_center'] = relative_x_center
        row['relative_y_center'] = relative_y_center

        #Tamaño relativo de la Bounding Box: Calcular el tamaño de la bounding box como una fracción del tamaño total de la imagen puede 
        #ser útil para entender la escala de los objetos detectados.
        relative_area = area / (row['image_width'] * row['image_height'])
        row['relative_bbox_area'] = relative_area

        #Cercanía a los bordes de la imagen: Puede ser interesante saber si los objetos tienden a estar cerca de los bordes de las imágenes, lo que podría influir en la dificultad de detección.
        distance_to_edge_x = min(bbox[0], row['image_width'] - (bbox[0] + bbox[2]))
        distance_to_edge_y = min(bbox[1], row['image_height'] - (bbox[1] + bbox[3]))
        row['distance_to_edge_x'] = distance_to_edge_x
        row['distance_to_edge_y'] = distance_to_edge_y

    
        # Si necesitas categorías basadas en 'class_id', asegúrate de tener un mapeo de ID a categoría
        # Ejemplo: Si tienes un diccionario que mapea class_id a nombres de categoría
        #class_id_to_category = {0: 'Cat', 1: 'Dog'}  # Asume que tienes un mapeo como este
        #row['category'] = class_id_to_category.get(row['class_id'], 'Unknown')
    
        return row



# Uso de la clase DataFunctions
#df_func = DataFunctions()
#df_all = df_func.get_all_data()
#df_all = df_all.apply(df_func.create_metadata, axis=1)

#print(df_all.head())
