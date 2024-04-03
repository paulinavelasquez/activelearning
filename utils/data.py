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
    def __init__(self, base_dir='data', to_name='image', from_name='label', label_type='bbox'):
        # Se asume que label_type es 'bbox' para detección de objetos (cuadros delimitadores)
        self.base_dir = base_dir
        self.to_name = to_name
        self.from_name = from_name
        self.label_type = label_type
    
    def load_data(self, split):
        images_dir = os.path.join(self.base_dir, split, 'images')
        labels_dir = os.path.join(self.base_dir, split, 'labels')

        # Carga las rutas de las imágenes y las etiquetas
        data = []
        for img_filename in os.listdir(images_dir):
            if img_filename.endswith('.jpg'):  # Asegúrate de procesar solo imágenes .jpg
                img_path = os.path.join(images_dir, img_filename)
                label_filename = img_filename.replace('.jpg', '.txt')
                label_path = os.path.join(labels_dir, label_filename)

                if os.path.exists(label_path):  # Solo agrega datos si existe la etiqueta correspondiente
                    data.append({
                        'image_path': img_path,
                        'label_path': label_path,
                        'split': split
                    })
        
        return pd.DataFrame(data)

    # Función para generar el DataFrame completo con todos los splits
    def get_all_data(self):
        df_train = self.load_data('train')
        df_valid = self.load_data('valid')
        df_test = self.load_data('test')
        return pd.concat([df_train, df_valid, df_test], ignore_index=True)
    
    def create_metadata(self, row):
        with open(row['label_path'], 'r') as file:
            lines = file.readlines()
    
        objects = []
        class_counts = Counter()
        total_bbox_area = 0  # Para calcular la complejidad de la imagen
    
        for line in lines:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            bbox_area = width * height  # Área del bounding box
            aspect_ratio = width / height if height > 0 else 0  # Proporción de aspecto
            total_bbox_area += bbox_area  # Acumular para la complejidad de la imagen
    
            objects.append({
                'class_id': int(class_id),
                'bbox': [x_center, y_center, width, height],
                'bbox_area': bbox_area,
                'aspect_ratio': aspect_ratio,
            })
    
            class_counts[int(class_id)] += 1
    
        with Image.open(row['image_path']) as img:
            image_width, image_height = img.size
    
        image_complexity = total_bbox_area / (image_width * image_height) if image_width * image_height > 0 else 0  # Ejemplo simple de complejidad
        
        # Agrega metadatos adicionales
        row.update({
            'num_objects': len(objects), #representa el número total de objetos detectados en una imagen.
            'objects': objects, #lista de diccionarios contiene información detallada sobre cada objeto detectado, incluidas las coordenadas de sus bounding boxes y sus clases. 
            'image_width': image_width,
            'image_height': image_height,
            'class_counts': dict(class_counts),
            'image_complexity': image_complexity,
        })
    
        return row

# Uso de la clase DataFunctions
#df_func = DataFunctions()
#df_all = df_func.get_all_data()
#df_all = df_all.apply(df_func.create_metadata, axis=1)

#print(df_all.head())