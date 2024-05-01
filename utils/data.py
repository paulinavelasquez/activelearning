import hashlib
import json
import os
import pandas as pd
import shutil
import yaml

from sklearn.model_selection import train_test_split
from .yolo_converter import YOLOAnnotationConverter

class DataFunctions():
    def __init__(self, yolo_dir, classes_file, to_name='image', from_name='label', label_type='bbox'):
        self.classes_file = classes_file
        
        #abrir el archivo txt con las clases
        with open(self.classes_file, 'r') as file:
            self.classes = [line.strip() for line in file.readlines()]
            
        self.yolo_conv = YOLOAnnotationConverter(
            dataset_dir=yolo_dir, 
            classes=self.classes,
            to_name=to_name, 
            from_name=from_name,
            label_type=label_type)

    def remove_yolo_v8_labels(self):
        labels = os.path.join(self.yolo_conv.dataset_dir, 'labels')
        shutil.rmtree(labels, ignore_errors=True)
    
    def remove_yolo_v8_dataset(self):
        shutil.rmtree(self.yolo_conv.dataset_dir, ignore_errors=True)
        if os.path.exists('custom_yolo.yaml'):
            os.remove('custom_yolo.yaml')

    def create_yolo_v8_dataset_yaml(self, dataset, download=True):
        path = os.path.abspath(self.yolo_conv.dataset_dir)

        if download:
            self.remove_yolo_v8_dataset()
            for split in ('train', 'valid', 'test'):
                split_ds = dataset[dataset['split'] == split]
                target_dir = os.path.join(path, f'images/{split}')
                _ = split_ds.all().download_files(target_dir=target_dir, keep_source_prefix=False)
        else:
            self.remove_yolo_v8_labels()

        for dp in dataset.all().get_blob_fields("annotation"):
            if 'split' in dp:
                self.yolo_conv.from_de(dp)
            else:
                print("Warning: No 'split' found for datapoint", dp)


        train = 'images/train'
        val = 'images/valid'
        test = 'images/test'

        yaml_dict = {
            'path': path, 
            'train': train, 
            'val': val,
            'test': test,
            'names': {i: name for i, name in enumerate(self.yolo_conv.classes)}
        }
        with open("custom_yolo.yaml", "w") as file:
            file.write(yaml.dump(yaml_dict))
    
    def create_categories_COCO(self, annotations):
        categories = set()
        json_annotation = json.loads(annotations.decode())
        if 'annotations' in json_annotation:
            for annotation in json_annotation["annotations"]:
                for result in annotation['result']:
                    categories.add(result['value'][result['type']][0])
        return ', '.join(str(item) for item in categories)

    def extract_categories(self, annotations):
        categories = set()
        # Ejemplo: supongamos que las anotaciones incluyen categorías directamente
        for annotation in annotations:
            categories.add(annotation['annotation_classes'])  # Ajusta el acceso según cómo estén estructuradas tus anotaciones.
        return ', '.join(str(item) for item in categories)


    def create_metadata(self, s):
        s["valid_datapoint"] = True
        s['year'] = 2024
        # Add annotations where relevant
        if not ('annotation' in s and s['annotation']):
            # Supongamos que 'annotation' ya está en el formato necesario o es fácilmente convertible
            s['annotation'] = self.yolo_conv.to_de(s)
            if 'annotation' in s and s['annotation']:
                s['annotation_classes'] = self.extract_categories(s["annotation"])
        return s

#############
