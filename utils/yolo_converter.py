import os
import json  # better to use "imports ujson as json" for the best performance

import uuid
import logging

from PIL import Image

from label_studio_converter.imports.label_config import generate_label_config

from .dags_converter import DagsConverter

logger = logging.getLogger('root')

class YOLOAnnotationConverter:
    def __init__(self, dataset_dir, classes=[], to_name='image', from_name='label', label_type='bbox'):
        self.ann_type = "YOLO"
        self.dataset_dir = dataset_dir
        self.classes = classes
        self.to_name = to_name
        self.from_name = from_name
        self.label_type = label_type
        
        if len(self.classes) == 0:
            self._update_classes_from_file()
        
        # Si es necesario generar un archivo de configuración para las etiquetas
        self.config = generate_label_config(
            {i: name for i, name in enumerate(self.classes)},
            {from_name: 'RectangleLabels'},
            to_name,
            from_name
        )
    
    def _update_classes_from_file(self):
        classes_file = os.path.join(self.dataset_dir, 'classes.txt')
        if os.path.exists(classes_file):
            with open(classes_file) as f:
                self.classes = [line.strip() for line in f.readlines()]
    
    def from_de(self, row):
        # Suponiendo que el archivo 'annotation' es una representación de la anotación en un formato intermedio,
        # como puede ser un JSON que viene de Label Studio
        annotation_data = row["annotation"]
        ls_converter = DagsConverter(self.config, self.dataset_dir, download_resources=False)
        ls_converter.convert_to_yolo(
            input_data=annotation_data,
            output_dir=os.path.join(self.dataset_dir, row['split']),
            output_label_dir=os.path.join(self.dataset_dir, row['split'], 'labels', *os.path.split(row['path'][:-1])),
            is_dir=False
        )
        
        # Aquí se actualizaría el archivo de clases si hay nuevas clases en las anotaciones
        ls_converter._get_labels()
        self._update_classes_from_file()

    def to_de(self, row, out_type="annotations"):
        # Esta función se supone que convierte las anotaciones YOLO en formato Label Studio JSON
        # ... Este código deberá ser actualizado según tus necesidades y formato de salida esperado

# Esta sería la manera de usar la clase para convertir anotaciones
yolo_conv = YOLOAnnotationConverter(dataset_dir='tu_ruta_dataset')
# yolo_conv.from_de(tu_fila_de_datos)  # Aquí deberías pasar una fila de tu DataFrame
