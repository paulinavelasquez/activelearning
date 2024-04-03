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
        notes_file = os.path.join(self.dataset_dir, 'classes.txt')
        if os.path.exists(notes_file):
            with open(notes_file) as f:
                self.classes = [line.strip() for line in f.readlines()]
            return True
        return False


    def _create_bbox(self, line, image_width, image_height):
        label_id, x, y, width, height = line.split()
        x, y, width, height = (
            float(x),
            float(y),
            float(width),
            float(height),
        )
        item = {
            "id": uuid.uuid4().hex[0:10],
            "type": "rectanglelabels",
            "value": {
                "x": (x - width / 2) * 100,
                "y": (y - height / 2) * 100,
                "width": width * 100,
                "height": height * 100,
                "rotation": 0,
                "rectanglelabels": [self.classes[int(label_id)]],
            },
            "to_name": self.to_name,
            "from_name": self.from_name,
            "image_rotation": 0,
            "original_width": image_width,
            "original_height": image_height,
        }
        return item
    
    def _create_segmentation(self, line, image_width, image_height):
        label_id = line.split()[0]
        points = [[float(x[0]), float(x[1])] for x in zip(*[iter(line.split()[1:])]*2)]

        for i in range(len(points)):
            points[i][0] = points[i][0] * 100.0
            points[i][1] = points[i][1] * 100.0

        item = {
            "id": uuid.uuid4().hex[0:10],
            "type": "polygonlabels",
            "value": {
                "closed": True,
                "points": points,
                "polygonlabels": [self.classes[int(label_id)]],
            },
            "to_name": self.to_name,
            "from_name": self.from_name,
            "image_rotation": 0,
            "original_width": image_width,
            "original_height": image_height,
        }
        return item


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
            """Convert YOLO labeling to Label Studio JSON
            :param out_type: annotation type - "annotations" or "predictions"
            """
            # define coresponding label file and check existence
            image_path = row["path"]
            if not "images/" in image_path:
                image_path = os.path.join("images", image_path)
            label_path = image_path.replace(image_path.split(".")[-1], "txt")
            if "/images/" in label_path or label_path.startswith("images/"):
                label_path = label_path.replace("images/","labels/")
            else:
                label_path = os.path.join("labels", label_path)
        
            label_file = os.path.join(self.dataset_dir, label_path)
            image_file = os.path.join(self.dataset_dir, image_path)
            image_width = 0
            image_height = 0
        
            task = None
                                    
            if os.path.exists(label_file):
                task = {
                    "data": {
                        # eg. '../../foo+you.py' -> '../../foo%2Byou.py'
                        "image": row[self.url_col]
                    }
                }
                                    
                task[out_type] = [
                    {
                        "result": [],
                        "ground_truth": False,
                    }
                ]

                # read image sizes
                if not (image_width and image_height):
                    # default to opening file if we aren't given image dims. slow!
                    with Image.open(os.path.join(image_file)) as im:
                        image_width, image_height = im.size

                with open(label_file) as file:
                    # convert all bounding boxes to Label Studio Results
                    lines = file.readlines()
                    for line in lines:
                        if 'bbox' in self.label_type:
                            item = self._create_bbox(line, image_width, image_height)
                            task[out_type][0]['result'].append(item)
                        if 'segmentation' in self.label_type:
                            item = self._create_segmentation(line, image_width, image_height)
                            task[out_type][0]['result'].append(item)
                            task['is_labeled'] = True

            if task:
                return json.dumps(task).encode()
# Esta sería la manera de usar la clase para convertir anotaciones
#yolo_conv = YOLOAnnotationConverter(dataset_dir='tu_ruta_dataset')
# yolo_conv.from_de(tu_fila_de_datos)  # Aquí deberías pasar una fila de tu DataFrame
