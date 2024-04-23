import os
import json  # better to use "imports ujson as json" for the best performance

import uuid
import logging

from label_studio_converter.imports.label_config import generate_label_config

from .dags_converter import DagsConverter

logger = logging.getLogger('root')

class COCOAnnotationConverter():
    def __init__(self, annotation_file, to_name='image', from_name='label', label_type='bbox'):
        """Instantiate COCO Annotation Converter
        """
        self.ann_type = "COCO"
        self.annotation_file = annotation_file
        self.url_col = "dagshub_download_url"
        self.to_name = to_name
        self.from_name = from_name
        self.label_type = label_type
        
        # build categories=>labels dict
        if not self._update_data_from_file():
            raise ImportError(f'Unable to import annotations from {self.annotation_file}')
        
        categories = {i: line for i, line in enumerate(self.classes)}
        logger.info(f'Found {len(categories)} categories')

        if label_type == 'bbox':
            tags = {from_name: 'RectangleLabels'}
        elif label_type == 'segmentation':
            tags = {from_name: 'PolygonLabels'}
        else:
            raise NotImplementedError(f'Label type ({label_type}) has not been implemented.')

        # generate and save labeling config
        self.config = generate_label_config(
            categories,
            tags,
            to_name,
            from_name
        )
    
    def _update_data_from_file(self):
        if os.path.exists(self.annotation_file):
            with open(self.annotation_file) as f:
                data = json.load(f)
                self.class_map = {c['id']: c['name'] for c in data['categories']}
                self.classes = [c['name'] for c in data['categories']]

                self.images = {i['file_name']: {'height': i['height'], 'width': i['width'], 'id': i['id']} for i in data['images']}
                self.annotations = {}

                for ann in data['annotations']:
                    image_id = ann['image_id']
                    self.annotations[image_id] = self.annotations.get(image_id, []) + [ann]
            return True
        return False

    def _create_bbox(self, image_info, annotation_info):
        if image_info['id'] != annotation_info['image_id']:
            raise ValueError(f'Image ID ({image_info["id"]}) does not match Annotation Image ID ({annotation_info["image_id"]})')
    
        label_id = annotation_info['category_id']
        
        image_width = image_info['width']
        image_height = image_info['height']

        x, y, width, height = annotation_info['bbox']
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
                "x": (x / image_width) * 100,
                "y": (y / image_height) * 100,
                "width": (width / image_width) * 100,
                "height": (height / image_height) * 100,
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
    
    def _create_segmentation(self, image_info, annotation_info):
        if image_info['id'] != annotation_info['image_id']:
            raise ValueError(f'Image ID ({image_info["id"]}) does not match Annotation Image ID ({annotation_info["image_id"]})')
    
        label_id = annotation_info['category_id']
        
        image_width = image_info['width']
        image_height = image_info['height']

        segmentation = annotation_info['segmentation'][0]
        points = zip(segmentation[::2], segmentation[1::2])
        points = [[100.0 * float(x) / image_width, 100.0 * float(y) / image_height] for x, y in points]

        item = {
            "id": uuid.uuid4().hex[0:10],
            "type": "polygonlabels",
            "value": {
                "closed": True,
                "points": points,
                "polygonlabels": [self.class_map[label_id]],
            },
            "to_name": self.to_name,
            "from_name": self.from_name,
            "image_rotation": 0,
            "original_width": image_width,
            "original_height": image_height,
        }
        return item
                        
    def to_de(self, row, out_type="annotations"):
        """Convert COCO labeling to Label Studio JSON
        :param out_type: annotation type - "annotations" or "predictions"
        """
        # define coresponding label file and check existence
        image_path = row["path"]

        image_info = self.images.get(image_path, None) or self.images.get(os.path.split(image_path)[-1], None)

        task = None
        
        if image_info is not None:
            task = {
                "data": {
                    # eg. '../../foo+you.py' -> '../../foo%2Byou.py'
                    "image": row[self.url_col]
                }
            }

            image_width = image_info['width']
            image_height = image_info['height']
                                  
            task[out_type] = [
                {
                    "result": [],
                    "ground_truth": False,
                }
            ]

            # convert all bounding boxes to Label Studio Results
            for annotation in self.annotations.get(image_info['id'], []):
                if 'bbox' in self.label_type:
                    item = self._create_bbox(image_info, annotation)
                    task[out_type][0]['result'].append(item)
                if 'segmentation' in self.label_type:
                    item = self._create_segmentation(image_info, annotation)
                    task[out_type][0]['result'].append(item)
                    task['is_labeled'] = True

        if task:
            return json.dumps(task).encode()
    
    def from_de(self, row):
        annotation_data = row["annotation"]
        ls_converter = DagsConverter(self.config, self.dataset_dir, download_resources=False)
        output_dir = os.path.split(self.annotation_file)[0]
        ls_converter.convert_to_coco(input_data=annotation_data,
                                     output_dir=output_dir,
                                     output_image_dir=os.path.join(output_dir, 'data'),
                                     is_dir=False)
