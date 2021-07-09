import copy
from pathlib import Path

import numpy as np
import yaml
from openvino.inference_engine import IECore

from constants import configs, data_path


def get_accuracy_config(model_name) -> dict:
    config_path = configs / f'{model_name}.yml'
    with config_path.open() as config_file:
        return yaml.safe_load(config_file)


def set_path_to_data(config: dict):
    new_config = copy.deepcopy(config)
    dataset = new_config['models'][0]['datasets'][0]
    dataset['data_source'] = str(data_path / dataset['data_source'])
    dataset['dataset_meta'] = str(data_path / dataset['dataset_meta'])
    dataset['annotation'] = str(data_path / dataset['annotation'])
    annotation_conversion = dataset['annotation_conversion']
    if 'annotation_file' in annotation_conversion:
        annotation_conversion['annotation_file'] = str(data_path / annotation_conversion['annotation_file'])
    if 'annotations_dir' in annotation_conversion:
        annotation_conversion['annotations_dir'] = str(data_path / annotation_conversion['annotations_dir'])
    if 'images_dir' in annotation_conversion:
        annotation_conversion['images_dir'] = str(data_path / annotation_conversion['images_dir'])
    if 'imageset_file' in annotation_conversion:
        annotation_conversion['imageset_file'] = str(data_path / annotation_conversion['imageset_file'])
    return new_config


def set_shape_to_config(config: dict, input_shape: list) -> dict:
    new_config = copy.deepcopy(config)
    preprocessings = new_config['models'][0]['datasets'][0]['preprocessing']
    for preprocessing in preprocessings:
        if preprocessing['type'] in ('resize', 'padding'):
            preprocessing['size'] = int(input_shape[2])
    postprocessings = new_config['models'][0]['datasets'][0]['postprocessing']
    for postprocessing in postprocessings:
        if postprocessing['type'] in ('faster_rcnn_postprocessing_resize'):
            postprocessing['size'] = int(input_shape[2])
    return new_config


def get_yolo_cells(xml_path: Path):
    core = IECore()
    network = core.read_network(xml_path)
    output = list(network.outputs.values())[0]
    return int(np.sqrt(output.shape[1] / 425))


def set_cells_to_config(config: dict, xml_path) -> dict:
    new_config = copy.deepcopy(config)
    adapter = new_config['models'][0]['launchers'][0]['adapter']
    if 'cells' in adapter:
        adapter['cells'] = get_yolo_cells(xml_path)
    return new_config


def create_config(model_name: str, input_shape: list, xml_path: Path) -> dict:
    source_config = get_accuracy_config(model_name)
    config = set_shape_to_config(source_config, input_shape)
    return set_cells_to_config(config, xml_path)
