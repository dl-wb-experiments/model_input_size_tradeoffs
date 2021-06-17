import copy
import shutil
import subprocess
from pathlib import Path

import numpy as np
from openvino.inference_engine import IECore

import yaml

from constants import configs, models_path, pot, data_path
from utils import load_model_info, arg_parser


def run_pot(model_path: Path):
    model_name = model_path.parent.stem
    dir_name = model_path.stem
    accuracy_command = [pot,
                        '--name', f'{model_name}',
                        '--ac-config', f'{model_path}/accuracy_checker_config.yml',
                        '-m', f'{model_path}/{model_name}.xml',
                        '-w', f'{model_path}/{model_name}.bin',
                        '--preset', 'performance',
                        '-q', 'default',
                        '--output-dir', f'{models_path}/{model_name}/quantized/{dir_name}',
                        '--direct-dump'
                        ]

    print(' '.join((str(i) for i in accuracy_command)))

    subprocess.run(accuracy_command)

    quantized_root_folder_path = models_path / model_name / 'quantized'
    quantized_folder_path = quantized_root_folder_path / dir_name
    optimized_folder_path = quantized_folder_path / 'optimized'
    move_quantized_model(quantized_folder_path, optimized_folder_path, model_name)


def move_quantized_model(quantized_folder_path, optimized_folder_path, model_name):
    quantized_xml = optimized_folder_path / f'{model_name}.xml'
    quantized_xml.rename(quantized_folder_path / f'{model_name}.xml')
    quantized_bin = optimized_folder_path / f'{model_name}.bin'
    quantized_bin.rename(quantized_folder_path / f'{model_name}.bin')
    shutil.rmtree(str(optimized_folder_path))


def get_config(model_name) -> dict:
    config_path = configs / f'{model_name}.yml'
    with config_path.open() as config_file:
        return yaml.safe_load(config_file)


def set_shape_to_config(config: dict, input_shape: list) -> dict:
    new_config = copy.deepcopy(config)
    preprocessings = new_config['models'][0]['datasets'][0]['preprocessing']
    for preprocessing in preprocessings:
        if preprocessing['type'] == 'resize':
            preprocessing['size'] = int(input_shape[2])
    return new_config


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


def get_cells(xml_path: Path):
    core = IECore()
    network = core.read_network(xml_path)
    output = list(network.outputs.values())[0]
    return int(np.sqrt(output.shape[1] / 425))


def set_cells_to_config(config: dict, xml_path) -> dict:
    new_config = copy.deepcopy(config)
    adapter = new_config['models'][0]['launchers'][0]['adapter']
    if 'cells' in adapter:
        adapter['cells'] = get_cells(xml_path)
    return new_config


def create_config(model_name: str, input_shape: list, xml_path) -> dict:
    source_config = get_config(model_name)
    new_config = set_shape_to_config(source_config, input_shape)
    new_config = set_cells_to_config(new_config, xml_path)
    return set_path_to_data(new_config)


def quantize_models(models_path: Path, model_info: dict):
    model_name = model_info['name']
    input_shapes = model_info['input_shapes']
    for percent, shape in input_shapes.items():

        model_path = models_path / percent
        xml_path = next(model_path.glob('*.xml'))
        if not xml_path or not list(model_path.glob('*.bin')):
            continue

        new_config = create_config(model_name, shape.split(','), xml_path)
        config_path = model_path / 'accuracy_checker_config.yml'
        with config_path.open('w') as config_file:
            yaml.dump(new_config, config_file)
        run_pot(model_path)


def quantize(model_info: dict):
    model_name = model_info['name']
    model_path = models_path / model_name
    quantize_models(model_path, model_info)


def main(arguments):
    models_info = load_model_info(arguments.model)
    for model_info in models_info:
        quantize(model_info)


if __name__ == '__main__':
    ARGUMENTS = arg_parser()
    main(ARGUMENTS)
