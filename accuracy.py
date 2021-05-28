import copy
import subprocess
from pathlib import Path

import yaml

from constants import configs, models_path, accuracy_checker, data_path
from utils import load_model_info, arg_parser


def run_accuracy_check(model_path: str):
    accuracy_command = [accuracy_checker,
                        '-c', f'{model_path}/accuracy_checker_config.yml',
                        '-m', f'{model_path}',
                        '-s', f'{data_path}',
                        '--csv_result', f'{model_path}/accuracy_result.csv',
                        '--async_mode', '1']

    print(' '.join((str(i) for i in accuracy_command)))

    subprocess.run(accuracy_command)


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


def create_config(model_name: str, input_shape: list) -> dict:
    source_config = get_config(model_name)
    return set_shape_to_config(source_config, input_shape)


def measure_models_accuracy(models_path: Path, model_info: dict):
    model_name = model_info['name']
    input_shapes = model_info['input_shapes']
    for percent, shape in input_shapes.items():

        model_path = models_path / percent
        if not list(model_path.glob('*.xml')) or not list(model_path.glob('*.bin')):
            continue

        new_config = create_config(model_name, shape.split(','))
        config_path = model_path / 'accuracy_checker_config.yml'
        with config_path.open('w') as config_file:
            yaml.dump(new_config, config_file)
        run_accuracy_check(model_path)


def accuracy(model_info: dict):
    model_name = model_info['name']
    input_shapes = model_info['input_shapes']
    model_path = models_path / model_name
    measure_models_accuracy(model_path, model_info)
    model_path = models_path / model_name / 'quantized'
    measure_models_accuracy(model_path, model_info)


def main(arguments):
    models_info = load_model_info(arguments.model)
    for model_info in models_info:
        accuracy(model_info)


if __name__ == '__main__':
    ARGUMENTS = arg_parser()
    main(ARGUMENTS)
