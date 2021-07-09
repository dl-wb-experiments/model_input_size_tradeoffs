import subprocess
from pathlib import Path

import yaml

from common import get_accuracy_config, set_shape_to_config, set_cells_to_config
from constants import models_path, accuracy_checker, data_path
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


def create_config(model_name: str, input_shape: list, xml_path: Path) -> dict:
    source_config = get_accuracy_config(model_name)
    config = set_shape_to_config(source_config, input_shape)
    return set_cells_to_config(config, xml_path)


def measure_models_accuracy(models_path: Path, model_info: dict):
    model_name = model_info['name']
    input_shapes = model_info['input_shapes']
    for percent, shape in input_shapes.items():

        model_path = models_path / percent
        xml_path = next(model_path.glob('*.xml'))
        if not list(model_path.glob('*.xml')) or not list(model_path.glob('*.bin')):
            continue

        new_config = create_config(model_name, shape.split(','), xml_path)
        config_path = model_path / 'accuracy_checker_config.yml'
        with config_path.open('w') as config_file:
            yaml.dump(new_config, config_file)
        run_accuracy_check(model_path)


def accuracy(model_info: dict):
    model_name = model_info['name']
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
