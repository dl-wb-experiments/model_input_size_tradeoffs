import shutil
import subprocess
from pathlib import Path

import yaml

from common import get_accuracy_config, set_shape_to_config, set_cells_to_config, set_path_to_data
from constants import models_path, pot
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
                        '--direct-dump',
                        '--evaluate',
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


def create_config(model_name: str, input_shape: list, xml_path) -> dict:
    source_config = get_accuracy_config(model_name)
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
