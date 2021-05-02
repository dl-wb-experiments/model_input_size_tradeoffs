import shutil
import subprocess
import sys
from pathlib import Path

from common import move, load_model_info, arg_parser
from constants import model_converter_script


def calculate_shape(source_shape: list, percent: int) -> tuple:
    h, w = source_shape[-2:]
    return source_shape[0], source_shape[1], int(h / 100 * percent), int(w / 100 * percent)


def convert_model(model_info: dict, precision: str = 'FP16'):
    model_name = model_info['name']
    for percent in range(3, 10):
        percent = percent * 10
        input_resolution = calculate_shape(model_info['input_shape'], percent)
        resolution_sub_folder = '_'.join((str(i) for i in input_resolution))

        result_path = Path('models') / model_name / resolution_sub_folder

        input_shape_argument = f'[{",".join((str(i) for i in input_resolution))}]'

        subprocess.run([sys.executable, model_converter_script,
                        '--name', model_info['name'],
                        '--download_dir', 'models',
                        '--precision', precision,
                        '--output_dir', str(result_path),
                        f'--add_mo_arg=--input_shape={input_shape_argument}'])

        converted_model_path = result_path / 'public' / model_name / precision
        move(converted_model_path, result_path)
        shutil.rmtree(result_path / 'public')


def main(arguments):
    models_info = load_model_info(arguments.model)
    for model in models_info:
        # download_model(model['name'])
        convert_model(model)


if __name__ == '__main__':
    ARGUMENTS = arg_parser()
    main(ARGUMENTS)
