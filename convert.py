import shutil
import subprocess
import sys
from pathlib import Path

from utils import move, load_model_info, arg_parser
from constants import model_converter_script


def calculate_shape(source_shape: list, percent: int, input_layout: str) -> tuple:
    h_index = input_layout.index('H')
    w_index = input_layout.index('W')
    c_index = input_layout.index('C')

    h, w = source_shape[h_index], source_shape[w_index]

    if c_index == 1:
        return source_shape[0], source_shape[c_index], int(h / 100 * percent), int(w / 100 * percent)
    if c_index == 3:
        return source_shape[0], int(h / 100 * percent), int(w / 100 * percent), source_shape[c_index]


def convert_model(model_info: dict, precision: str = 'FP16'):
    model_name = model_info['name']
    for percent in range(3, 11):
        percent = percent * 10
        input_layout = model_info.get('input_layout', 'NCHW')
        input_resolution = calculate_shape(model_info['input_shape'], percent, input_layout)
        resolution_sub_folder = '_'.join((str(i) for i in input_resolution))

        result_path = Path('models') / model_name / resolution_sub_folder

        input_shape_argument = f'[{",".join((str(i) for i in input_resolution))}]'

        convert_command = [sys.executable, model_converter_script,
                        '--name', model_info['name'],
                        '--download_dir', 'models',
                        '--precision', precision,
                        '--output_dir', str(result_path),
                        f'--add_mo_arg=--input_shape={input_shape_argument}']

        print(' '.join((str(i) for i in convert_command)))

        subprocess.run(convert_command)

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
