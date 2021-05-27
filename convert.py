import shutil
import subprocess
import sys
from pathlib import Path

from constants import model_converter_script
from utils import move, load_model_info, arg_parser


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
    for index, input_shape in model_info['input_shapes'].items():
        result_path = Path('models1') / model_name / index

        convert_command = [sys.executable, model_converter_script,
                           '--name', model_info['name'],
                           '--download_dir', 'models',
                           '--precision', precision,
                           '--output_dir', str(result_path),
                           f'--add_mo_arg=--input_shape=[{input_shape}]']

        print(' '.join((str(i) for i in convert_command)))

        subprocess.run(convert_command)

        converted_model_path = result_path / 'public' / model_name / precision
        move(converted_model_path, result_path)
        shutil.rmtree(result_path / 'public')


def main(arguments):
    models_info = load_model_info(arguments.model)
    for model in models_info:
        convert_model(model)


if __name__ == '__main__':
    ARGUMENTS = arg_parser()
    main(ARGUMENTS)
