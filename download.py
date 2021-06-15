import subprocess
import sys

from utils import load_model_info, arg_parser
from constants import model_downloader_script


def download_model(model_name: str):
    subprocess.run([sys.executable, model_downloader_script, '--name', model_name, '--output_dir', 'models'])


def main(arguments):
    models_info = load_model_info(arguments.model)
    for model in models_info:
        download_model(model['name'])


if __name__ == '__main__':
    ARGUMENTS = arg_parser()
    main(ARGUMENTS)
