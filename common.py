import argparse
import json
import shutil
from pathlib import Path


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=False)
    return parser.parse_args()


def load_model_info(model: str) -> list:
    with open('models.json') as model_file:
        models_info = json.load(model_file)
    if not model:
        return models_info
    for model_info in models_info:
        if model_info['name'] == model:
            return [model_info, ]


def move(source_path: Path, destination_path: Path):
    for each_file in source_path.glob('*.*'):
        shutil.copy(each_file, destination_path)
