import argparse
from enum import Enum
from pathlib import Path
from typing import Dict
from pandas.core.frame import DataFrame
import matplotlib.pyplot as plt
import pandas as pd

markers_map = {
    40: 's',
    60: 'o',
    70: '^',
    80: 'v',
    90: 'D',
    100: 'h',
}


def create_axes(name: str):
    figure, axes = plt.subplots()
    figure.suptitle(name)
    axes.grid(True)
    axes.set_xlabel('Accuracy, %')
    axes.set_ylabel('Throughput, fps')
    figure.set_size_inches(8, 6)
    return figure, axes


def add_plot_to_axes(axes, data: Dict, line_color: str, line: str = '-'):
    plots = []
    shrinks = []
    accuracy = []
    fps = []

    for percent, experiment in data.items():
        accuracy_ = experiment['Accuracy'] * 100
        fps_ = experiment['FPS']
        scatter = axes.scatter(accuracy_,
                               fps_,
                               c='#21befc',
                               marker=markers_map[percent],
                               s=60)
        shrinks_name = f'{100 - percent}%'
        if 100 == percent:
            shrinks_name += ' (original shape)'
        shrinks.append(shrinks_name)
        fps.append(fps_)
        accuracy.append(accuracy_)
        plots.append(scatter)

    line = axes.plot(accuracy, fps, line, color=line_color)[0]
    max_fps = max(fps)
    axes.set_ylim((0, max_fps + max_fps / 100 * 52))
    return line, plots, shrinks


def save_plot_combined(data, int8_data, name):
    figure, axes = create_axes(name)

    line, plots, shrinks = add_plot_to_axes(axes, data, line_color='#f4cc70')
    if int8_data:
        line_int8, plots_int8, shrinks_int8 = add_plot_to_axes(axes, int8_data, line='--', line_color='#fd8dbe')

        plots.extend(plots_int8)
        legend2 = plt.legend([line, line_int8], ['FP32', 'INT8'], loc='upper left')

    else:
        legend2 = plt.legend([line], ['FP32'], loc='upper left')

    legend = plt.legend(plots, shrinks)

    axes.add_artist(legend)
    axes.add_artist(legend2)

    figure.show()
    figure.savefig(f'plots/{name}.eps', dpi=800)


colors = {
    'basic inference configuration': {
        'FP32': '#20aedf',
        'INT8': '#c1b94d'
    },
    'optimal inference configuration': {
        'FP32': '#ed7fb0',
        'INT8': '#76d2b1'
    }
}

line_types = {
    'FP32': '-',
    'INT8': '--'
}


def save_plot_combined_model(model_name, data):
    if model_name not in ('SSD512', 'YOLOv2', 'Faster R-CNN'):
        return
    figure, axes = create_axes(f'Applying optimization pipeline to {model_name} model on Xeon platform')

    all_plots = []
    lines = {}

    for label, data_per_configuration in data.items():
        for device, data_per_device in data_per_configuration.items():
            if 'XEON' not in device:
                continue
            for precision, d in data_per_device.items():
                color = colors[label][precision]
                line, plots, shrinks = add_plot_to_axes(axes, d, line_color=color, line=line_types[precision])
                fps = d[60]['FPS']
                lines[fps] = (f'{precision} model, {label}', line)
                all_plots.extend(plots)

    l = [lines[line][0] for line in sorted(lines, reverse=True)]
    l2 = [lines[line][1] for line in sorted(lines, reverse=True)]
    legend2 = plt.legend(l2, l, loc='upper left', title='Inference experiments')

    legend = plt.legend(all_plots, shrinks, loc='upper right', title='Shape reduction')

    axes.add_artist(legend)
    axes.add_artist(legend2)

    figure.show()
    figure.savefig(f'plots/{model_name}.eps', dpi=800)


def get_reductions_indexes(data_frame: DataFrame) -> Dict[int, Dict[str, int]]:
    previous_reduction = None
    fractions = {}

    for index, fraction in enumerate(data_frame):
        if 'Unnamed' in str(fraction):
            continue
        if fraction != previous_reduction:
            fractions[fraction] = index

    result_fractions = {}

    for fraction, start_index in fractions.items():
        percent = int(fraction * 100)
        result_fractions[percent] = {}
        headers = ('Image size', 'Accuracy', 'FPS', 'Latency', 'Batch', 'Stream')
        for index, head in enumerate(headers):
            result_fractions[percent][head] = index + start_index

    return result_fractions


def parse_data_from_sheet(data_frame: DataFrame) -> dict:
    reductions = get_reductions_indexes(data_frame)
    raws = data_frame.iterrows()
    next(raws)
    models = {}
    for _, raw_data in raws:

        model_raw_data = list(raw_data)
        model_name = model_raw_data[0]
        device_name = model_raw_data[1]
        precision = model_raw_data[2]
        if pd.isnull(model_name):
            continue
        for reduction, headers in reductions.items():

            if model_name not in models:
                models[model_name] = {}
            if device_name not in models[model_name]:
                models[model_name][device_name] = {}
            if precision not in models[model_name][device_name]:
                models[model_name][device_name][precision] = {}

            processed_model_data = models[model_name][device_name][precision]

            for header, index in headers.items():
                if pd.isnull(model_raw_data[index]):
                    continue
                if reduction not in processed_model_data:
                    processed_model_data[reduction] = {}

                processed_model_data[reduction][header] = model_raw_data[index]

    return models


def read_data_from_excel(file_path: str) -> dict:
    data_frame_1 = pd.read_excel(file_path,
                                 sheet_name='1-1',
                                 engine='openpyxl')
    data_1_1 = parse_data_from_sheet(data_frame_1)

    data_frame_auto = pd.read_excel(file_path,
                                    sheet_name='auto',
                                    engine='openpyxl', )
    data_auto = parse_data_from_sheet(data_frame_auto)

    models_data = {}
    for model in data_auto:
        print(model)
        if model not in models_data:
            models_data[model] = {}
        models_data[model]['basic inference configuration'] = data_1_1[model]
        models_data[model]['optimal inference configuration'] = data_auto[model]

    return models_data


class Devices(Enum):
    xeon = 'XEON CPU'
    # tgl_multi = 'TGL MULTI'
    # tgl_myriad = 'TGL MYRIAD'


class Models(Enum):
    ssd = 'SSD512'
    yolo_v2 = 'YOLOv2'
    faster_rcnn = 'Faster R-CNN'
    # yolo_v3 = 'Yolo V3 TF'


class Precision(Enum):
    fp = 'FP32'
    int = 'INT8'


DEVICE_FOR_HEADER = {
    # Devices.tgl_myriad: 'MYRIAD',
    # Devices.tgl_multi: 'Multi',
    Devices.xeon: 'Xeon Gold CPU'
}


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-data', required=True, type=Path)
    parser.add_argument('--output-path', required=False, default=Path.cwd() / 'plots')
    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parser()
    models_data = read_data_from_excel(args.input_data)
    data_per_models = {}

    # for configuration, data in models_data.items():
    #     for model in Models:
    #         if model not in data_per_models:
    #             data_per_models[model] = {}
    #
    #         data_per_models[model][configuration] = {}
    #
    #         model_data = data[model.value]
    #
    #         for device in Devices:
    #             model_device_data = model_data[device.value]
    #
    #             data_per_models[model][configuration] = {
    #                 'FP32': model_device_data[Precision.fp.value],
    #                 'INT8': model_device_data[Precision.int.value]
    #             }

    for model, data in models_data.items():
        print(model)
        save_plot_combined_model(model, data)
