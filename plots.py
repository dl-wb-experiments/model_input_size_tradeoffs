from enum import Enum
from typing import Dict

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
    axes.set_ylabel('Throughput (FPS)')
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
                               s=80)
        shrinks.append(f'{percent}%')
        fps.append(fps_)
        accuracy.append(accuracy_)
        plots.append(scatter)

    line = axes.plot(accuracy, fps, line, color=line_color)[0]
    max_fps = max(fps)
    axes.set_ylim((0, max_fps + max_fps / 100 * 30))
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
    figure.savefig(f'{name}.eps', dpi=800)


def get_fractions_indexes(data_frame) -> Dict[int, Dict[str, int]]:
    previous_fraction = None
    fractions = {}
    for index, fraction in enumerate(data_frame):
        if 'Unnamed' in str(fraction):
            continue
        if fraction != previous_fraction:
            fractions[fraction] = index

    result_fractions = {}
    for fraction, start_index in fractions.items():
        percent = int(fraction * 100)
        result_fractions[percent] = {}
        headers = ('Image size', 'Accuracy', 'FPS', 'Latency', 'Batch', 'Stream')
        for index, head in enumerate(headers):
            result_fractions[percent][head] = index + start_index
    return result_fractions


def parse_data_from_sheet(data_frame) -> dict:
    fractions = get_fractions_indexes(data_frame)
    raws = data_frame.iterrows()
    next(raws)
    models = {}
    for _, data in raws:
        model_data = list(data)
        model_name = model_data[0]
        device = model_data[1]
        precision = model_data[2]
        for fraction, headers in fractions.items():
            if model_name not in models:
                models[model_name] = {}
            if device not in models[model_name]:
                models[model_name][device] = {}
            if precision not in models[model_name][device]:
                models[model_name][device][precision] = {}
            processed_model_data = models[model_name][device][precision]

            for header, index in headers.items():
                if not pd.notnull(model_data[index]):
                    continue
                if fraction not in processed_model_data:
                    processed_model_data[fraction] = {}
                processed_model_data[fraction][header] = model_data[index]
    return models


def read_data_from_excel(file_path: str) -> dict:
    data_frame_1 = pd.read_excel(file_path, sheet_name='1-1')
    data_1_1 = parse_data_from_sheet(data_frame_1)
    data_frame_auto = pd.read_excel(file_path, sheet_name='auto')
    data_auto = parse_data_from_sheet(data_frame_auto)
    return {
        '1_1': data_1_1,
        'auto': data_auto,
    }


class Devices(Enum):
    xeon = 'XEON CPU'
    tgl_multi = 'TGL MULTI'
    tgl_myriad = 'TGL MYRIAD'


class Models(Enum):
    ssd = 'ssd512'
    yolo_v2 = 'Yolo V2 TF'
    yolo_v3 = 'Yolo V3 TF'


class Precision(Enum):
    fp = 'FP32'
    int = 'INT8'


DEVICE_FOR_HEADER = {
    Devices.tgl_myriad: 'MYRIAD',
    Devices.tgl_multi: 'Multi',
    Devices.xeon: 'Xeon Gold CPU'
}

if __name__ == '__main__':
    models_data = read_data_from_excel('data/table.xlsx')

    for data in models_data.values():
        for model in Models:
            model_data = data[model.value]
            for device in Devices:
                model_device_data = model_data[device.value]
                fp_data = model_device_data[Precision.fp.value]
                int_data = None
                if device != Devices.tgl_myriad:
                    int_data = model_device_data[Precision.int.value]
                save_plot_combined(fp_data, int_data,
                                   f'{DEVICE_FOR_HEADER[device]}-device-execution {model.value} stream {fp_data[100]["Stream"]} batch {fp_data[100]["Batch"]} ')
