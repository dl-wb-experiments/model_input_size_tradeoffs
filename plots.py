from typing import Dict

import matplotlib.pyplot as plt

data = {
    'ssd512': [
        {
            'precision': 'FP32',
            'image_size': 307,
            'shrink_percent': 60,
            'accuracy': 0.860219279,
            'fps': 31.05,
            'latency': 32.06
        },
        {
            'precision': 'FP32',
            'image_size': 358,
            'shrink_percent': 70,
            'accuracy': 0.8849146,
            'fps': 24,
            'latency': 41.42
        },
        {
            'precision': 'FP32',
            'image_size': 409,
            'shrink_percent': 80,
            'accuracy': 0.900116568,
            'fps': 18.99,
            'latency': 52.44
        },
        {
            'precision': 'FP32',
            'image_size': 460,
            'shrink_percent': 90,
            'accuracy': 0.925786965,
            'fps': 15.29,
            'latency': 65.12
        },
        {
            'precision': 'FP32',
            'image_size': 512,
            'shrink_percent': 100,
            'accuracy': 0.935635092,
            'fps': 12.82,
            'latency': 77.47
        },
        {
            'precision': 'INT8',
            'image_size': 307,
            'shrink_percent': 60,
            'accuracy': 0.860887752,
            'fps': 63.99,
            'latency': 14.64
        },
        {
            'precision': 'INT8',
            'image_size': 358,
            'shrink_percent': 70,
            'accuracy': 0.884815653,
            'fps': 61.29,
            'latency': 15.8
        },
        {
            'precision': 'INT8',
            'image_size': 409,
            'shrink_percent': 80,
            'accuracy': 0.900743537,
            'fps': 42.56,
            'latency': 22.61
        },
        {
            'precision': 'INT8',
            'image_size': 460,
            'shrink_percent': 90,
            'accuracy': 0.924497067,
            'fps': 37.71,
            'latency': 25.91
        },
        {
            'precision': 'INT8',
            'image_size': 512,
            'shrink_percent': 100,
            'accuracy': 0.936863839,
            'fps': 36.19,
            'latency': 26.05
        },
    ],
    'yolo_v3': [
        {
            'precision': 'FP32',
            'image_size': 130,
            'shrink_percent': 40,
            'accuracy': 0.321967471,
            'fps': 87.41,
            'latency': 11.33
        },
        {
            'precision': 'FP32',
            'image_size': 290,
            'shrink_percent': 70,
            'accuracy': 0.546742586,
            'fps': 38.66,
            'latency': 25.71
        },
        {
            'precision': 'FP32',
            'image_size': 416,
            'shrink_percent': 100,
            'accuracy': 0.622759,
            'fps': 24.55,
            'latency': 40.53
        },
        {
            'precision': 'INT8',
            'image_size': 130,
            'shrink_percent': 40,
            'accuracy': 0.31829058,
            'fps': 215.05,
            'latency': 4.57
        },
        {
            'precision': 'INT8',
            'image_size': 290,
            'shrink_percent': 70,
            'accuracy': 0.547299563,
            'fps': 94.94,
            'latency': 10.47
        },
        {
            'precision': 'INT8',
            'image_size': 416,
            'shrink_percent': 100,
            'accuracy': 0.622583988,
            'fps': 59.62,
            'latency': 16.75
        }
    ]
}
data_auto = {
    'ssd512': [
        {
            'precision': 'FP32',
            'image_size': 307,
            'shrink_percent': 60,
            'accuracy': 0.860219279,
            'fps': 41.49,
            'latency': 144.17
        },
        {
            'precision': 'FP32',
            'image_size': 358,
            'shrink_percent': 70,
            'accuracy': 0.8849146,
            'fps': 31.24,
            'latency': 190.95
        },
        {
            'precision': 'FP32',
            'image_size': 409,
            'shrink_percent': 80,
            'accuracy': 0.900116568,
            'fps': 24.54,
            'latency': 243.37
        },
        {
            'precision': 'FP32',
            'image_size': 460,
            'shrink_percent': 90,
            'accuracy': 0.925786965,
            'fps': 19.19,
            'latency': 309.49
        },
        {
            'precision': 'FP32',
            'image_size': 512,
            'shrink_percent': 100,
            'accuracy': 0.935635092,
            'fps': 15.65,
            'latency': 380.3
        },
        {
            'precision': 'INT8',
            'image_size': 307,
            'shrink_percent': 60,
            'accuracy': 0.860887752,
            'fps': 133.87,
            'latency': 44.06
        },
        {
            'precision': 'INT8',
            'image_size': 358,
            'shrink_percent': 70,
            'accuracy': 0.884815653,
            'fps': 97.67,
            'latency': 60.79
        },
        {
            'precision': 'INT8',
            'image_size': 409,
            'shrink_percent': 80,
            'accuracy': 0.900743537,
            'fps': 77.23,
            'latency': 76.78
        },
        {
            'precision': 'INT8',
            'image_size': 460,
            'shrink_percent': 90,
            'accuracy': 0.924497067,
            'fps': 60.08,
            'latency': 97.69
        },
        {
            'precision': 'INT8',
            'image_size': 512,
            'shrink_percent': 100,
            'accuracy': 0.936863839,
            'fps': 50.4,
            'latency': 117.61
        },
    ],
    'yolo_v3': [
        {
            'precision': 'FP32',
            'image_size': 130,
            'shrink_percent': 40,
            'accuracy': 0.321967471,
            'fps': 198.64,
            'latency': 29.99
        },
        {
            'precision': 'FP32',
            'image_size': 290,
            'shrink_percent': 70,
            'accuracy': 0.546742586,
            'fps': 68.51,
            'latency': 88.35
        },
        {
            'precision': 'FP32',
            'image_size': 416,
            'shrink_percent': 100,
            'accuracy': 0.622759,
            'fps': 37.69,
            'latency': 158.88
        },
        {
            'precision': 'INT8',
            'image_size': 130,
            'shrink_percent': 40,
            'accuracy': 0.31829058,
            'fps': 718.57,
            'latency': 8.27
        },
        {
            'precision': 'INT8',
            'image_size': 290,
            'shrink_percent': 70,
            'accuracy': 0.547299563,
            'fps': 226.38,
            'latency': 26.44
        },
        {
            'precision': 'INT8',
            'image_size': 416,
            'shrink_percent': 100,
            'accuracy': 0.622583988,
            'fps': 118.67,
            'latency': 50.38
        }
    ]
}

markers_map = {
    40: 's',
    60: 'o',
    70: '^',
    80: 'v',
    90: 'D',
    100: 'h',
}


def create_axes(name: str, x_lim=(80, 100), y_lim=(10, 40)):
    figure, axes = plt.subplots()
    figure.suptitle(name)
    axes.grid(True)
    axes.set_xlabel('Accuracy, %')
    axes.set_xlim(x_lim)
    axes.set_ylabel('Throughput (FPS)')
    axes.set_ylim(y_lim)
    return figure, axes


def add_plot_to_axes(axes, data: Dict, line_color: str, line: str = '-'):
    plots = []
    shrinks = set()
    accuracy = []
    fps = []

    for experiment in data:
        scatter = axes.scatter(experiment['accuracy'] * 100,
                               experiment['fps'],
                               c='#21befc',
                               marker=markers_map[experiment['shrink_percent']],
                               s=80)
        shrinks.add(f'{experiment["shrink_percent"]}%')
        fps.append(experiment['fps'])
        accuracy.append(experiment['accuracy'] * 100)
        plots.append(scatter)

    line = axes.plot(accuracy, fps, line, color=line_color)[0]

    return line, plots, shrinks


def save_plot_combined(data, int8_data, name, xlim=(80, 100), ylim=(10, 40)):

    figure, axes = create_axes(name)

    line, plots, shrinks = add_plot_to_axes(axes, data, line_color='#f4cc70')
    line_int8, plots_int8, shrinks_int8 = add_plot_to_axes(axes, int8_data, line='--', line_color='#fd8dbe')

    plots.extend(plots_int8)
    shrinks.update(shrinks_int8)

    legend = plt.legend(plots, shrinks)
    legend2 = plt.legend([line, line_int8], ['FP32', 'INT8'], loc='upper left')
    axes.add_artist(legend)
    axes.add_artist(legend2)

    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    figure.show()
    figure.savefig(f'{name}.eps', dpi =800)

if __name__ == '__main__':

    ssd_fp32_data = list(filter(lambda item: item['precision'] == 'FP32', data['ssd512']))
    ssd_int8_data = list(filter(lambda item: item['precision'] == 'INT8', data['ssd512']))
    yolo_fp32_data = list(filter(lambda item: item['precision'] == 'FP32', data['yolo_v3']))
    yolo_int8_data = list(filter(lambda item: item['precision'] == 'INT8', data['yolo_v3']))

    ssd_fp32_data_auto = list(filter(lambda item: item['precision'] == 'FP32', data_auto['ssd512']))
    ssd_int8_data_auto = list(filter(lambda item: item['precision'] == 'INT8', data_auto['ssd512']))
    yolo_fp32_data_auto = list(filter(lambda item: item['precision'] == 'FP32', data_auto['yolo_v3']))
    yolo_int8_data_auto = list(filter(lambda item: item['precision'] == 'INT8', data_auto['yolo_v3']))

    save_plot_combined(ssd_fp32_data, ssd_int8_data, 'SSD512 FP32 INT8', ylim=(0, 70))
    save_plot_combined(yolo_fp32_data, yolo_int8_data, 'YOLO v3 FP32 INT8', ylim=(0, 270), xlim=(30, 70))

    save_plot_combined(ssd_fp32_data_auto, ssd_int8_data_auto, 'SSD512 FP32 INT8 auto mode', ylim=(0, 140))
    save_plot_combined(yolo_fp32_data_auto, yolo_int8_data_auto, 'YOLO v3 FP32 INT8 auto mode', ylim=(0, 900),
                       xlim=(30, 70))
