from pathlib import Path
import os

openvino_dir = Path(os.environ['INTEL_OPENVINO_DIR'])
model_downloader_script = openvino_dir / 'deployment_tools' / 'open_model_zoo' / 'tools' / 'downloader' / 'downloader.py'
model_converter_script = openvino_dir / 'deployment_tools' / 'open_model_zoo' / 'tools' / 'downloader' / 'converter.py'
