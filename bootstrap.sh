#!/bin/bash

source ~/intel/openvino_2021/bin/setupvars.sh

VENV_PATH="$(pwd)/.venv"

python3 -m virtualenv ${VENV_PATH}

source ${VENV_PATH}/bin/activate

DEPLOYMENT_TOOLS_PATH=${INTEL_OPENVINO_DIR}/deployment_tools/
OPEN_MODEL_ZOO_PATH=${INTEL_OPENVINO_DIR}/deployment_tools/open_model_zoo

python -m pip install -r ${OPEN_MODEL_ZOO_PATH}/tools/downloader/requirements.in
python -m pip install -r ${OPEN_MODEL_ZOO_PATH}/tools/downloader/requirements-pytorch.in
python -m pip install -r ${OPEN_MODEL_ZOO_PATH}/tools/downloader/requirements-tensorflow.in
python -m pip install -r ${DEPLOYMENT_TOOLS_PATH}/model_optimizer/requirements_tf.txt
python -m pip install -r ${INTEL_OPENVINO_DIR}/python/requirements_tf.txt
python -m pip install -r ${DEPLOYMENT_TOOLS_PATH}/tools/benchmark_tool/requirements.txt

pushd ${OPEN_MODEL_ZOO_PATH}/tools/accuracy_checker
  python setup.py install
  python -m pip install pycocotools
popd