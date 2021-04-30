source ~/intel/openvino_2021/bin/setupvars.sh

VENV_PATH="$(pwd)/.venv"

python3 -m virtualenv ${VENV_PATH}

source ${VENV_PATH}/bin/activate

python3 -m pip install -r ${INTEL_OPENVINO_DIR}/deployment_tools/open_model_zoo/tools/downloader/requirements.in
python3 -m pip install -r ${INTEL_OPENVINO_DIR}/deployment_tools/open_model_zoo/tools/downloader/requirements-pytorch.in
python3 -m pip install -r ${INTEL_OPENVINO_DIR}/deployment_tools/open_model_zoo/tools/downloader/requirements-tensorflow.in
python3 -m pip install -r ${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer/requirements_tf.txt