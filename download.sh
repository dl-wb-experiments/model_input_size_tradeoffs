#!/bin/bash

set -e

source ~/intel/openvino_2021/bin/setupvars.sh
source .venv/bin/activate

models="models.txt"

while read model; do
  
  echo "Download model ${model}"
  python ${INTEL_OPENVINO_DIR}/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name ${model}
  
  echo "Convert model ${model}"
  python ${INTEL_OPENVINO_DIR}/deployment_tools/open_model_zoo/tools/downloader/converter.py --name ${model} --precisions FP16

done < ${models}