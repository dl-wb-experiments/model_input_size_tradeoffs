#!/bin/bash

source ~/intel/openvino_2021/bin/setupvars.sh
source .venv/bin/activate
benchmark=${INTEL_OPENVINO_DIR}/deployment_tools/tools/benchmark_tool/benchmark_app.py
root_models_path=models
models=$(ls models -I "public")

profile_model () {
  model_path=$1
  model_exists=`ls -1 ${model_path}/*.xml 2>/dev/null | wc -l`
  if [ ${model_exists} == 0 ];  then
    echo " Folder ${model_path} does not contain model. Skip"
    return
  fi
  model_file_name=$(find ${model_path} -name '*.xml')

  echo ${model_file_name}
  python ${benchmark} -m ${model_file_name} -report_type no_counters --report_folder ${model_path}
}

profile_all_models(){
  root_model_path=$1
  sub_folders=$(ls ${root_model_path})

  for sub_folder in ${sub_folders}
  do
    if [ "${sub_folder}" == "quantized" ]; then
        profile_all_models ${root_model_path}/${sub_folder}
        continue
    fi
    profile_model ${root_model_path}/${sub_folder}
  done

}

for model in ${models}
do
  profile_all_models ${root_models_path}/${model}
done