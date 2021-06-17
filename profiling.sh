#!/bin/bash

source ~/intel/openvino_2021/bin/setupvars.sh
source .venv/bin/activate
benchmark=${INTEL_OPENVINO_DIR}/deployment_tools/tools/benchmark_tool/benchmark_app.py
root_models_path=models
models=$(ls models -I "public" -I "ssd512" -I "yolo-v2-tf" )

profile_model () {
  model_path=$1
  model_exists=`ls -1 ${model_path}/*.xml 2>/dev/null | wc -l`
  if [ ${model_exists} == 0 ];  then
    echo " Folder ${model_path} does not contain model. Skip"
    return
  fi
  model_file_name=$(find ${model_path} -name '*.xml')

  echo "${model_file_name} on MYRIAD"
  python ${benchmark} -d MYRIAD -m ${model_file_name} -report_type no_counters --report_folder ${model_path}
  mv ${model_path}/benchmark_report.csv ${model_path}/benchmark_report_MYRIAD_auto.csv
  
  python ${benchmark} -d MYRIAD -m ${model_file_name} -report_type no_counters --report_folder ${model_path} -nireq 1
  mv ${model_path}/benchmark_report.csv ${model_path}/benchmark_report_MYRIAD_1_1.csv

  echo "${model_file_name} on MULTY"
  python ${benchmark} -d MULTI:CPU,GPU -m ${model_file_name} -report_type no_counters --report_folder ${model_path}
  mv ${model_path}/benchmark_report.csv ${model_path}/benchmark_report_MULTI_auto.csv

  python ${benchmark} -d MULTI:CPU,GPU -m ${model_file_name} -report_type no_counters --report_folder ${model_path} -nstreams CPU:1,GPU:1 -b 1
  mv ${model_path}/benchmark_report.csv ${model_path}/benchmark_report_MULTI_2_2_4.csv 
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