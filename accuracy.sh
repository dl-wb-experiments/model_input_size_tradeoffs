#!/bin/bash


source ~/intel/openvino_2021/bin/setupvars.sh
source .venv/bin/activate

root_models_path=models
models=$(ls models -I "public")
configs_folder=configs

set -e

measure_accuracy_for_model () {
  model_path=$1
  model_exists=`ls -1 ${model_path}/*.xml 2>/dev/null | wc -l`
  if [ ${model_exists} == 0 ];  then
    echo " Folder ${model_path} does not contain model. Skip"
    return
  fi
  model_name=$(basename -- $(dirname -- "$model_path"))
  dir_name=$(basename -- "$model_path")

  rm -rf *.pickle

  accuracy_check -c ${configs_folder}/${model_name}/${dir_name}.yml \
                   -m ${model_path} \
                   -s data/ \
                   --csv_result ${model_path}/accuracy_result.csv \
                   --async_mode 1 \
                   --profile_report_type json
}

measure_accuracy_for_all_models(){
  root_model_path=$1
  sub_folders=$(ls ${root_model_path})

  for sub_folder in ${sub_folders}
  do
    measure_accuracy_for_model ${root_model_path}/${sub_folder}
  done

}

for model in ${models}
do
  measure_accuracy_for_all_models ${root_models_path}/${model}
done
