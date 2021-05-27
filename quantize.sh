#!/bin/bash


source ~/intel/openvino_2021/bin/setupvars.sh
source .venv/bin/activate

root_models_path=models
models=$(ls models -I "public")
configs_folder=configs

set -e

quantize   () {
  model_path=$1
  model_exists=`ls -1 ${model_path}/*.xml 2>/dev/null | wc -l`
  if [ ${model_exists} == 0 ];  then
    echo " Folder ${model_path} does not contain model. Skip"
    return
  fi
  model_name=$(basename -- $(dirname -- "$model_path"))
  dir_name=$(basename -- "$model_path")

  rm -rf *.pickle

  pot --ac-config ${configs_folder}/${model_name}/${dir_name}.yml \
                   --name ${model_name} \
                   -m ${model_path}/${model_name}.xml \
                   -w ${model_path}/${model_name}.bin \
                   --preset performance \
                   -q default \
                   --output-dir models/${model_name}/quantized/${dir_name} \
                   --direct-dump

  mv models/${model_name}/quantized/${dir_name}/optimized/* models/${model_name}/quantized/${dir_name}/
  rm -rf models/${model_name}/quantized/${dir_name}/optimized/
}

quantize_all_models(){
  root_model_path=$1
  sub_folders=$(ls ${root_model_path})

  for sub_folder in ${sub_folders}
  do
    quantize ${root_model_path}/${sub_folder}
  done

}

for model in ${models}
do
  quantize_all_models ${root_models_path}/${model}
done
