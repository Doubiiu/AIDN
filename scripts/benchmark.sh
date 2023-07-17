#!/bin/sh
set -x
set -e

export OMP_NUM_THREADS=10
export KMP_INIT_AT_FORK=FALSE

PYTHON=python
dataset=DIV2K
TEST_CODE=benchmark.py

exp_name=$1
config=$2

exp_dir=LOG/${dataset}/${exp_name}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result

mkdir -p ${model_dir} ${result_dir}
mkdir -p ${exp_dir}/result/best/${dataset}

now=$(date +"%Y%m%d_%H%M%S")

export PYTHONPATH=./

$PYTHON -u main/${TEST_CODE} \
--config=${config} \
save_folder ${exp_dir}/result \
model_path ${model_dir}/AIDN.pth.tar