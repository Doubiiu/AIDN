#!/bin/sh
set -x
set -e

export OMP_NUM_THREADS=10
export KMP_INIT_AT_FORK=FALSE

PYTHON=python
dataset=DIV2K
TRAIN_CODE=train.py
TEST_CODE=test.py

exp_name=$1
config=$2

exp_dir=LOG/${dataset}/${exp_name}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result

now=$(date +"%Y%m%d_%H%M%S")

mkdir -p ${model_dir} ${result_dir}
mkdir -p ${exp_dir}/result/last
mkdir -p ${exp_dir}/result/best

export PYTHONPATH=./

## TRAIN
$PYTHON -u main/${TRAIN_CODE} \
  --config=${config} \
  save_path ${exp_dir} \
  2>&1 | tee -a ${exp_dir}/train-$now.log