#!/bin/sh
set -x
set -e

export OMP_NUM_THREADS=10
export KMP_INIT_AT_FORK=FALSE

PYTHON=python
dataset=DIV2K
TEST_CODE=benchmark.py

config=$1

exp_dir=LOG/${dataset}/pre-train
model_dir=${exp_dir}
result_dir=${exp_dir}/result

mkdir -p ${result_dir}

now=$(date +"%Y%m%d_%H%M%S")

export PYTHONPATH=./

$PYTHON -u main/${TEST_CODE} \
--config=${config} \
save_folder ${exp_dir}/result \
model_path ${model_dir}/AIDN.pth.tar