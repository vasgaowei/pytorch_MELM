#!/usr/bin/env bash
export CUDA_PATH=/usr/local/cuda/
#You may also want to ad the following
export C_INCLUDE_PATH=/usr/local/cuda/include
python setup_cuda.py develop