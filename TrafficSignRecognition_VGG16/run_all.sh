#!/bin/bash

conda activate vitis-ai-pytorch

# folders
export BUILD=./build
export LOG=${BUILD}/logs
mkdir -p ${LOG}


# quantize & export quantized model
python -u quantize.py -d ${BUILD} --quant_mode calib 2>&1 | tee ${LOG}/quant_calib.log
python -u quantize.py -d ${BUILD} --quant_mode test  2>&1 | tee ${LOG}/quant_test.log


# generate the XIR Graph of the xmodel
xdputil xmodel -p ./build/quant_model/XIR_Graph.png ./build/quant_model/CNN_int.xmodel

# compile for target boards
source compile.sh zcu102 ${BUILD} ${LOG}

# make target folders
python -u target.py --target zcu102 -d ${BUILD} 2>&1 | tee ${LOG}/target_zcu102.log
