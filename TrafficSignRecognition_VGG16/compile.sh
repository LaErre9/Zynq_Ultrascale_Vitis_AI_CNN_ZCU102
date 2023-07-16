#!/bin/sh

if [ $1 = zcu102 ]; then
      ARCH=/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU102/arch.json
      TARGET=zcu102
      echo "-----------------------------------------"
      echo "COMPILING MODEL FOR ZCU102.."
      echo "-----------------------------------------"
elif [ $1 = zcu104 ]; then
      ARCH=/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json
      TARGET=zcu104
      echo "-----------------------------------------"
      echo "COMPILING MODEL FOR ZCU104.."
      echo "-----------------------------------------"
elif [ $1 = vck190 ]; then
      ARCH=/opt/vitis_ai/compiler/arch/DPUCVDX8G/VCK190/arch.json
      TARGET=vck190
      echo "-----------------------------------------"
      echo "COMPILING MODEL FOR VCK190.."
      echo "-----------------------------------------"
elif [ $1 = u50 ]; then
      ARCH=/opt/vitis_ai/compiler/arch/DPUCAHX8H/U50/arch.json
      TARGET=u50
      echo "-----------------------------------------"
      echo "COMPILING MODEL FOR ALVEO U50.."
      echo "-----------------------------------------"
else
      echo  "Target not found. Valid choices are: zcu102, zcu104, vck190, u50 ..exiting"
      exit 1
fi

BUILD=$2
LOG=$3

compile() {
  vai_c_xir \
  --xmodel      ${BUILD}/quant_model/CNN_int.xmodel \
  --arch        $ARCH \
  --net_name    CNN_${TARGET} \
  --output_dir  ${BUILD}/compiled_model
}

compile 2>&1 | tee ${LOG}/compile_$TARGET.log


echo "-----------------------------------------"
echo "MODEL COMPILED"
echo "-----------------------------------------"

