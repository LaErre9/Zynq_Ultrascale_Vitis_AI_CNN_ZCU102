#!/usr/bin/env bash

if [[ "$CONDA_DEFAULT_ENV" = "base" ]]; then
  echo "WARNING: No conda environment has been activated."
fi

if [[ -z $VAI_HOME ]]; then
	export VAI_HOME="$( readlink -f "$( dirname "${BASH_SOURCE[0]}" )/../.." )"
fi

echo "------------------"
echo "VAI_HOME = $VAI_HOME"
echo "------------------"

source /opt/xilinx/xrt/setup.sh
echo "---------------------"
echo "XILINX_XRT = $XILINX_XRT"
echo "---------------------"

source /opt/xilinx/xrm/setup.sh
echo "---------------------"
echo "XILINX_XRM = $XILINX_XRM"
echo "---------------------"

echo "---------------------"
echo "LD_LIBRARY_PATH = $LD_LIBRARY_PATH"
echo "---------------------"

case $1 in

  DPUCAHX8H | dpuv3e)
    export XLNX_VART_FIRMWARE=/opt/xilinx/overlaybins/dpuv3e
    ;;
  
  DPUCAHX8L | dpuv3me)
    export XLNX_VART_FIRMWARE=/opt/xilinx/overlaybins/dpuv3me
    ;;
  
  DPUCADF8H | dpuv3int8)
    export XLNX_VART_FIRMWARE=/opt/xilinx/overlaybins/dpuv3int8
    ;;

  *)
    echo "Invalid argument $1!!!!"
    ;;
esac


echo "---------------------"
echo "XLNX_VART_FIRMWARE = $XLNX_VART_FIRMWARE"
echo "---------------------"
