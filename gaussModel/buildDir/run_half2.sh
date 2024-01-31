#!/bin/bash

MPIHOME=/public/software/openmpi-4.1.1-cuda.10
# CUDAHOME=/public/software/cuda-11.5
CUDAHOME=/data0/home/wjl/software/cuda-11.0

export LD_LIBRARY_PATH=/public/software/proj-8.1.0/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/public/software/sqlite3/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${MPIHOME}/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${CUDAHOME}/lib64:${LD_LIBRARY_PATH}
export PROJ_LIB=/public/software/proj-8.1.0/share/proj

PX=`cat paramsDir/paramsCGFDM3D_half2.json | grep "\"PX\"" | tr -cd "[0-9]"`
PY=`cat paramsDir/paramsCGFDM3D_half2.json | grep "\"PY\"" | tr -cd "[0-9]"`
PZ=`cat paramsDir/paramsCGFDM3D_half2.json | grep "\"PZ\"" | tr -cd "[0-9]"`

RUN=${MPIHOME}/bin/mpirun

${RUN} -np $(($PX*$PY*$PZ)) ./bin/CGFDM3D_half2 | tee ./logDir/log_half2

# nsys profile --trace=cuda,osrt,nvtx,mpi --mpi-impl=openmpi -f true -o nsight/profile0311-CGFDM ${RUN} -np $(($PX*$PY*$PZ)) ./bin/CGFDM3D-half2
# nsys stats nsight/profile0311-CGFDM.qdrep | tee nsight/profile0311-CGFDM.log