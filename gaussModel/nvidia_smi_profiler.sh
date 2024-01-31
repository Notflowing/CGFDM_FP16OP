#!/bin/bash

# nvidia-smi -q -i 0,1,2,3 -d MEMORY,UTILIZATION,TEMPERATURE,POWER

# nvidia-smi -q -i 4,5,6,7 -lms 10 -f profile_query.log \
#     -d MEMORY,UTILIZATION,TEMPERATURE,POWER

# nvidia-smi --format=csv -i 4,5,6,7 -lms 10 -f profile_query_gpu.log \
#     --query-gpu=count,gpu_name,index,memory.total,memory.used,utilization.gpu,utilization.memory,power.management,power.draw

# nvidia-smi dmon -i 4,5,6,7 -d 1 -f profile_dmon.log \
#     -s pum

# nvidia-smi pmon -i 4,5,6,7 -d 1 -f profile_pmon.log \
#     -s um

# ################## 1 GPU ##################
# nvidia-smi --format=csv -i 7 -lms 10 -f profilersDir/profile_fp32_singleGPU.log \
#     --query-gpu=count,gpu_name,index,memory.total,memory.used,utilization.gpu,utilization.memory,power.management,power.draw

# nvidia-smi --format=csv -i 7 -lms 10 -f profilersDir/profile_fp16_singleGPU.log \
#     --query-gpu=count,gpu_name,index,memory.total,memory.used,utilization.gpu,utilization.memory,power.management,power.draw

# nvidia-smi --format=csv -i 7 -lms 10 -f profilersDir/profile_half2_singleGPU.log \
#     --query-gpu=count,gpu_name,index,memory.total,memory.used,utilization.gpu,utilization.memory,power.management,power.draw

# ################## 2 GPU ##################
# nvidia-smi --format=csv -i 6,7 -lms 10 -f profilersDir/profile_fp32_2GPU.log \
#     --query-gpu=count,gpu_name,index,memory.total,memory.used,utilization.gpu,utilization.memory,power.management,power.draw

# nvidia-smi --format=csv -i 6,7 -lms 10 -f profilersDir/profile_fp16_2GPU.log \
#     --query-gpu=count,gpu_name,index,memory.total,memory.used,utilization.gpu,utilization.memory,power.management,power.draw

# nvidia-smi --format=csv -i 6,7 -lms 10 -f profilersDir/profile_half2_2GPU.log \
#     --query-gpu=count,gpu_name,index,memory.total,memory.used,utilization.gpu,utilization.memory,power.management,power.draw

# ################## 4 GPU ##################
# nvidia-smi --format=csv -i 4,5,6,7 -lms 10 -f profilersDir/profile_fp32_4GPU.log  \
#     --query-gpu=count,gpu_name,index,memory.total,memory.used,utilization.gpu,utilization.memory,power.management,power.draw

# nvidia-smi --format=csv -i 4,5,6,7 -lms 10 -f profilersDir/profile_fp16_4GPU.log \
#     --query-gpu=count,gpu_name,index,memory.total,memory.used,utilization.gpu,utilization.memory,power.management,power.draw

nvidia-smi --format=csv -i 4,5,6,7 -lms 10 -f profilersDir/profile_half2_4GPU.log \
    --query-gpu=count,gpu_name,index,memory.total,memory.used,utilization.gpu,utilization.memory,power.management,power.draw


# ################## 8 GPU ##################
# nvidia-smi --format=csv -i 0,1,2,3,4,5,6,7 -lms 10 -f profilersDir/profile_fp32_8GPU.log \
#     --query-gpu=count,gpu_name,index,memory.total,memory.used,utilization.gpu,utilization.memory,power.management,power.draw

# nvidia-smi --format=csv -i 0,1,2,3,4,5,6,7 -lms 10 -f profilersDir/profile_fp16_8GPU.log \
#     --query-gpu=count,gpu_name,index,memory.total,memory.used,utilization.gpu,utilization.memory,power.management,power.draw

# nvidia-smi --format=csv -i 0,1,2,3,4,5,6,7 -lms 10 -f profilersDir/profile_half2_8GPU.log \
#     --query-gpu=count,gpu_name,index,memory.total,memory.used,utilization.gpu,utilization.memory,power.management,power.draw
