#!/bin/bash

set -e

make clean -f ./buildDir/Makefile_fp32
make -j -f ./buildDir/Makefile_fp32
bash ./buildDir/run_fp32.sh

make clean -f ./buildDir/Makefile_fp16
make -j -f ./buildDir/Makefile_fp16
bash ./buildDir/run_fp16.sh

make clean -f ./buildDir/Makefile_half2
make -j -f ./buildDir/Makefile_half2
bash ./buildDir/run_half2.sh