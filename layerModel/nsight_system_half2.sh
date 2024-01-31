#!/bin/bash

set -e

make clean -f ./buildDir/Makefile_half2
make -j -f ./buildDir/Makefile_half2
bash ./buildDir/nsight-half2.sh