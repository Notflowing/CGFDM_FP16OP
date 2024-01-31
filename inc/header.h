#ifndef HEADER_H
#define HEADER_H
#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <memory.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

// half-precision float-point format
#ifdef FP16
#define FLOAT half
#include <cuda_fp16.h>
#include "overload.cuh"
#include "overload_half2.cuh"
//#include <cuda_bf16.h>
#else
#define FLOAT float
#endif
#include <cublas_v2.h>
#include "helper_cuda.h"
#include "helper_string.h"

#include <fstream>
#include <iostream>
#include <iomanip>
using namespace std;
#include <list>
#include <map>
#include <vector>

#if __GNUC__
#include <sys/stat.h>
#include <sys/types.h>
#elif _MSC_VER
#include <windows.h>
#include <direct.h>
#endif

#include <proj.h>
#include "macro.h"
#include "macro_half2.h"
#include "struct.h"
#include "cJSON.h"
#include "functions.h"

#endif  // HEADER_H

