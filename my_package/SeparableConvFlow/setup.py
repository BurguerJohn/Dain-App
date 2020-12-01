#!/usr/bin/env python3
import os
import torch
from Arch import GetArchs

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cxx_args = ['-std=c++14']

nvcc_args = GetArchs()

setup(
    name='separableconvflow_cuda',
    ext_modules=[
        CUDAExtension('separableconvflow_cuda', [
            'separableconvflow_cuda.cc',
            'separableconvflow_cuda_kernel.cu'
        ], extra_compile_args={'cxx': cxx_args, 'nvcc': nvcc_args})
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
