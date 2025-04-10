#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import torch
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, ROCM_HOME
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
# Include this line immediately after the import statements
TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])
is_rocm = False
if TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 5):
  is_rocm = True if ((torch.version.hip is not None) and (ROCM_HOME is not None)) else False

def get_sources():
  if is_rocm:
    return ["hip_rasterizer/rasterizer_impl.hip", "hip_rasterizer/forward.hip", "hip_rasterizer/backward.hip", "rasterize_points.hip", "ext.cpp"]
  else:
    return ["cuda_rasterizer/rasterizer_impl.cu", "cuda_rasterizer/forward.cu", "cuda_rasterizer/backward.cu", "rasterize_points.cu", "ext.cpp"]
  
glm_include_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")
setup(
    name="diff_gaussian_rasterization",
    packages=['diff_gaussian_rasterization'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization._C",
            sources=get_sources(),
            extra_compile_args={"nvcc":["-I"+glm_include_dir], "cxx":["-I"+glm_include_dir]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    },
    version='1.0.0'
)
