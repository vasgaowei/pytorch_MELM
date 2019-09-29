#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 11:27:44 2019

@author: vasgaoweithu
"""

import os
import platform
from setuptools import Extension, dist, find_packages, setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension
dist.Distribution().fetch_build_eggs(['Cython', 'numpy>=1.11.1'])
import numpy as np
from Cython.Build import cythonize


def make_cuda_ext(name, module, sources):

    return CUDAExtension(
        name='{}.{}'.format(module, name),
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        extra_compile_args={
            'cxx': [],
            'nvcc': [
                '-D__CUDA_NO_HALF_OPERATORS__',
                '-D__CUDA_NO_HALF_CONVERSIONS__',
                '-D__CUDA_NO_HALF2_OPERATORS__',
            ]
        })

    
def make_cython_ext(name, module, sources):
    extra_compile_args = None
    if platform.system() != 'Windows':
        extra_compile_args = {
            'cxx': ['-Wno-unused-function', '-Wno-write-strings']
        }

    extension = Extension(
        '{}.{}'.format(module, name),
        [os.path.join(*module.split('.'), p) for p in sources],
        include_dirs=[np.get_include()],
        language='c++',
        extra_compile_args=extra_compile_args)
    extension, = cythonize(extension)
    return extension

if __name__ == '__main__':
    setup(
        name="faster rcnn",
        version="0.1",
        author="GaoWei",
        description="Weakly Supervised Obejct Detection",
        packages=find_packages(exclude=("cfgs", "tools")),
        ext_modules=[
# =============================================================================
#             make_cython_ext(
#                 name='soft_nms_cpu',
#                 module='ops.nms',
#                 sources=['src/soft_nms_cpu.pyx']),
#             make_cuda_ext(
#                 name='nms_cpu',
#                 module='ops.nms',
#                 sources=['src/nms_cpu.cpp']),
#             make_cuda_ext(
#                 name='nms_cuda',
#                 module='ops.nms',
#                 sources=['src/nms_cuda.cpp', 'src/nms_kernel.cu']),
#             make_cuda_ext(
#                 name='roi_align_cuda',
#                 module='ops.roi_align',
#                 sources=['src/roi_align_cuda.cpp', 'src/roi_align_kernel.cu']),
#             make_cuda_ext(
#                 name='roi_pool_cuda',
#                 module='ops.roi_pool',
#                 sources=['src/roi_pool_cuda.cpp', 'src/roi_pool_kernel.cu']),
#             make_cuda_ext(
#                     name='roi_crop_cpu',
#                     module='ops.roi_crop',
#                     sources=['src/roi_crop_cpu.cpp']
#                     ),
#             make_cuda_ext(
#                     name='roi_crop_cuda',
#                     module='ops.roi_crop',
#                     sources=['src/roi_crop_cuda.cpp', 'src/roi_crop_kernel.cu']
#                     ),
# =============================================================================
            make_cuda_ext(
                    name='roi_ring_pool_cuda',
                    module='ops.roi_ring_pool',
                    sources=['src/roi_ring_pool_cuda.cpp', 'src/roi_ring_pool_kernel.cu']
                    )
                    ],
            cmdclass={'build_ext': BuildExtension},
            zip_safe=False
            )