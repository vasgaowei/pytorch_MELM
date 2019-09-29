#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 13:56:53 2019

@author: vasgaoweithu
"""

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

from . import roi_crop_cpu
from . import roi_crop_cuda

class RoICropFunction(Function):
    @staticmethod
    def forward(ctx, input1, input2):
        ctx.input1 = input1.clone()
        ctx.input2 = input2.clone()
        output = input2.new(input2.size()[0], input1.size()[1], input2.size()[1], input2.size()[2]).zero_()
        assert output.get_device() == input1.get_device(), "output and input1 must on the same device"
        assert output.get_device() == input2.get_device(), "output and input2 must on the same device"
        if input1.is_cuda:
            roi_crop_cuda.forward(input1, input2, output)
        else:
            roi_crop_cpu.forward(input1, input2, output)
        
        return output
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        grad_input1 = ctx.input1.new(ctx.input1.size()).zero_()
        grad_input2 = ctx.input2.new(ctx.input2.size()).zero_()
        if grad_output.is_cuda:
            roi_crop_cuda.backward(ctx.input1, ctx.input2, grad_input1, grad_input2, grad_output)
        else:
            roi_crop_cpu.backward(ctx.input1, ctx.input2, grad_input1, grad_input2, grad_output)
        
        return grad_input1, grad_input2
    
roi_crop = RoICropFunction.apply

class RoICrop(nn.Module):
    def __init__(self, layout='BHWD'):
        super(RoICrop, self).__init__()
    def forward(self, input1, input2):
        return roi_crop(input1, input2)