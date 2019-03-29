#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 15:17:33 2018

@author: vasgaoweithu
"""

import os.path as osp
import sys
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
this_dir = osp.dirname(__file__)

lib_path = osp.join(this_dir, '..', 'lib')
add_path(lib_path)

coco_path = osp.join(this_dir, '..', 'data', 'coco', 'PythonAPI')
add_path(coco_path)
