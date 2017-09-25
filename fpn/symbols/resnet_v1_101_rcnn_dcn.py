# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Guodong Zhang
# --------------------------------------------------------

import cPickle
import mxnet as mx
from utils.symbol import Symbol
from operator_py.proposal import *
from operator_py.proposal_target import *
from operator_py.box_annotator_ohem import *
from operator_py.assign_rois import *
from operator_py.focal_loss import *
from operator_py.check import *
from operator_py.assign import *
eps = 2e-5
bn_mom  = 0.9

class resnet_v1_101_rcnn_dcn(Symbol):
    def __init__(self):
        """
        Use __init__ to define parameter network needs
        """
        self.eps = 1e-5
        self.use_global_stats = True
        self.workspace = 512
        self.units = (3, 4, 23, 3)  # use for 101
        self.filter_list = [256, 512, 1024, 2048]

    def get_resnet_v1_conv2(self, data):
        conv1 = mx.symbol.Convolution(name='conv1', data=data, num_filter=64, pad=(3, 3), kernel=(7, 7), stride=(2, 2),
                                      no_bias=True)
        bn_conv1 = mx.symbol.BatchNorm(name='bn_conv1', data=conv1, use_global_stats=True, fix_gamma=False,
                                       eps=self.eps)
        scale_conv1 = bn_conv1
        conv1_relu = mx.symbol.Activation(name='conv1_relu', data=scale_conv1, act_type='relu')
        pool1 = mx.symbol.Pooling(name='pool1', data=conv1_relu, pooling_convention='full', pad=(0, 0), kernel=(3, 3),
                                  stride=(2, 2), pool_type='max')
        res2a_branch1 = mx.symbol.Convolution(name='res2a_branch1', data=pool1, num_filter=256, pad=(0, 0),
                                              kernel=(1, 1),
                                              stride=(1, 1), no_bias=True)
        bn2a_branch1 = mx.symbol.BatchNorm(name='bn2a_branch1', data=res2a_branch1, use_global_stats=True,
                                           fix_gamma=False, eps=self.eps)
        scale2a_branch1 = bn2a_branch1
        res2a_branch2a = mx.symbol.Convolution(name='res2a_branch2a', data=pool1, num_filter=64, pad=(0, 0),
                                               kernel=(1, 1),
                                               stride=(1, 1), no_bias=True)
        bn2a_branch2a = mx.symbol.BatchNorm(name='bn2a_branch2a', data=res2a_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale2a_branch2a = bn2a_branch2a
        res2a_branch2a_relu = mx.symbol.Activation(name='res2a_branch2a_relu', data=scale2a_branch2a, act_type='relu')
        res2a_branch2b = mx.symbol.Convolution(name='res2a_branch2b', data=res2a_branch2a_relu, num_filter=64,
                                               pad=(1, 1),
                                               kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn2a_branch2b = mx.symbol.BatchNorm(name='bn2a_branch2b', data=res2a_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale2a_branch2b = bn2a_branch2b
        res2a_branch2b_relu = mx.symbol.Activation(name='res2a_branch2b_relu', data=scale2a_branch2b, act_type='relu')
        res2a_branch2c = mx.symbol.Convolution(name='res2a_branch2c', data=res2a_branch2b_relu, num_filter=256,
                                               pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2a_branch2c = mx.symbol.BatchNorm(name='bn2a_branch2c', data=res2a_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale2a_branch2c = bn2a_branch2c
        res2a = mx.symbol.broadcast_add(name='res2a', *[scale2a_branch1, scale2a_branch2c])
        res2a_relu = mx.symbol.Activation(name='res2a_relu', data=res2a, act_type='relu')
        res2b_branch2a = mx.symbol.Convolution(name='res2b_branch2a', data=res2a_relu, num_filter=64, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2b_branch2a = mx.symbol.BatchNorm(name='bn2b_branch2a', data=res2b_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale2b_branch2a = bn2b_branch2a
        res2b_branch2a_relu = mx.symbol.Activation(name='res2b_branch2a_relu', data=scale2b_branch2a, act_type='relu')
        res2b_branch2b = mx.symbol.Convolution(name='res2b_branch2b', data=res2b_branch2a_relu, num_filter=64,
                                               pad=(1, 1),
                                               kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn2b_branch2b = mx.symbol.BatchNorm(name='bn2b_branch2b', data=res2b_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale2b_branch2b = bn2b_branch2b
        res2b_branch2b_relu = mx.symbol.Activation(name='res2b_branch2b_relu', data=scale2b_branch2b, act_type='relu')
        res2b_branch2c = mx.symbol.Convolution(name='res2b_branch2c', data=res2b_branch2b_relu, num_filter=256,
                                               pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2b_branch2c = mx.symbol.BatchNorm(name='bn2b_branch2c', data=res2b_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale2b_branch2c = bn2b_branch2c
        res2b = mx.symbol.broadcast_add(name='res2b', *[res2a_relu, scale2b_branch2c])
        res2b_relu = mx.symbol.Activation(name='res2b_relu', data=res2b, act_type='relu')
        res2c_branch2a = mx.symbol.Convolution(name='res2c_branch2a', data=res2b_relu, num_filter=64, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2c_branch2a = mx.symbol.BatchNorm(name='bn2c_branch2a', data=res2c_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale2c_branch2a = bn2c_branch2a
        res2c_branch2a_relu = mx.symbol.Activation(name='res2c_branch2a_relu', data=scale2c_branch2a, act_type='relu')
        res2c_branch2b = mx.symbol.Convolution(name='res2c_branch2b', data=res2c_branch2a_relu, num_filter=64,
                                               pad=(1, 1),
                                               kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn2c_branch2b = mx.symbol.BatchNorm(name='bn2c_branch2b', data=res2c_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale2c_branch2b = bn2c_branch2b
        res2c_branch2b_relu = mx.symbol.Activation(name='res2c_branch2b_relu', data=scale2c_branch2b, act_type='relu')
        res2c_branch2c = mx.symbol.Convolution(name='res2c_branch2c', data=res2c_branch2b_relu, num_filter=256,
                                               pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2c_branch2c = mx.symbol.BatchNorm(name='bn2c_branch2c', data=res2c_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale2c_branch2c = bn2c_branch2c
        res2c = mx.symbol.broadcast_add(name='res2c', *[res2b_relu, scale2c_branch2c])
        res2c_relu = mx.symbol.Activation(name='res2c_relu', data=res2c, act_type='relu')
        return res2c_relu

    def get_resnet_v1_conv3(self, conv2):
        res3a_branch1 = mx.symbol.Convolution(name='res3a_branch1', data=conv2, num_filter=512, pad=(0, 0),
                                              kernel=(1, 1), stride=(2, 2), no_bias=True) 
        bn3a_branch1 = mx.symbol.BatchNorm(name='bn3a_branch1', data=res3a_branch1, use_global_stats=True,
                                           fix_gamma=False, eps=self.eps)
        scale3a_branch1 = bn3a_branch1
        res3a_branch2a = mx.symbol.Convolution(name='res3a_branch2a', data=conv2, num_filter=128, pad=(0, 0),
                                               kernel=(1, 1), stride=(2, 2), no_bias=True)
        bn3a_branch2a = mx.symbol.BatchNorm(name='bn3a_branch2a', data=res3a_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale3a_branch2a = bn3a_branch2a
        res3a_branch2a_relu = mx.symbol.Activation(name='res3a_branch2a_relu', data=scale3a_branch2a, act_type='relu')
        res3a_branch2b = mx.symbol.Convolution(name='res3a_branch2b', data=res3a_branch2a_relu, num_filter=128,
                                               pad=(1, 1),
                                               kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn3a_branch2b = mx.symbol.BatchNorm(name='bn3a_branch2b', data=res3a_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale3a_branch2b = bn3a_branch2b
        res3a_branch2b_relu = mx.symbol.Activation(name='res3a_branch2b_relu', data=scale3a_branch2b, act_type='relu')
        res3a_branch2c = mx.symbol.Convolution(name='res3a_branch2c', data=res3a_branch2b_relu, num_filter=512,
                                               pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3a_branch2c = mx.symbol.BatchNorm(name='bn3a_branch2c', data=res3a_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale3a_branch2c = bn3a_branch2c
        res3a = mx.symbol.broadcast_add(name='res3a', *[scale3a_branch1, scale3a_branch2c])
        res3a_relu = mx.symbol.Activation(name='res3a_relu', data=res3a, act_type='relu')
        res3b1_branch2a = mx.symbol.Convolution(name='res3b1_branch2a', data=res3a_relu, num_filter=128, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b1_branch2a = mx.symbol.BatchNorm(name='bn3b1_branch2a', data=res3b1_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale3b1_branch2a = bn3b1_branch2a
        res3b1_branch2a_relu = mx.symbol.Activation(name='res3b1_branch2a_relu', data=scale3b1_branch2a,
                                                    act_type='relu')
        res3b1_branch2b = mx.symbol.Convolution(name='res3b1_branch2b', data=res3b1_branch2a_relu, num_filter=128,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn3b1_branch2b = mx.symbol.BatchNorm(name='bn3b1_branch2b', data=res3b1_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale3b1_branch2b = bn3b1_branch2b
        res3b1_branch2b_relu = mx.symbol.Activation(name='res3b1_branch2b_relu', data=scale3b1_branch2b,
                                                    act_type='relu')
        res3b1_branch2c = mx.symbol.Convolution(name='res3b1_branch2c', data=res3b1_branch2b_relu, num_filter=512,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b1_branch2c = mx.symbol.BatchNorm(name='bn3b1_branch2c', data=res3b1_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale3b1_branch2c = bn3b1_branch2c
        res3b1 = mx.symbol.broadcast_add(name='res3b1', *[res3a_relu, scale3b1_branch2c])
        res3b1_relu = mx.symbol.Activation(name='res3b1_relu', data=res3b1, act_type='relu')
        res3b2_branch2a = mx.symbol.Convolution(name='res3b2_branch2a', data=res3b1_relu, num_filter=128, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b2_branch2a = mx.symbol.BatchNorm(name='bn3b2_branch2a', data=res3b2_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale3b2_branch2a = bn3b2_branch2a
        res3b2_branch2a_relu = mx.symbol.Activation(name='res3b2_branch2a_relu', data=scale3b2_branch2a,
                                                    act_type='relu')
        res3b2_branch2b = mx.symbol.Convolution(name='res3b2_branch2b', data=res3b2_branch2a_relu, num_filter=128,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn3b2_branch2b = mx.symbol.BatchNorm(name='bn3b2_branch2b', data=res3b2_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale3b2_branch2b = bn3b2_branch2b
        res3b2_branch2b_relu = mx.symbol.Activation(name='res3b2_branch2b_relu', data=scale3b2_branch2b,
                                                    act_type='relu')
        res3b2_branch2c = mx.symbol.Convolution(name='res3b2_branch2c', data=res3b2_branch2b_relu, num_filter=512,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b2_branch2c = mx.symbol.BatchNorm(name='bn3b2_branch2c', data=res3b2_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale3b2_branch2c = bn3b2_branch2c
        res3b2 = mx.symbol.broadcast_add(name='res3b2', *[res3b1_relu, scale3b2_branch2c])
        res3b2_relu = mx.symbol.Activation(name='res3b2_relu', data=res3b2, act_type='relu')
        res3b3_branch2a = mx.symbol.Convolution(name='res3b3_branch2a', data=res3b2_relu, num_filter=128, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b3_branch2a = mx.symbol.BatchNorm(name='bn3b3_branch2a', data=res3b3_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale3b3_branch2a = bn3b3_branch2a
        res3b3_branch2a_relu = mx.symbol.Activation(name='res3b3_branch2a_relu', data=scale3b3_branch2a,
                                                    act_type='relu')
        res3b3_branch2b = mx.symbol.Convolution(name='res3b3_branch2b', data=res3b3_branch2a_relu, num_filter=128,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn3b3_branch2b = mx.symbol.BatchNorm(name='bn3b3_branch2b', data=res3b3_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale3b3_branch2b = bn3b3_branch2b
        res3b3_branch2b_relu = mx.symbol.Activation(name='res3b3_branch2b_relu', data=scale3b3_branch2b,
                                                    act_type='relu')
        res3b3_branch2c = mx.symbol.Convolution(name='res3b3_branch2c', data=res3b3_branch2b_relu, num_filter=512,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b3_branch2c = mx.symbol.BatchNorm(name='bn3b3_branch2c', data=res3b3_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale3b3_branch2c = bn3b3_branch2c
        res3b3 = mx.symbol.broadcast_add(name='res3b3', *[res3b2_relu, scale3b3_branch2c])
        res3b3_relu = mx.symbol.Activation(name='res3b3_relu', data=res3b3, act_type='relu')
        return  res3b3_relu  



    def get_resnet_v1_conv4(self, conv3):
        res4a_branch1 = mx.symbol.Convolution(name='res4a_branch1', data=conv3, num_filter=1024, pad=(0, 0),
                                              kernel=(1, 1), stride=(2, 2), no_bias=True)
        bn4a_branch1 = mx.symbol.BatchNorm(name='bn4a_branch1', data=res4a_branch1, use_global_stats=True,
                                           fix_gamma=False, eps=self.eps)
        scale4a_branch1 = bn4a_branch1
        res4a_branch2a = mx.symbol.Convolution(name='res4a_branch2a', data=conv3, num_filter=256, pad=(0, 0),
                                               kernel=(1, 1), stride=(2, 2), no_bias=True)
        bn4a_branch2a = mx.symbol.BatchNorm(name='bn4a_branch2a', data=res4a_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale4a_branch2a = bn4a_branch2a
        res4a_branch2a_relu = mx.symbol.Activation(name='res4a_branch2a_relu', data=scale4a_branch2a, act_type='relu')
        res4a_branch2b = mx.symbol.Convolution(name='res4a_branch2b', data=res4a_branch2a_relu, num_filter=256,
                                               pad=(1, 1),
                                               kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4a_branch2b = mx.symbol.BatchNorm(name='bn4a_branch2b', data=res4a_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale4a_branch2b = bn4a_branch2b
        res4a_branch2b_relu = mx.symbol.Activation(name='res4a_branch2b_relu', data=scale4a_branch2b, act_type='relu')
        res4a_branch2c = mx.symbol.Convolution(name='res4a_branch2c', data=res4a_branch2b_relu, num_filter=1024,
                                               pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4a_branch2c = mx.symbol.BatchNorm(name='bn4a_branch2c', data=res4a_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale4a_branch2c = bn4a_branch2c
        res4a = mx.symbol.broadcast_add(name='res4a', *[scale4a_branch1, scale4a_branch2c])
        res4a_relu = mx.symbol.Activation(name='res4a_relu', data=res4a, act_type='relu')
        res4b1_branch2a = mx.symbol.Convolution(name='res4b1_branch2a', data=res4a_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b1_branch2a = mx.symbol.BatchNorm(name='bn4b1_branch2a', data=res4b1_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b1_branch2a = bn4b1_branch2a
        res4b1_branch2a_relu = mx.symbol.Activation(name='res4b1_branch2a_relu', data=scale4b1_branch2a,
                                                    act_type='relu')
        res4b1_branch2b = mx.symbol.Convolution(name='res4b1_branch2b', data=res4b1_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b1_branch2b = mx.symbol.BatchNorm(name='bn4b1_branch2b', data=res4b1_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b1_branch2b = bn4b1_branch2b
        res4b1_branch2b_relu = mx.symbol.Activation(name='res4b1_branch2b_relu', data=scale4b1_branch2b,
                                                    act_type='relu')
        res4b1_branch2c = mx.symbol.Convolution(name='res4b1_branch2c', data=res4b1_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b1_branch2c = mx.symbol.BatchNorm(name='bn4b1_branch2c', data=res4b1_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b1_branch2c = bn4b1_branch2c
        res4b1 = mx.symbol.broadcast_add(name='res4b1', *[res4a_relu, scale4b1_branch2c])
        res4b1_relu = mx.symbol.Activation(name='res4b1_relu', data=res4b1, act_type='relu')
        res4b2_branch2a = mx.symbol.Convolution(name='res4b2_branch2a', data=res4b1_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b2_branch2a = mx.symbol.BatchNorm(name='bn4b2_branch2a', data=res4b2_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b2_branch2a = bn4b2_branch2a
        res4b2_branch2a_relu = mx.symbol.Activation(name='res4b2_branch2a_relu', data=scale4b2_branch2a,
                                                    act_type='relu')
        res4b2_branch2b = mx.symbol.Convolution(name='res4b2_branch2b', data=res4b2_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b2_branch2b = mx.symbol.BatchNorm(name='bn4b2_branch2b', data=res4b2_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b2_branch2b = bn4b2_branch2b
        res4b2_branch2b_relu = mx.symbol.Activation(name='res4b2_branch2b_relu', data=scale4b2_branch2b,
                                                    act_type='relu')
        res4b2_branch2c = mx.symbol.Convolution(name='res4b2_branch2c', data=res4b2_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b2_branch2c = mx.symbol.BatchNorm(name='bn4b2_branch2c', data=res4b2_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b2_branch2c = bn4b2_branch2c
        res4b2 = mx.symbol.broadcast_add(name='res4b2', *[res4b1_relu, scale4b2_branch2c])
        res4b2_relu = mx.symbol.Activation(name='res4b2_relu', data=res4b2, act_type='relu')
        res4b3_branch2a = mx.symbol.Convolution(name='res4b3_branch2a', data=res4b2_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b3_branch2a = mx.symbol.BatchNorm(name='bn4b3_branch2a', data=res4b3_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b3_branch2a = bn4b3_branch2a
        res4b3_branch2a_relu = mx.symbol.Activation(name='res4b3_branch2a_relu', data=scale4b3_branch2a,
                                                    act_type='relu')
        res4b3_branch2b = mx.symbol.Convolution(name='res4b3_branch2b', data=res4b3_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b3_branch2b = mx.symbol.BatchNorm(name='bn4b3_branch2b', data=res4b3_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b3_branch2b = bn4b3_branch2b
        res4b3_branch2b_relu = mx.symbol.Activation(name='res4b3_branch2b_relu', data=scale4b3_branch2b,
                                                    act_type='relu')
        res4b3_branch2c = mx.symbol.Convolution(name='res4b3_branch2c', data=res4b3_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b3_branch2c = mx.symbol.BatchNorm(name='bn4b3_branch2c', data=res4b3_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b3_branch2c = bn4b3_branch2c
        res4b3 = mx.symbol.broadcast_add(name='res4b3', *[res4b2_relu, scale4b3_branch2c])
        res4b3_relu = mx.symbol.Activation(name='res4b3_relu', data=res4b3, act_type='relu')
        res4b4_branch2a = mx.symbol.Convolution(name='res4b4_branch2a', data=res4b3_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b4_branch2a = mx.symbol.BatchNorm(name='bn4b4_branch2a', data=res4b4_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b4_branch2a = bn4b4_branch2a
        res4b4_branch2a_relu = mx.symbol.Activation(name='res4b4_branch2a_relu', data=scale4b4_branch2a,
                                                    act_type='relu')
        res4b4_branch2b = mx.symbol.Convolution(name='res4b4_branch2b', data=res4b4_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b4_branch2b = mx.symbol.BatchNorm(name='bn4b4_branch2b', data=res4b4_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b4_branch2b = bn4b4_branch2b
        res4b4_branch2b_relu = mx.symbol.Activation(name='res4b4_branch2b_relu', data=scale4b4_branch2b,
                                                    act_type='relu')
        res4b4_branch2c = mx.symbol.Convolution(name='res4b4_branch2c', data=res4b4_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b4_branch2c = mx.symbol.BatchNorm(name='bn4b4_branch2c', data=res4b4_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b4_branch2c = bn4b4_branch2c
        res4b4 = mx.symbol.broadcast_add(name='res4b4', *[res4b3_relu, scale4b4_branch2c])
        res4b4_relu = mx.symbol.Activation(name='res4b4_relu', data=res4b4, act_type='relu')
        res4b5_branch2a = mx.symbol.Convolution(name='res4b5_branch2a', data=res4b4_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b5_branch2a = mx.symbol.BatchNorm(name='bn4b5_branch2a', data=res4b5_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b5_branch2a = bn4b5_branch2a
        res4b5_branch2a_relu = mx.symbol.Activation(name='res4b5_branch2a_relu', data=scale4b5_branch2a,
                                                    act_type='relu')
        res4b5_branch2b = mx.symbol.Convolution(name='res4b5_branch2b', data=res4b5_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b5_branch2b = mx.symbol.BatchNorm(name='bn4b5_branch2b', data=res4b5_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b5_branch2b = bn4b5_branch2b
        res4b5_branch2b_relu = mx.symbol.Activation(name='res4b5_branch2b_relu', data=scale4b5_branch2b,
                                                    act_type='relu')
        res4b5_branch2c = mx.symbol.Convolution(name='res4b5_branch2c', data=res4b5_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b5_branch2c = mx.symbol.BatchNorm(name='bn4b5_branch2c', data=res4b5_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b5_branch2c = bn4b5_branch2c
        res4b5 = mx.symbol.broadcast_add(name='res4b5', *[res4b4_relu, scale4b5_branch2c])
        res4b5_relu = mx.symbol.Activation(name='res4b5_relu', data=res4b5, act_type='relu')
        res4b6_branch2a = mx.symbol.Convolution(name='res4b6_branch2a', data=res4b5_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b6_branch2a = mx.symbol.BatchNorm(name='bn4b6_branch2a', data=res4b6_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b6_branch2a = bn4b6_branch2a
        res4b6_branch2a_relu = mx.symbol.Activation(name='res4b6_branch2a_relu', data=scale4b6_branch2a,
                                                    act_type='relu')
        res4b6_branch2b = mx.symbol.Convolution(name='res4b6_branch2b', data=res4b6_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b6_branch2b = mx.symbol.BatchNorm(name='bn4b6_branch2b', data=res4b6_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b6_branch2b = bn4b6_branch2b
        res4b6_branch2b_relu = mx.symbol.Activation(name='res4b6_branch2b_relu', data=scale4b6_branch2b,
                                                    act_type='relu')
        res4b6_branch2c = mx.symbol.Convolution(name='res4b6_branch2c', data=res4b6_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b6_branch2c = mx.symbol.BatchNorm(name='bn4b6_branch2c', data=res4b6_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b6_branch2c = bn4b6_branch2c
        res4b6 = mx.symbol.broadcast_add(name='res4b6', *[res4b5_relu, scale4b6_branch2c])
        res4b6_relu = mx.symbol.Activation(name='res4b6_relu', data=res4b6, act_type='relu')
        res4b7_branch2a = mx.symbol.Convolution(name='res4b7_branch2a', data=res4b6_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b7_branch2a = mx.symbol.BatchNorm(name='bn4b7_branch2a', data=res4b7_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b7_branch2a = bn4b7_branch2a
        res4b7_branch2a_relu = mx.symbol.Activation(name='res4b7_branch2a_relu', data=scale4b7_branch2a,
                                                    act_type='relu')
        res4b7_branch2b = mx.symbol.Convolution(name='res4b7_branch2b', data=res4b7_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b7_branch2b = mx.symbol.BatchNorm(name='bn4b7_branch2b', data=res4b7_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b7_branch2b = bn4b7_branch2b
        res4b7_branch2b_relu = mx.symbol.Activation(name='res4b7_branch2b_relu', data=scale4b7_branch2b,
                                                    act_type='relu')
        res4b7_branch2c = mx.symbol.Convolution(name='res4b7_branch2c', data=res4b7_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b7_branch2c = mx.symbol.BatchNorm(name='bn4b7_branch2c', data=res4b7_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b7_branch2c = bn4b7_branch2c
        res4b7 = mx.symbol.broadcast_add(name='res4b7', *[res4b6_relu, scale4b7_branch2c])
        res4b7_relu = mx.symbol.Activation(name='res4b7_relu', data=res4b7, act_type='relu')
        res4b8_branch2a = mx.symbol.Convolution(name='res4b8_branch2a', data=res4b7_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b8_branch2a = mx.symbol.BatchNorm(name='bn4b8_branch2a', data=res4b8_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b8_branch2a = bn4b8_branch2a
        res4b8_branch2a_relu = mx.symbol.Activation(name='res4b8_branch2a_relu', data=scale4b8_branch2a,
                                                    act_type='relu')
        res4b8_branch2b = mx.symbol.Convolution(name='res4b8_branch2b', data=res4b8_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b8_branch2b = mx.symbol.BatchNorm(name='bn4b8_branch2b', data=res4b8_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b8_branch2b = bn4b8_branch2b
        res4b8_branch2b_relu = mx.symbol.Activation(name='res4b8_branch2b_relu', data=scale4b8_branch2b,
                                                    act_type='relu')
        res4b8_branch2c = mx.symbol.Convolution(name='res4b8_branch2c', data=res4b8_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b8_branch2c = mx.symbol.BatchNorm(name='bn4b8_branch2c', data=res4b8_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b8_branch2c = bn4b8_branch2c
        res4b8 = mx.symbol.broadcast_add(name='res4b8', *[res4b7_relu, scale4b8_branch2c])
        res4b8_relu = mx.symbol.Activation(name='res4b8_relu', data=res4b8, act_type='relu')
        res4b9_branch2a = mx.symbol.Convolution(name='res4b9_branch2a', data=res4b8_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b9_branch2a = mx.symbol.BatchNorm(name='bn4b9_branch2a', data=res4b9_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b9_branch2a = bn4b9_branch2a
        res4b9_branch2a_relu = mx.symbol.Activation(name='res4b9_branch2a_relu', data=scale4b9_branch2a,
                                                    act_type='relu')
        res4b9_branch2b = mx.symbol.Convolution(name='res4b9_branch2b', data=res4b9_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b9_branch2b = mx.symbol.BatchNorm(name='bn4b9_branch2b', data=res4b9_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b9_branch2b = bn4b9_branch2b
        res4b9_branch2b_relu = mx.symbol.Activation(name='res4b9_branch2b_relu', data=scale4b9_branch2b,
                                                    act_type='relu')
        res4b9_branch2c = mx.symbol.Convolution(name='res4b9_branch2c', data=res4b9_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b9_branch2c = mx.symbol.BatchNorm(name='bn4b9_branch2c', data=res4b9_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b9_branch2c = bn4b9_branch2c
        res4b9 = mx.symbol.broadcast_add(name='res4b9', *[res4b8_relu, scale4b9_branch2c])
        res4b9_relu = mx.symbol.Activation(name='res4b9_relu', data=res4b9, act_type='relu')
        res4b10_branch2a = mx.symbol.Convolution(name='res4b10_branch2a', data=res4b9_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b10_branch2a = mx.symbol.BatchNorm(name='bn4b10_branch2a', data=res4b10_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b10_branch2a = bn4b10_branch2a
        res4b10_branch2a_relu = mx.symbol.Activation(name='res4b10_branch2a_relu', data=scale4b10_branch2a,
                                                     act_type='relu')
        res4b10_branch2b = mx.symbol.Convolution(name='res4b10_branch2b', data=res4b10_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b10_branch2b = mx.symbol.BatchNorm(name='bn4b10_branch2b', data=res4b10_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b10_branch2b = bn4b10_branch2b
        res4b10_branch2b_relu = mx.symbol.Activation(name='res4b10_branch2b_relu', data=scale4b10_branch2b,
                                                     act_type='relu')
        res4b10_branch2c = mx.symbol.Convolution(name='res4b10_branch2c', data=res4b10_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b10_branch2c = mx.symbol.BatchNorm(name='bn4b10_branch2c', data=res4b10_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b10_branch2c = bn4b10_branch2c
        res4b10 = mx.symbol.broadcast_add(name='res4b10', *[res4b9_relu, scale4b10_branch2c])
        res4b10_relu = mx.symbol.Activation(name='res4b10_relu', data=res4b10, act_type='relu')
        res4b11_branch2a = mx.symbol.Convolution(name='res4b11_branch2a', data=res4b10_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b11_branch2a = mx.symbol.BatchNorm(name='bn4b11_branch2a', data=res4b11_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b11_branch2a = bn4b11_branch2a
        res4b11_branch2a_relu = mx.symbol.Activation(name='res4b11_branch2a_relu', data=scale4b11_branch2a,
                                                     act_type='relu')
        res4b11_branch2b = mx.symbol.Convolution(name='res4b11_branch2b', data=res4b11_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b11_branch2b = mx.symbol.BatchNorm(name='bn4b11_branch2b', data=res4b11_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b11_branch2b = bn4b11_branch2b
        res4b11_branch2b_relu = mx.symbol.Activation(name='res4b11_branch2b_relu', data=scale4b11_branch2b,
                                                     act_type='relu')
        res4b11_branch2c = mx.symbol.Convolution(name='res4b11_branch2c', data=res4b11_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b11_branch2c = mx.symbol.BatchNorm(name='bn4b11_branch2c', data=res4b11_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b11_branch2c = bn4b11_branch2c
        res4b11 = mx.symbol.broadcast_add(name='res4b11', *[res4b10_relu, scale4b11_branch2c])
        res4b11_relu = mx.symbol.Activation(name='res4b11_relu', data=res4b11, act_type='relu')
        res4b12_branch2a = mx.symbol.Convolution(name='res4b12_branch2a', data=res4b11_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b12_branch2a = mx.symbol.BatchNorm(name='bn4b12_branch2a', data=res4b12_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b12_branch2a = bn4b12_branch2a
        res4b12_branch2a_relu = mx.symbol.Activation(name='res4b12_branch2a_relu', data=scale4b12_branch2a,
                                                     act_type='relu')
        res4b12_branch2b = mx.symbol.Convolution(name='res4b12_branch2b', data=res4b12_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b12_branch2b = mx.symbol.BatchNorm(name='bn4b12_branch2b', data=res4b12_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b12_branch2b = bn4b12_branch2b
        res4b12_branch2b_relu = mx.symbol.Activation(name='res4b12_branch2b_relu', data=scale4b12_branch2b,
                                                     act_type='relu')
        res4b12_branch2c = mx.symbol.Convolution(name='res4b12_branch2c', data=res4b12_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b12_branch2c = mx.symbol.BatchNorm(name='bn4b12_branch2c', data=res4b12_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b12_branch2c = bn4b12_branch2c
        res4b12 = mx.symbol.broadcast_add(name='res4b12', *[res4b11_relu, scale4b12_branch2c])
        res4b12_relu = mx.symbol.Activation(name='res4b12_relu', data=res4b12, act_type='relu')
        res4b13_branch2a = mx.symbol.Convolution(name='res4b13_branch2a', data=res4b12_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b13_branch2a = mx.symbol.BatchNorm(name='bn4b13_branch2a', data=res4b13_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b13_branch2a = bn4b13_branch2a
        res4b13_branch2a_relu = mx.symbol.Activation(name='res4b13_branch2a_relu', data=scale4b13_branch2a,
                                                     act_type='relu')
        res4b13_branch2b = mx.symbol.Convolution(name='res4b13_branch2b', data=res4b13_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b13_branch2b = mx.symbol.BatchNorm(name='bn4b13_branch2b', data=res4b13_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b13_branch2b = bn4b13_branch2b
        res4b13_branch2b_relu = mx.symbol.Activation(name='res4b13_branch2b_relu', data=scale4b13_branch2b,
                                                     act_type='relu')
        res4b13_branch2c = mx.symbol.Convolution(name='res4b13_branch2c', data=res4b13_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b13_branch2c = mx.symbol.BatchNorm(name='bn4b13_branch2c', data=res4b13_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b13_branch2c = bn4b13_branch2c
        res4b13 = mx.symbol.broadcast_add(name='res4b13', *[res4b12_relu, scale4b13_branch2c])
        res4b13_relu = mx.symbol.Activation(name='res4b13_relu', data=res4b13, act_type='relu')
        res4b14_branch2a = mx.symbol.Convolution(name='res4b14_branch2a', data=res4b13_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b14_branch2a = mx.symbol.BatchNorm(name='bn4b14_branch2a', data=res4b14_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b14_branch2a = bn4b14_branch2a
        res4b14_branch2a_relu = mx.symbol.Activation(name='res4b14_branch2a_relu', data=scale4b14_branch2a,
                                                     act_type='relu')
        res4b14_branch2b = mx.symbol.Convolution(name='res4b14_branch2b', data=res4b14_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b14_branch2b = mx.symbol.BatchNorm(name='bn4b14_branch2b', data=res4b14_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b14_branch2b = bn4b14_branch2b
        res4b14_branch2b_relu = mx.symbol.Activation(name='res4b14_branch2b_relu', data=scale4b14_branch2b,
                                                     act_type='relu')
        res4b14_branch2c = mx.symbol.Convolution(name='res4b14_branch2c', data=res4b14_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b14_branch2c = mx.symbol.BatchNorm(name='bn4b14_branch2c', data=res4b14_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b14_branch2c = bn4b14_branch2c
        res4b14 = mx.symbol.broadcast_add(name='res4b14', *[res4b13_relu, scale4b14_branch2c])
        res4b14_relu = mx.symbol.Activation(name='res4b14_relu', data=res4b14, act_type='relu')
        res4b15_branch2a = mx.symbol.Convolution(name='res4b15_branch2a', data=res4b14_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b15_branch2a = mx.symbol.BatchNorm(name='bn4b15_branch2a', data=res4b15_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b15_branch2a = bn4b15_branch2a
        res4b15_branch2a_relu = mx.symbol.Activation(name='res4b15_branch2a_relu', data=scale4b15_branch2a,
                                                     act_type='relu')
        res4b15_branch2b = mx.symbol.Convolution(name='res4b15_branch2b', data=res4b15_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b15_branch2b = mx.symbol.BatchNorm(name='bn4b15_branch2b', data=res4b15_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b15_branch2b = bn4b15_branch2b
        res4b15_branch2b_relu = mx.symbol.Activation(name='res4b15_branch2b_relu', data=scale4b15_branch2b,
                                                     act_type='relu')
        res4b15_branch2c = mx.symbol.Convolution(name='res4b15_branch2c', data=res4b15_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b15_branch2c = mx.symbol.BatchNorm(name='bn4b15_branch2c', data=res4b15_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b15_branch2c = bn4b15_branch2c
        res4b15 = mx.symbol.broadcast_add(name='res4b15', *[res4b14_relu, scale4b15_branch2c])
        res4b15_relu = mx.symbol.Activation(name='res4b15_relu', data=res4b15, act_type='relu')
        res4b16_branch2a = mx.symbol.Convolution(name='res4b16_branch2a', data=res4b15_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b16_branch2a = mx.symbol.BatchNorm(name='bn4b16_branch2a', data=res4b16_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b16_branch2a = bn4b16_branch2a
        res4b16_branch2a_relu = mx.symbol.Activation(name='res4b16_branch2a_relu', data=scale4b16_branch2a,
                                                     act_type='relu')
        res4b16_branch2b = mx.symbol.Convolution(name='res4b16_branch2b', data=res4b16_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b16_branch2b = mx.symbol.BatchNorm(name='bn4b16_branch2b', data=res4b16_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b16_branch2b = bn4b16_branch2b
        res4b16_branch2b_relu = mx.symbol.Activation(name='res4b16_branch2b_relu', data=scale4b16_branch2b,
                                                     act_type='relu')
        res4b16_branch2c = mx.symbol.Convolution(name='res4b16_branch2c', data=res4b16_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b16_branch2c = mx.symbol.BatchNorm(name='bn4b16_branch2c', data=res4b16_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b16_branch2c = bn4b16_branch2c
        res4b16 = mx.symbol.broadcast_add(name='res4b16', *[res4b15_relu, scale4b16_branch2c])
        res4b16_relu = mx.symbol.Activation(name='res4b16_relu', data=res4b16, act_type='relu')
        res4b17_branch2a = mx.symbol.Convolution(name='res4b17_branch2a', data=res4b16_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b17_branch2a = mx.symbol.BatchNorm(name='bn4b17_branch2a', data=res4b17_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b17_branch2a = bn4b17_branch2a
        res4b17_branch2a_relu = mx.symbol.Activation(name='res4b17_branch2a_relu', data=scale4b17_branch2a,
                                                     act_type='relu')
        res4b17_branch2b = mx.symbol.Convolution(name='res4b17_branch2b', data=res4b17_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b17_branch2b = mx.symbol.BatchNorm(name='bn4b17_branch2b', data=res4b17_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b17_branch2b = bn4b17_branch2b
        res4b17_branch2b_relu = mx.symbol.Activation(name='res4b17_branch2b_relu', data=scale4b17_branch2b,
                                                     act_type='relu')
        res4b17_branch2c = mx.symbol.Convolution(name='res4b17_branch2c', data=res4b17_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b17_branch2c = mx.symbol.BatchNorm(name='bn4b17_branch2c', data=res4b17_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b17_branch2c = bn4b17_branch2c
        res4b17 = mx.symbol.broadcast_add(name='res4b17', *[res4b16_relu, scale4b17_branch2c])
        res4b17_relu = mx.symbol.Activation(name='res4b17_relu', data=res4b17, act_type='relu')
        res4b18_branch2a = mx.symbol.Convolution(name='res4b18_branch2a', data=res4b17_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b18_branch2a = mx.symbol.BatchNorm(name='bn4b18_branch2a', data=res4b18_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b18_branch2a = bn4b18_branch2a
        res4b18_branch2a_relu = mx.symbol.Activation(name='res4b18_branch2a_relu', data=scale4b18_branch2a,
                                                     act_type='relu')
        res4b18_branch2b = mx.symbol.Convolution(name='res4b18_branch2b', data=res4b18_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b18_branch2b = mx.symbol.BatchNorm(name='bn4b18_branch2b', data=res4b18_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b18_branch2b = bn4b18_branch2b
        res4b18_branch2b_relu = mx.symbol.Activation(name='res4b18_branch2b_relu', data=scale4b18_branch2b,
                                                     act_type='relu')
        res4b18_branch2c = mx.symbol.Convolution(name='res4b18_branch2c', data=res4b18_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b18_branch2c = mx.symbol.BatchNorm(name='bn4b18_branch2c', data=res4b18_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b18_branch2c = bn4b18_branch2c
        res4b18 = mx.symbol.broadcast_add(name='res4b18', *[res4b17_relu, scale4b18_branch2c])
        res4b18_relu = mx.symbol.Activation(name='res4b18_relu', data=res4b18, act_type='relu')
        res4b19_branch2a = mx.symbol.Convolution(name='res4b19_branch2a', data=res4b18_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b19_branch2a = mx.symbol.BatchNorm(name='bn4b19_branch2a', data=res4b19_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b19_branch2a = bn4b19_branch2a
        res4b19_branch2a_relu = mx.symbol.Activation(name='res4b19_branch2a_relu', data=scale4b19_branch2a,
                                                     act_type='relu')
        res4b19_branch2b = mx.symbol.Convolution(name='res4b19_branch2b', data=res4b19_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b19_branch2b = mx.symbol.BatchNorm(name='bn4b19_branch2b', data=res4b19_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b19_branch2b = bn4b19_branch2b
        res4b19_branch2b_relu = mx.symbol.Activation(name='res4b19_branch2b_relu', data=scale4b19_branch2b,
                                                     act_type='relu')
        res4b19_branch2c = mx.symbol.Convolution(name='res4b19_branch2c', data=res4b19_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b19_branch2c = mx.symbol.BatchNorm(name='bn4b19_branch2c', data=res4b19_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b19_branch2c = bn4b19_branch2c
        res4b19 = mx.symbol.broadcast_add(name='res4b19', *[res4b18_relu, scale4b19_branch2c])
        res4b19_relu = mx.symbol.Activation(name='res4b19_relu', data=res4b19, act_type='relu')
        res4b20_branch2a = mx.symbol.Convolution(name='res4b20_branch2a', data=res4b19_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b20_branch2a = mx.symbol.BatchNorm(name='bn4b20_branch2a', data=res4b20_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b20_branch2a = bn4b20_branch2a
        res4b20_branch2a_relu = mx.symbol.Activation(name='res4b20_branch2a_relu', data=scale4b20_branch2a,
                                                     act_type='relu')
        res4b20_branch2b = mx.symbol.Convolution(name='res4b20_branch2b', data=res4b20_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b20_branch2b = mx.symbol.BatchNorm(name='bn4b20_branch2b', data=res4b20_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b20_branch2b = bn4b20_branch2b
        res4b20_branch2b_relu = mx.symbol.Activation(name='res4b20_branch2b_relu', data=scale4b20_branch2b,
                                                     act_type='relu')
        res4b20_branch2c = mx.symbol.Convolution(name='res4b20_branch2c', data=res4b20_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b20_branch2c = mx.symbol.BatchNorm(name='bn4b20_branch2c', data=res4b20_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b20_branch2c = bn4b20_branch2c
        res4b20 = mx.symbol.broadcast_add(name='res4b20', *[res4b19_relu, scale4b20_branch2c])
        res4b20_relu = mx.symbol.Activation(name='res4b20_relu', data=res4b20, act_type='relu')
        res4b21_branch2a = mx.symbol.Convolution(name='res4b21_branch2a', data=res4b20_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b21_branch2a = mx.symbol.BatchNorm(name='bn4b21_branch2a', data=res4b21_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b21_branch2a = bn4b21_branch2a
        res4b21_branch2a_relu = mx.symbol.Activation(name='res4b21_branch2a_relu', data=scale4b21_branch2a,
                                                     act_type='relu')
        res4b21_branch2b = mx.symbol.Convolution(name='res4b21_branch2b', data=res4b21_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b21_branch2b = mx.symbol.BatchNorm(name='bn4b21_branch2b', data=res4b21_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b21_branch2b = bn4b21_branch2b
        res4b21_branch2b_relu = mx.symbol.Activation(name='res4b21_branch2b_relu', data=scale4b21_branch2b,
                                                     act_type='relu')
        res4b21_branch2c = mx.symbol.Convolution(name='res4b21_branch2c', data=res4b21_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b21_branch2c = mx.symbol.BatchNorm(name='bn4b21_branch2c', data=res4b21_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b21_branch2c = bn4b21_branch2c
        res4b21 = mx.symbol.broadcast_add(name='res4b21', *[res4b20_relu, scale4b21_branch2c])
        res4b21_relu = mx.symbol.Activation(name='res4b21_relu', data=res4b21, act_type='relu')
        res4b22_branch2a = mx.symbol.Convolution(name='res4b22_branch2a', data=res4b21_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b22_branch2a = mx.symbol.BatchNorm(name='bn4b22_branch2a', data=res4b22_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b22_branch2a = bn4b22_branch2a
        res4b22_branch2a_relu = mx.symbol.Activation(name='res4b22_branch2a_relu', data=scale4b22_branch2a,
                                                     act_type='relu')
        res4b22_branch2b = mx.symbol.Convolution(name='res4b22_branch2b', data=res4b22_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b22_branch2b = mx.symbol.BatchNorm(name='bn4b22_branch2b', data=res4b22_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b22_branch2b = bn4b22_branch2b
        res4b22_branch2b_relu = mx.symbol.Activation(name='res4b22_branch2b_relu', data=scale4b22_branch2b,
                                                     act_type='relu')
        res4b22_branch2c = mx.symbol.Convolution(name='res4b22_branch2c', data=res4b22_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b22_branch2c = mx.symbol.BatchNorm(name='bn4b22_branch2c', data=res4b22_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b22_branch2c = bn4b22_branch2c
        res4b22 = mx.symbol.broadcast_add(name='res4b22', *[res4b21_relu, scale4b22_branch2c])
        res4b22_relu = mx.symbol.Activation(name='res4b22_relu', data=res4b22, act_type='relu')
        return res4b22_relu

    def get_resnet_v1_conv5(self, conv4):
        res5a_branch1 = mx.symbol.Convolution(name='res5a_branch1', data=conv4, num_filter=2048, pad=(0, 0),
                                              kernel=(1, 1), stride=(2, 2), no_bias=True)
        bn5a_branch1 = mx.symbol.BatchNorm(name='bn5a_branch1', data=res5a_branch1, use_global_stats=True, fix_gamma=False, eps=self.eps)
        scale5a_branch1 = bn5a_branch1
        res5a_branch2a = mx.symbol.Convolution(name='res5a_branch2a', data=conv4, num_filter=512, pad=(0, 0),
                                               kernel=(1, 1), stride=(2, 2), no_bias=True)
        bn5a_branch2a = mx.symbol.BatchNorm(name='bn5a_branch2a', data=res5a_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5a_branch2a = bn5a_branch2a

        res5a_branch2a_relu = mx.symbol.Activation(name='res5a_branch2a_relu', data=scale5a_branch2a, act_type='relu')


        # res5a_branch2b_offset = mx.symbol.Convolution(name='res5a_branch2b_offset', data = res5a_branch2a_relu,
        #                                               num_filter=72, pad=(2, 2), kernel=(3, 3), stride=(1, 1), dilate=(2, 2), cudnn_off=True)
        # res5a_branch2b = mx.contrib.symbol.DeformableConvolution(name='res5a_branch2b', data=res5a_branch2a_relu, offset=res5a_branch2b_offset,
        #                                                          num_filter=512, pad=(2, 2), kernel=(3, 3), num_deformable_group=4,
        #                                                          stride=(1, 1), dilate=(2, 2), no_bias=True)

        res5a_branch2b = mx.symbol.Convolution(name='res5a_branch2b', data=res5a_branch2a_relu, num_filter=512, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)

        bn5a_branch2b = mx.symbol.BatchNorm(name='bn5a_branch2b', data=res5a_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        
        scale5a_branch2b = bn5a_branch2b
        res5a_branch2b_relu = mx.symbol.Activation(name='res5a_branch2b_relu', data=scale5a_branch2b, act_type='relu')
        res5a_branch2c = mx.symbol.Convolution(name='res5a_branch2c', data=res5a_branch2b_relu, num_filter=2048, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5a_branch2c = mx.symbol.BatchNorm(name='bn5a_branch2c', data=res5a_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5a_branch2c = bn5a_branch2c
        res5a = mx.symbol.broadcast_add(name='res5a', *[scale5a_branch1, scale5a_branch2c])
        res5a_relu = mx.symbol.Activation(name='res5a_relu', data=res5a, act_type='relu')
        res5b_branch2a = mx.symbol.Convolution(name='res5b_branch2a', data=res5a_relu, num_filter=512, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5b_branch2a = mx.symbol.BatchNorm(name='bn5b_branch2a', data=res5b_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5b_branch2a = bn5b_branch2a
        res5b_branch2a_relu = mx.symbol.Activation(name='res5b_branch2a_relu', data=scale5b_branch2a, act_type='relu')

        # res5b_branch2b_offset = mx.symbol.Convolution(name='res5b_branch2b_offset', data = res5b_branch2a_relu,
        #                                               num_filter=72, pad=(2, 2), kernel=(3, 3), stride=(1, 1), dilate=(2, 2), cudnn_off=True)
        # res5b_branch2b = mx.contrib.symbol.DeformableConvolution(name='res5b_branch2b', data=res5b_branch2a_relu, offset=res5b_branch2b_offset,
        #                                                          num_filter=512, pad=(2, 2), kernel=(3, 3), num_deformable_group=4,
        #                                                          stride=(1, 1), dilate=(2, 2), no_bias=True)
        res5b_branch2b = mx.symbol.Convolution(name='res5b_branch2b', data=res5b_branch2a_relu, num_filter=512, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5b_branch2b = mx.symbol.BatchNorm(name='bn5b_branch2b', data=res5b_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5b_branch2b = bn5b_branch2b
        res5b_branch2b_relu = mx.symbol.Activation(name='res5b_branch2b_relu', data=scale5b_branch2b, act_type='relu')
        res5b_branch2c = mx.symbol.Convolution(name='res5b_branch2c', data=res5b_branch2b_relu, num_filter=2048, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5b_branch2c = mx.symbol.BatchNorm(name='bn5b_branch2c', data=res5b_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5b_branch2c = bn5b_branch2c
        res5b = mx.symbol.broadcast_add(name='res5b', *[res5a_relu, scale5b_branch2c])
        res5b_relu = mx.symbol.Activation(name='res5b_relu', data=res5b, act_type='relu')
        res5c_branch2a = mx.symbol.Convolution(name='res5c_branch2a', data=res5b_relu, num_filter=512, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5c_branch2a = mx.symbol.BatchNorm(name='bn5c_branch2a', data=res5c_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5c_branch2a = bn5c_branch2a
        res5c_branch2a_relu = mx.symbol.Activation(name='res5c_branch2a_relu', data=scale5c_branch2a, act_type='relu')

        # res5c_branch2b_offset = mx.symbol.Convolution(name='res5c_branch2b_offset', data = res5c_branch2a_relu,
        #                                               num_filter=72, pad=(2, 2), kernel=(3, 3), stride=(1, 1), dilate=(2, 2), cudnn_off=True)
        # res5c_branch2b = mx.contrib.symbol.DeformableConvolution(name='res5c_branch2b', data=res5c_branch2a_relu, offset=res5c_branch2b_offset,
        #                                                          num_filter=512, pad=(2, 2), kernel=(3, 3), num_deformable_group=4,
        #                                                          stride=(1, 1), dilate=(2, 2), no_bias=True)
    

        res5c_branch2b = mx.symbol.Convolution(name='res5c_branch2b', data=res5c_branch2a_relu, num_filter=512, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)

        bn5c_branch2b = mx.symbol.BatchNorm(name='bn5c_branch2b', data=res5c_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5c_branch2b = bn5c_branch2b
        res5c_branch2b_relu = mx.symbol.Activation(name='res5c_branch2b_relu', data=scale5c_branch2b, act_type='relu')
        res5c_branch2c = mx.symbol.Convolution(name='res5c_branch2c', data=res5c_branch2b_relu, num_filter=2048, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5c_branch2c = mx.symbol.BatchNorm(name='bn5c_branch2c', data=res5c_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5c_branch2c = bn5c_branch2c
        res5c = mx.symbol.broadcast_add(name='res5c', *[res5b_relu, scale5c_branch2c])
        res5c_relu = mx.symbol.Activation(name='res5c_relu', data=res5c, act_type='relu')
        res5c_relu = mx.sym.Custom(op_type='Check',  data=res5c_relu)
        
        return res5c_relu


    def get_rpn(self, conv_feat, num_anchors,p_str,bn1_box_gamma,bn1_box_beta,bn1_box_moving_mean,bn1_box_moving_var):
        rpn_conv = mx.sym.Convolution(
            data=conv_feat, kernel=(3, 3), pad=(1, 1), num_filter=512, name="rpn_conv_3x3"+p_str)
        rpn_conv = mx.sym.BatchNorm(data=rpn_conv, use_global_stats=False, eps=eps,gamma=bn1_box_gamma, beta =bn1_box_beta, moving_mean =bn1_box_moving_mean,
            moving_var= bn1_box_moving_var,momentum=bn_mom, name = 'bn1_box')           
        rpn_relu = mx.sym.Activation(data=rpn_conv, act_type="relu", name="rpn_relu"+p_str)
        rpn_cls_score = mx.sym.Convolution(
            data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name="rpn_cls_score"+p_str)
        rpn_bbox_pred = mx.sym.Convolution(
            data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name="rpn_bbox_pred"+p_str)
        return rpn_cls_score, rpn_bbox_pred

    def get_symbol(self, cfg, is_train=True):

#######################share#############################
        fpn_conv_weight = mx.symbol.Variable(name='fpn_conv_weight')
        fpn_conv_bias = mx.symbol.Variable(name='fpn_conv_bias')
        fpn_cls_weight = mx.symbol.Variable(name='fpn_cls_weight')
        fpn_cls_bias = mx.symbol.Variable(name='fpn_cls_bias')
        fpn_bbox_pred_weight = mx.symbol.Variable(name='fpn_bbox_pred_weight')
        fpn_bbox_pred_bias = mx.symbol.Variable(name='fpn_bbox_pred_bias')


        fc_new_1_weight = mx.symbol.Variable(name = 'fc_new_1_weight')
        fc_new_1_bias = mx.symbol.Variable(name = 'fc_new_1_bias')

        fc_new_2_weight = mx.symbol.Variable(name = 'fc_new_2_weight')
        fc_new_2_bias = mx.symbol.Variable(name = 'fc_new_2_bias')

        rcnn_cls_weight =  mx.symbol.Variable(name = 'rcnn_cls_weight')
        rcnn_cls_bias =  mx.symbol.Variable(name = 'rcnn_cls_bias')

        rcnn_bbox_weight =  mx.symbol.Variable(name = 'rcnn_bbox_weight')
        rcnn_bbox_bias =  mx.symbol.Variable(name = 'rcnn_bbox_bias')

        bn1_cls_gamma = mx.symbol.Variable(name = 'bn1_cls_gamma')
        bn1_cls_beta = mx.symbol.Variable(name = 'bn1_cls_beta')
        bn1_cls_moving_mean = mx.symbol.Variable(name = 'bn1_cls_moving_mean')
        bn1_cls_moving_var = mx.symbol.Variable(name = 'bn1_cls_moving_var')
        bn2_cls_gamma = mx.symbol.Variable(name = 'bn2_cls_gamma')
        bn2_cls_beta = mx.symbol.Variable(name = 'bn2_cls_beta')
        bn2_cls_moving_mean = mx.symbol.Variable(name = 'bn2_cls_moving_mean')
        bn2_cls_moving_var = mx.symbol.Variable(name = 'bn2_cls_moving_var')

        bn1_box_gamma = mx.symbol.Variable(name = 'bn1_box_gamma')
        bn1_box_beta = mx.symbol.Variable(name = 'bn1_box_beta')
        bn1_box_moving_mean = mx.symbol.Variable(name = 'bn1_box_moving_mean')
        bn1_box_moving_var = mx.symbol.Variable(name = 'bn1_box_moving_var')

        bn_n1_gamma = mx.symbol.Variable(name = 'bn_n1_gamma')
        bn_n1_beta = mx.symbol.Variable(name = 'bn_n1_beta')
        bn_n1_moving_mean = mx.symbol.Variable(name = 'bn_n1_moving_mean')
        bn_n1_moving_var = mx.symbol.Variable(name = 'bn_n1_moving_var')

        bn_n2_gamma = mx.symbol.Variable(name = 'bn_n2_gamma')
        bn_n2_beta = mx.symbol.Variable(name = 'bn_n2_beta')
        bn_n2_moving_mean = mx.symbol.Variable(name = 'bn_n2_moving_mean')
        bn_n2_moving_var = mx.symbol.Variable(name = 'bn_n2_moving_var')

        bn_n3_gamma = mx.symbol.Variable(name = 'bn_n3_gamma')
        bn_n3_beta = mx.symbol.Variable(name = 'bn_n3_beta')
        bn_n3_moving_mean = mx.symbol.Variable(name = 'bn_n3_moving_mean')
        bn_n3_moving_var = mx.symbol.Variable(name = 'bn_n3_moving_var')
########################
        depth = 4
        # config alias for convenient
        num_classes = cfg.dataset.NUM_CLASSES
        num_reg_classes = (2 if cfg.CLASS_AGNOSTIC else num_classes)

        num_anchors_p3 = cfg.network.p3_NUM_ANCHORS
        num_anchors_p4 = cfg.network.p4_NUM_ANCHORS
        num_anchors_p5 = cfg.network.p5_NUM_ANCHORS
        num_anchors_p6 = cfg.network.p6_NUM_ANCHORS
        num_anchors = []
        num_anchors.append(num_anchors_p3)
        num_anchors.append(num_anchors_p4)
        num_anchors.append(num_anchors_p5)
        num_anchors.append(num_anchors_p6)

        fpn_feat_stride = []
        fpn_feat_stride.append(cfg.network.p3_RPN_FEAT_STRIDE)
        fpn_feat_stride.append(cfg.network.p4_RPN_FEAT_STRIDE)
        fpn_feat_stride.append(cfg.network.p5_RPN_FEAT_STRIDE)
        fpn_feat_stride.append(cfg.network.p6_RPN_FEAT_STRIDE)
        anchor_scales = []
        anchor_scales.append(cfg.network.p3_ANCHOR_SCALES)
        anchor_scales.append(cfg.network.p4_ANCHOR_SCALES)
        anchor_scales.append(cfg.network.p5_ANCHOR_SCALES)
        anchor_scales.append(cfg.network.p6_ANCHOR_SCALES)
        anchor_ratios = []
        anchor_ratios.append(cfg.network.p3_ANCHOR_RATIOS)
        anchor_ratios.append(cfg.network.p4_ANCHOR_RATIOS)
        anchor_ratios.append(cfg.network.p5_ANCHOR_RATIOS)
        anchor_ratios.append(cfg.network.p6_ANCHOR_RATIOS)


        # input init
        if is_train:
            data = mx.sym.Variable(name="data")
            im_info = mx.sym.Variable(name="im_info")
            gt_boxes = mx.sym.Variable(name="gt_boxes")

            fpn_label =[]
            fpn_label_p3 = mx.sym.Variable(name='label/p3')
            fpn_label_p4 = mx.sym.Variable(name='label/p4')
            fpn_label_p5 = mx.sym.Variable(name='label/p5')
            fpn_label_p6 = mx.sym.Variable(name='label/p6')
            fpn_label.append(fpn_label_p3)
            fpn_label.append(fpn_label_p4)
            fpn_label.append(fpn_label_p5)
            fpn_label.append(fpn_label_p6)

            fpn_bbox_target = []
            fpn_bbox_target_p3 = mx.sym.Variable(name='bbox_target/p3')           
            fpn_bbox_target_p4 = mx.sym.Variable(name='bbox_target/p4')
            fpn_bbox_target_p5 = mx.sym.Variable(name='bbox_target/p5')
            fpn_bbox_target_p6 = mx.sym.Variable(name='bbox_target/p6')
            fpn_bbox_target.append(fpn_bbox_target_p3)
            fpn_bbox_target.append(fpn_bbox_target_p4)
            fpn_bbox_target.append(fpn_bbox_target_p5)
            fpn_bbox_target.append(fpn_bbox_target_p6)

            fpn_bbox_weight = []
            fpn_bbox_weight_p3 = mx.sym.Variable(name='bbox_weight/p3')            
            fpn_bbox_weight_p4 = mx.sym.Variable(name='bbox_weight/p4')
            fpn_bbox_weight_p5 = mx.sym.Variable(name='bbox_weight/p5')
            fpn_bbox_weight_p6 = mx.sym.Variable(name='bbox_weight/p6')
            fpn_bbox_weight.append(fpn_bbox_weight_p3)
            fpn_bbox_weight.append(fpn_bbox_weight_p4)
            fpn_bbox_weight.append(fpn_bbox_weight_p5)
            fpn_bbox_weight.append(fpn_bbox_weight_p6)

        else:
            data = mx.sym.Variable(name="data")
            im_info = mx.sym.Variable(name="im_info")

  ###########top-down
        conv2 = self.get_resnet_v1_conv2(data)#4
        conv3 = self.get_resnet_v1_conv3(conv2)#8
        conv4 = self.get_resnet_v1_conv4(conv3)#16
        conv5 = self.get_resnet_v1_conv5(conv4)#16

        c3 = mx.sym.Convolution(data = conv2, kernel = (1,1), pad = (0,0),stride=(1, 1), num_filter = 256, name = 'c3' )
        c4 = mx.sym.Convolution(data = conv3, kernel = (1,1), pad = (0,0), stride=(1, 1),num_filter = 256, name = 'c4' )
    	c5 = mx.sym.Convolution(data = conv4, kernel = (1,1), pad = (0,0), stride=(1, 1),num_filter = 256,name = 'c5')    
        c6 = mx.sym.Convolution(data = conv5, kernel = (1,1), pad = (0,0), stride=(2, 2),num_filter = 256,name = 'c6')   
        
        ######p5->newp5
        p5 = c5 
        ####p4->newp4
        p5Upx2 = mx.sym.UpSampling(p5, scale =2 , sample_type = 'nearest')
        p5UpCrop = mx.sym.Crop(p5Upx2,c4)
        p4 = mx.sym.ElementWiseSum(name = 'p4', *[c4, p5UpCrop])
      	newp4 = mx.sym.Convolution(data = p4, kernel = (1,1), pad = (0,0), stride=(1, 1),num_filter = 256,name = 'newp4')         
        ####p3->newp3

        p4Upx2 = mx.sym.UpSampling(p4, scale =2 , sample_type = 'nearest')
        
        p4UpCrop = mx.sym.Crop(p4Upx2,c3)
        p3 = mx.sym.ElementWiseSum(name = 'p3', *[c3, p4UpCrop])
      	newp3 = mx.sym.Convolution(data = p3, kernel = (1,1), pad = (0,0), stride=(1, 1),num_filter = 256,name = 'newp3')    

        newp5 = p5
        newp6 = c6
        fpn_p = []
        fpn_p.append(newp3)
        fpn_p.append(newp4)
        fpn_p.append(newp5)
        fpn_p.append(newp6)
        

#####################fpn
        rois_list =[]
        fpn_cls_prob = []
        fpn_bbox_loss = []
        score_list = []

        for i in range(depth):
        
            fpn_conv = mx.sym.Convolution(
                data=fpn_p[i], kernel=(3, 3), pad=(1, 1), weight = fpn_conv_weight,bias =fpn_conv_bias, num_filter=256, name="fpn_conv_3x3"+str(i+3)) 
            fpn_conv = mx.sym.BatchNorm(data=fpn_conv, use_global_stats=False, eps=eps,gamma=bn1_box_gamma, beta =bn1_box_beta, moving_mean =bn1_box_moving_mean,
                moving_var= bn1_box_moving_var,momentum=bn_mom, name = 'bn1_box'+str(i+3))  
                    
            fpn_relu = mx.sym.Activation(data=fpn_conv, act_type="relu", name="fpn_relu"+str(i+3))
            fpn_cls_score = mx.sym.Convolution(
                data=fpn_relu, weight = fpn_cls_weight, bias = fpn_cls_bias,kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors[i], name="fpn_cls_score"+str(i+3))
          
            fpn_bbox_pred = mx.sym.Convolution(
                data=fpn_relu,weight = fpn_bbox_pred_weight,bias = fpn_bbox_pred_bias, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors[i], name="fpn_bbox_pred"+str(i+3))

            fpn_cls_score_reshape = mx.sym.Reshape(
                 data=fpn_cls_score, shape=(0, 2, -1, 0), name="fpn_cls_score_reshape"+str(i+3)) 
            if is_train:
                fpn_cls_prob_ = mx.sym.SoftmaxOutput(data=fpn_cls_score_reshape, label=fpn_label[i], multi_output=True,
                                                 normalization='valid', use_ignore=True, ignore_label=-1,
                                                 name="fpn_cls_prob"+str(i+3)) 
                fpn_cls_prob_ = mx.sym.Reshape(data = fpn_cls_prob_, shape=(1,2,-1)) 
                fpn_bbox_loss_ = fpn_bbox_weight[i] * mx.sym.smooth_l1(name='fpn_bbox_loss_'+str(i+3), scalar=3.0,
                                                                data=(fpn_bbox_pred - fpn_bbox_target[i]))
                fpn_bbox_loss_temp = mx.sym.MakeLoss(name='fpn_bbox_loss'+str(i+3), data=fpn_bbox_loss_,
                                            grad_scale=1.0 / cfg.TRAIN.RPN_BATCH_SIZE)
                fpn_bbox_loss_temp = mx.sym.Reshape(data = fpn_bbox_loss_temp,shape=(1,4 * num_anchors[i],-1))
                fpn_cls_act = mx.sym.SoftmaxActivation(
                     data=fpn_cls_score_reshape, mode="channel", name="fpn_cls_act"+str(i+3))
                fpn_cls_act_reshape = mx.sym.Reshape(
                     data=fpn_cls_act, shape=(0, 2 * num_anchors[i], -1, 0), name='fpn_cls_act_reshape'+str(i+3))
                rois_ ,score= mx.sym.Custom(
                    cls_prob=fpn_cls_act_reshape, bbox_pred=fpn_bbox_pred, im_info=im_info, name='rois_'+str(i+3),
                    op_type='proposal', feat_stride=fpn_feat_stride[i],
                    px_shape = fpn_p[i],
                    scales=tuple(anchor_scales[i]), ratios=tuple(anchor_ratios[i]),
                    rpn_pre_nms_top_n=cfg.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TRAIN.RPN_POST_NMS_TOP_N,
                    threshold=cfg.TRAIN.RPN_NMS_THRESH, rpn_min_size=cfg.TRAIN.RPN_MIN_SIZE)
                rois_list.append(rois_)
                fpn_cls_prob.append(fpn_cls_prob_)
                fpn_bbox_loss.append(fpn_bbox_loss_temp)
                score_list.append(score)


        rois_concat = mx.symbol.Concat(rois_list[0],rois_list[1],rois_list[2],rois_list[3],dim=0,name='rois_concat')

        score_concat = mx.symbol.Concat(score_list[0],score_list[1],score_list[2],score_list[3],dim=0,name='score_concat')

        rois_as = mx.symbol.Custom(rois = rois_concat,score =score_concat,rois_num = 1000,layer_num=depth,op_type='assign')

        rois_as_list = mx.symbol.SliceChannel(data=rois_as, axis=0, num_outputs=depth, name='slice_rois')



        gt_boxes_reshape = mx.sym.Reshape(data=gt_boxes, shape=(-1, 5), name='gt_boxes_reshape')  
        rois3, label3, bbox_target3, bbox_weight3 = mx.sym.Custom(rois=rois_as_list[0], gt_boxes=gt_boxes_reshape,
                                                                  op_type='proposal_target',
                                                                  num_classes=num_reg_classes,
                                                                  batch_images=cfg.TRAIN.BATCH_IMAGES,
                                                                  batch_rois=cfg.TRAIN.BATCH_ROIS,
                                                                  cfg=cPickle.dumps(cfg),
                                                                  fg_fraction=cfg.TRAIN.FG_FRACTION)
        rois4, label4, bbox_target4, bbox_weight4 = mx.sym.Custom(rois=rois_as_list[1], gt_boxes=gt_boxes_reshape,
                                                                  op_type='proposal_target',
                                                                  num_classes=num_reg_classes,
                                                                  batch_images=cfg.TRAIN.BATCH_IMAGES,
                                                                  batch_rois=cfg.TRAIN.BATCH_ROIS,
                                                                  cfg=cPickle.dumps(cfg),
                                                                  fg_fraction=cfg.TRAIN.FG_FRACTION)
        rois5, label5, bbox_target5, bbox_weight5 = mx.sym.Custom(rois=rois_as_list[2], gt_boxes=gt_boxes_reshape,
                                                                  op_type='proposal_target',
                                                                  num_classes=num_reg_classes,
                                                                  batch_images=cfg.TRAIN.BATCH_IMAGES,
                                                                  batch_rois=cfg.TRAIN.BATCH_ROIS,
                                                                  cfg=cPickle.dumps(cfg),
                                                                  fg_fraction=cfg.TRAIN.FG_FRACTION)
        rois6, label6, bbox_target6, bbox_weight6 = mx.sym.Custom(rois=rois_as_list[3], gt_boxes=gt_boxes_reshape,
                                                                  op_type='proposal_target',
                                                                  num_classes=num_reg_classes,
                                                                  batch_images=cfg.TRAIN.BATCH_IMAGES,
                                                                  batch_rois=cfg.TRAIN.BATCH_ROIS,
                                                                  cfg=cPickle.dumps(cfg),
                                                                  fg_fraction=cfg.TRAIN.FG_FRACTION)

        # gt_boxes_reshape = mx.sym.Reshape(data=gt_boxes, shape=(-1, 5), name='gt_boxes_reshape')  
        # rois_concat, label, bbox_target, bbox_weight = mx.sym.Custom(rois=rois_concat, gt_boxes=gt_boxes_reshape,
        #                                                           op_type='proposal_target',
        #                                                           num_classes=num_reg_classes,
        #                                                           batch_images=cfg.TRAIN.BATCH_IMAGES,
        #                                                           batch_rois=cfg.TRAIN.BATCH_ROIS,
        #                                                           cfg=cPickle.dumps(cfg),
        #                                                           fg_fraction=cfg.TRAIN.FG_FRACTION)
        rois = []
        labels = []
        bbox_targets = []
        bbox_weights = []
    #    rois3, label3, bbox_target3,bbox_weight3,rois4, label4, bbox_target4, bbox_weight4,rois5, label5, bbox_target5, bbox_weight5,rois6, label6, bbox_target6, bbox_weight6 = mx.symbol.Custom(rois = rois_concat ,label = label,bbox_target = bbox_target,bbox_weight = bbox_weight,op_type='assign_rois') 
       
       
        rois.append(rois3)
        rois.append(rois4)
        rois.append(rois5)
        rois.append(rois6)
        labels.append(label3)
        labels.append(label4)
        labels.append(label5)
        labels.append(label6)
        bbox_targets.append(bbox_target3)
        bbox_targets.append(bbox_target4)
        bbox_targets.append(bbox_target5)
        bbox_targets.append(bbox_target6)

        bbox_weights.append(bbox_weight3)
        bbox_weights.append(bbox_weight4)
        bbox_weights.append(bbox_weight5)
        bbox_weights.append(bbox_weight6)
        bbox_loss_list= []
        cls_prob_list = []
        rcnn_label_list = []
       
        for i in range(depth):
            roi_pool = mx.symbol.ROIPooling(name='roi_pool_'+str(i+3), data=fpn_p[i], rois=rois[i], pooled_size=(7, 7), spatial_scale=1.0/fpn_feat_stride[i])
            fc_new_1 = mx.sym.FullyConnected(name='fc_new_1'+str(i+3),weight=fc_new_1_weight,bias =fc_new_1_bias, data=roi_pool, num_hidden=1024)
            fc_new_1 = mx.sym.BatchNorm(data=fc_new_1, use_global_stats=False, eps=eps,gamma=bn1_cls_gamma, beta =bn1_cls_beta, moving_mean =bn1_cls_moving_mean,
                moving_var= bn1_cls_moving_var,momentum=bn_mom, name = 'bn1_cls'+str(i+3))
            fc_new_1_relu = mx.sym.Activation(data=fc_new_1, act_type='relu', name='fc_new_1_relu'+str(i+3))

            fc_new_2 = mx.sym.FullyConnected(name='fc_new_2'+str(i+3),weight=fc_new_2_weight,bias =fc_new_2_bias, data=fc_new_1_relu, num_hidden=1024)
            fc_new_2 = mx.sym.BatchNorm(data=fc_new_2, use_global_stats=False, eps=eps,gamma=bn2_cls_gamma, beta =bn2_cls_beta, moving_mean =bn2_cls_moving_mean,
                 moving_var= bn2_cls_moving_var,momentum=bn_mom, name = 'bn2_cls'+str(i+3))
            fc_new_2_relu = mx.sym.Activation(data=fc_new_2, act_type='relu', name='fc_new_2_relu'+str(i+3))
         # cls_score/bbox_pred
            cls_score = mx.sym.FullyConnected(name='cls_score'+str(i+3),weight=rcnn_cls_weight,bias =rcnn_cls_bias, data=fc_new_2_relu, num_hidden=num_classes)
            bbox_pred = mx.sym.FullyConnected(name='bbox_pred'+str(i+3),weight=rcnn_bbox_weight,bias =rcnn_bbox_bias,data=fc_new_2_relu, num_hidden=num_reg_classes * 4)
           
            cls_prob = mx.sym.SoftmaxOutput(name='cls_prob'+str(i+3), data=cls_score, label=labels[i], normalization='valid')
            bbox_loss_ = bbox_weights[i]* mx.sym.smooth_l1(name='bbox_loss_'+str(i+3), scalar=1.0,
                                                            data=(bbox_pred - bbox_targets[i]))
            bbox_loss = mx.sym.MakeLoss(name='bbox_loss'+str(i+3), data=bbox_loss_, grad_scale=1.0 / cfg.TRAIN.BATCH_ROIS)
            rcnn_label = labels[i]

            rcnn_label = mx.sym.Reshape(data=rcnn_label, shape=(cfg.TRAIN.BATCH_IMAGES, -1), name='label_reshape')
            cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TRAIN.BATCH_IMAGES, -1, num_classes),
                                      name='cls_prob_reshape')
            bbox_loss = mx.sym.Reshape(data=bbox_loss, shape=(cfg.TRAIN.BATCH_IMAGES, -1, 4 * num_reg_classes),
                                       name='bbox_loss_reshape')
            bbox_loss_list.append(bbox_loss)
            cls_prob_list.append(cls_prob)
            rcnn_label_list.append(mx.sym.BlockGrad(rcnn_label))

        fpn_cls_prob_concat = mx.symbol.Concat(fpn_cls_prob[0],fpn_cls_prob[1],fpn_cls_prob[2],fpn_cls_prob[3],dim=2,name='fpn_cls_prob_concat')
        fpn_bbox_loss_concat = mx.symbol.Concat(fpn_bbox_loss[0],fpn_bbox_loss[1],fpn_bbox_loss[2],fpn_bbox_loss[3],dim=2,name='fpn_bbox_loss_concat')
        bbox_loss_concat = mx.symbol.Concat(bbox_loss_list[0],bbox_loss_list[1],bbox_loss_list[2],bbox_loss_list[3],dim=0,name='bbox_loss_concat')
        cls_prob_concat = mx.symbol.Concat(cls_prob_list[0],cls_prob_list[1],cls_prob_list[2],cls_prob_list[3],dim=0,name='cls_prob_concat')
        rcnn_label_concat = mx.symbol.Concat(rcnn_label_list[0],rcnn_label_list[1],rcnn_label_list[2],rcnn_label_list[3],dim=0,name='rcnn_label_concat')

        group = mx.sym.Group([fpn_cls_prob_concat,fpn_bbox_loss_concat,cls_prob_concat, bbox_loss_concat, rcnn_label_concat])
        self.sym = group
        return group





    def init_weight(self, cfg, arg_params, aux_params):

###share
        pi = 0.01
        arg_params['fpn_conv_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fpn_conv_weight'])
        arg_params['fpn_conv_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fpn_conv_bias'])
        arg_params['fpn_cls_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fpn_cls_weight'])
        arg_params['fpn_cls_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fpn_cls_bias'])
        arg_params['fpn_bbox_pred_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fpn_bbox_pred_weight'])
        arg_params['fpn_bbox_pred_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fpn_bbox_pred_bias'])

        # arg_params['fpn_conv_3x33_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fpn_conv_3x33_weight'])
        # arg_params['fpn_conv_3x33_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fpn_conv_3x33_bias'])
        # arg_params['fpn_cls_score3_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fpn_cls_score3_weight'])
        # arg_params['fpn_cls_score3_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fpn_cls_score3_bias'])
        # arg_params['fpn_bbox_pred3_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fpn_bbox_pred3_weight'])
        # arg_params['fpn_bbox_pred3_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fpn_bbox_pred3_bias'])
     
        # arg_params['fpn_conv_3x34_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fpn_conv_3x34_weight'])
        # arg_params['fpn_conv_3x34_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fpn_conv_3x34_bias'])
        # arg_params['fpn_cls_score4_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fpn_cls_score4_weight'])
        # arg_params['fpn_cls_score4_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fpn_cls_score4_bias'])
        # arg_params['fpn_bbox_pred4_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fpn_bbox_pred4_weight'])
        # arg_params['fpn_bbox_pred4_bias'] = mx.nd.ones(shape=self.arg_shape_dict['fpn_bbox_pred4_bias'])*(-np.log((1-pi)/pi))
 
        # arg_params['fpn_conv_3x35_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fpn_conv_3x35_weight'])
        # arg_params['fpn_conv_3x35_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fpn_conv_3x35_bias'])
        # arg_params['fpn_cls_score5_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fpn_cls_score5_weight'])
        # arg_params['fpn_cls_score5_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fpn_cls_score5_bias'])
        # arg_params['fpn_bbox_pred5_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fpn_bbox_pred5_weight'])
        # arg_params['fpn_bbox_pred5_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fpn_bbox_pred5_bias'])
## donot share
        # arg_params['fc_new_1/p3_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fc_new_1/p3_weight'])
        # arg_params['fc_new_1/p3_bias'] = mx.random.normal(0, 0.01,shape=self.arg_shape_dict['fc_new_1/p3_bias'])
        # arg_params['fc_new_2/p3_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fc_new_2/p3_weight'])
        # arg_params['fc_new_2/p3_bias'] = mx.random.normal(0, 0.01,shape=self.arg_shape_dict['fc_new_2/p3_bias'])
        # arg_params['cls_score/p3_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['cls_score/p3_weight'])
        # arg_params['cls_score/p3_bias'] = mx.random.normal(0, 0.01,shape=self.arg_shape_dict['cls_score/p3_bias'])
        # arg_params['bbox_pred/p3_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['bbox_pred/p3_weight'])
        # arg_params['bbox_pred/p3_bias'] = mx.random.normal(0, 0.01,shape=self.arg_shape_dict['bbox_pred/p3_bias'])


        # arg_params['fc_new_1/p4_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fc_new_1/p4_weight'])
        # arg_params['fc_new_1/p4_bias'] = mx.random.normal(0, 0.01,shape=self.arg_shape_dict['fc_new_1/p4_bias'])
        # arg_params['fc_new_2/p4_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fc_new_2/p4_weight'])
        # arg_params['fc_new_2/p4_bias'] = mx.random.normal(0, 0.01,shape=self.arg_shape_dict['fc_new_2/p4_bias'])
        # arg_params['cls_score/p4_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['cls_score/p4_weight'])
        # arg_params['cls_score/p4_bias'] = mx.random.normal(0, 0.01,shape=self.arg_shape_dict['cls_score/p4_bias'])
        # arg_params['bbox_pred/p4_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['bbox_pred/p4_weight'])
        # arg_params['bbox_pred/p4_bias'] = mx.random.normal(0, 0.01,shape=self.arg_shape_dict['bbox_pred/p4_bias'])


        # arg_params['fc_new_1/p5_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fc_new_1/p5_weight'])
        # arg_params['fc_new_1/p5_bias'] = mx.random.normal(0, 0.01,shape=self.arg_shape_dict['fc_new_1/p5_bias'])
        # arg_params['fc_new_2/p5_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fc_new_2/p5_weight'])
        # arg_params['fc_new_2/p5_bias'] = mx.random.normal(0, 0.01,shape=self.arg_shape_dict['fc_new_2/p5_bias'])
        # arg_params['cls_score/p5_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['cls_score/p5_weight'])
        # arg_params['cls_score/p5_bias'] = mx.random.normal(0, 0.01,shape=self.arg_shape_dict['cls_score/p5_bias'])
        # arg_params['bbox_pred/p5_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['bbox_pred/p5_weight'])
        # arg_params['bbox_pred/p5_bias'] = mx.random.normal(0, 0.01,shape=self.arg_shape_dict['bbox_pred/p5_bias'])





        arg_params['c3_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['c3_weight'])
        arg_params['c3_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['c3_bias'])

        arg_params['c4_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['c4_weight'])
        arg_params['c4_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['c4_bias'])
      
        arg_params['c5_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['c5_weight'])
        arg_params['c5_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['c5_bias'])
        arg_params['c6_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['c6_weight'])
        arg_params['c6_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['c6_bias'])

     
        arg_params['newp3_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['newp3_weight'])
        arg_params['newp3_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['newp3_bias'])
        
        arg_params['newp4_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['newp4_weight'])
        arg_params['newp4_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['newp4_bias'])

        
        arg_params['res5a_branch2b_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['res5a_branch2b_weight'])

        
        arg_params['res5b_branch2b_weight'] =   mx.random.normal(0, 0.01, shape=self.arg_shape_dict['res5b_branch2b_weight'])
    
        arg_params['res5c_branch2b_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['res5c_branch2b_weight'])

        # arg_params['offset/p3_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['offset/p3_weight'])
        # arg_params['offset/p3_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['offset/p3_bias'])       
        
        # arg_params['offset/p4_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['offset/p4_weight'])
        # arg_params['offset/p4_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['offset/p4_bias'])

   
        # arg_params['offset/p5_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['offset/p5_weight'])
        # arg_params['offset/p5_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['offset/p5_bias'])


        # arg_params['res5a_branch2b_offset_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['res5a_branch2b_offset_weight'])
        # arg_params['res5a_branch2b_offset_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['res5a_branch2b_offset_bias'])
        # arg_params['res5b_branch2b_offset_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['res5b_branch2b_offset_weight'])
        # arg_params['res5b_branch2b_offset_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['res5b_branch2b_offset_bias'])
        # arg_params['res5c_branch2b_offset_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['res5c_branch2b_offset_weight'])
        # arg_params['res5c_branch2b_offset_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['res5c_branch2b_offset_bias'])

     
# SHARE 
        a = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fc_new_1_weight'])
        arg_params['fc_new_1_weight'] = a
        arg_params['fc_new_1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fc_new_1_bias'])

        b = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fc_new_2_weight'])
        arg_params['fc_new_2_weight'] =b
        arg_params['fc_new_2_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fc_new_2_bias'])
        c = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rcnn_cls_weight'])
        arg_params['rcnn_cls_weight'] = c
        arg_params['rcnn_cls_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rcnn_cls_bias'])
        d = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rcnn_bbox_weight'])
        arg_params['rcnn_bbox_weight'] =  d
        arg_params['rcnn_bbox_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rcnn_bbox_bias'])


        arg_params['bn1_cls_gamma'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['bn1_cls_gamma'])
        arg_params['bn1_cls_beta'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['bn1_cls_beta'])
        aux_params['bn1_cls_moving_mean'] =  mx.random.normal(0, 0.01,shape=self.aux_shape_dict['bn1_cls_moving_mean'])
        aux_params['bn1_cls_moving_var'] =  mx.random.normal(0, 0.01,shape=self.aux_shape_dict['bn1_cls_moving_var'])

        arg_params['bn2_cls_gamma'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['bn2_cls_gamma'])
        arg_params['bn2_cls_beta'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['bn2_cls_beta'])
        aux_params['bn2_cls_moving_mean'] =  mx.random.normal(0, 0.01,shape=self.aux_shape_dict['bn2_cls_moving_mean'])
        aux_params['bn2_cls_moving_var'] =  mx.random.normal(0, 0.01,shape=self.aux_shape_dict['bn2_cls_moving_var'])


        arg_params['bn1_box_gamma'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['bn1_box_gamma'])
        arg_params['bn1_box_beta'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['bn1_box_beta'])
        aux_params['bn1_box_moving_mean'] =  mx.random.normal(0, 0.01,shape=self.aux_shape_dict['bn1_box_moving_mean'])
        aux_params['bn1_box_moving_var'] =  mx.random.normal(0, 0.01,shape=self.aux_shape_dict['bn1_box_moving_var'])

        # arg_params['bn_n1_gamma'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['bn_n1_gamma'])
        # arg_params['bn_n1_beta'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['bn_n1_beta'])
        # aux_params['bn_n1_moving_mean'] =  mx.random.normal(0, 0.01,shape=self.aux_shape_dict['bn_n1_moving_mean'])
        # aux_params['bn_n1_moving_var'] =  mx.random.normal(0, 0.01,shape=self.aux_shape_dict['bn_n1_moving_var'])

        # arg_params['bn_n2_gamma'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['bn_n2_gamma'])
        # arg_params['bn_n2_beta'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['bn_n2_beta'])
        # aux_params['bn_n2_moving_mean'] =  mx.random.normal(0, 0.01,shape=self.aux_shape_dict['bn_n2_moving_mean'])
        # aux_params['bn_n2_moving_var'] =  mx.random.normal(0, 0.01,shape=self.aux_shape_dict['bn_n2_moving_var'])

        # arg_params['bn_n3_gamma'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['bn_n3_gamma'])
        # arg_params['bn_n3_beta'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['bn_n3_beta'])
        # aux_params['bn_n3_moving_mean'] =  mx.random.normal(0, 0.01,shape=self.aux_shape_dict['bn_n3_moving_mean'])
        # aux_params['bn_n3_moving_var'] =  mx.random.normal(0, 0.01,shape=self.aux_shape_dict['bn_n3_moving_var'])
