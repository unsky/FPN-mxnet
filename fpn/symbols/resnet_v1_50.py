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
from operator_py.fpn_roi_pooling import *
from operator_py.roi_concat import *
eps = 2e-5
bn_mom  = 0.9

class resnet_v1_50(Symbol):
    def __init__(self):
        """
        Use __init__ to define parameter network needs
        """
        self.eps = 2e-5
        self.USE_GLOBAL_STATS = True
        self.workspace = 512
        self.res_deps = {'18': (2, 2, 2, 2), '34': (3, 4, 6, 3), '50': (3, 4, 6, 3), '101': (3, 4, 23, 3), '152': (3, 8, 36, 3), '200': (3, 24, 36, 3)}
    def residual_unit(self,data, num_filter, stride, dim_match, name,use_global_stats=True, bn_mom=0.9, bottle_neck=True, dilate=(1, 1)):
        workspace =self.workspace
        if bottle_neck:
            bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, momentum=bn_mom, name=name + '_bn1')
            act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
            conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter * 0.25), kernel=(1, 1), stride=(1, 1), pad=(0, 0), dilate=dilate, 
                                    no_bias=True, workspace=workspace, name=name + '_conv1')
            bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, momentum=bn_mom, name=name + '_bn2')
            act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
            conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter * 0.25), kernel=(3, 3), stride=stride, pad=(1, 1), dilate=dilate, 
                                    no_bias=True, workspace=workspace, name=name + '_conv2')
            bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, momentum=bn_mom, name=name + '_bn3')
            act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')
            conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), no_bias=True, dilate=dilate, 
                                    workspace=workspace, name=name + '_conv3')
            if dim_match:
                shortcut = data
            else:
                shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1, 1), stride=stride, no_bias=True, dilate=dilate, 
                                            workspace=workspace, name=name + '_sc')
            sum = mx.sym.ElementWiseSum(*[conv3, shortcut], name=name + '_plus')
            return sum
        else:
            bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, use_global_stats=use_global_stats, momentum=bn_mom, name=name + '_bn1')
            act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
            conv1 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3,3), stride=stride, pad=(1,1), dilate=dilate, 
                                        no_bias=True, workspace=workspace, name=name + '_conv1')
            bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, use_global_stats=use_global_stats, momentum=bn_mom, name=name + '_bn2')
            act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
            conv2 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1), dilate=dilate, 
                                        no_bias=True, workspace=workspace, name=name + '_conv2')
            if dim_match:
                shortcut = data
            else:
                shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True, dilate=dilate, 
                                                workspace=workspace, name=name+'_sc')
                
            sum = mx.sym.ElementWiseSum(*[conv2, shortcut], name=name + '_plus')
            return sum

    def get_fpn_resnet_conv(self,data, depth): #add bn to fpn layer:2017-08-01
        units = self.res_deps[str(depth)]
        filter_list = [256, 512, 1024, 2048, 256] if depth >= 50 else [64, 128, 256, 512, 256]

        bottle_neck = True if depth >= 50 else False
        USE_GLOBAL_STATS =self.USE_GLOBAL_STATS
        workspace = self.workspace
        # res1
        data_bn = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=eps, use_global_stats=USE_GLOBAL_STATS, name='bn_data')
        conv0 = mx.sym.Convolution(data=data_bn, num_filter=64, kernel=(7, 7), stride=(2, 2), pad=(3, 3),
                                no_bias=True, name="conv0", workspace=workspace)
        bn0 = mx.sym.BatchNorm(data=conv0, fix_gamma=False, eps=eps, use_global_stats=USE_GLOBAL_STATS, name='bn0')
        relu0 = mx.sym.Activation(data=bn0, act_type='relu', name='relu0')
        pool0 = mx.symbol.Pooling(data=relu0, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', name='pool0')

        # res2
        conv1 = self.residual_unit(data=pool0, num_filter=filter_list[0], stride=(1, 1), dim_match=False, name='stage1_unit1',
                            bottle_neck=bottle_neck)
        for i in range(2, units[0] + 1):
            conv1 = self.residual_unit(data=conv1, num_filter=filter_list[0], stride=(1, 1), dim_match=True, name='stage1_unit%s' % i,
                                bottle_neck=bottle_neck)
        #stride = 4

        # res3
        conv2 = self.residual_unit(data=conv1, num_filter=filter_list[1], stride=(2, 2), dim_match=False, name='stage2_unit1',
                            bottle_neck=bottle_neck)
        for i in range(2, units[1] + 1):
            conv2 = self.residual_unit(data=conv2, num_filter=filter_list[1], stride=(1, 1), dim_match=True, name='stage2_unit%s' % i,
                                bottle_neck=bottle_neck)
        # stride = 8
        # res4
        conv3 = self.residual_unit(data=conv2, num_filter=filter_list[2], stride=(2, 2), dim_match=False, name='stage3_unit1',
                            bottle_neck=bottle_neck)
        for i in range(2, units[2] + 1):
            conv3 = self.residual_unit(data=conv3, num_filter=filter_list[2], stride=(1, 1), dim_match=True, name='stage3_unit%s' % i,
                                bottle_neck=bottle_neck)
        #stride = 16
        # res5
        conv4 = self.residual_unit(data=conv3, num_filter=filter_list[3], stride=(2, 2), dim_match=False, name='stage4_unit1',
                            bottle_neck=bottle_neck)
        for i in range(2, units[3] + 1):
            conv4 = self.residual_unit(data=conv4, num_filter=filter_list[3], stride=(1, 1), dim_match=True, name='stage4_unit%s' % i,
                                bottle_neck=bottle_neck)
        # bn4 = mx.sym.BatchNorm(data=conv4, fix_gamma=False, eps=eps, use_global_stats=USE_GLOBAL_STATS, momentum=0.9, name='stage5_bn1')
        # act4 = mx.sym.Activation(data=bn4, act_type='relu', name='stage5_relu1')
        #stride = 32
        up_conv6_out = mx.symbol.Convolution(data=conv4, kernel=(3, 3), pad=(1,1), stride=(2,2), num_filter=filter_list[4], name='stage6_conv_3*3')
        #stride = 64
        # de-res5
        up_conv5_out = mx.symbol.Convolution(data=conv4, kernel=(1, 1), pad=(0, 0), num_filter=filter_list[4], name='stage5_conv_1x1')

        up_conv4 = mx.symbol.UpSampling(up_conv5_out, scale=2, sample_type="nearest")
        #bn_up_conv4 = mx.sym.BatchNorm(data=up_conv4, fix_gamma=False, eps=eps, use_global_stats=USE_GLOBAL_STATS, momentum=0.9, name='up_conv4_bn1')
        conv3_1 = mx.symbol.Convolution(
            data=conv3, kernel=(1, 1), pad=(0, 0), num_filter=filter_list[4], name='up_stage4_conv_1x1')
        #bn_conv3_1 = mx.sym.BatchNorm(data=conv3_1, fix_gamma=False, eps=eps, use_global_stats=USE_GLOBAL_STATS, momentum=0.9, name='conv3_1_bn1')
        up_conv4_ = up_conv4 + conv3_1
        up_conv4_out = mx.symbol.Convolution(
            data=up_conv4_, kernel=(3, 3), pad=(1, 1), num_filter=filter_list[4], name='up_stage4_conv_3x3')
        
        # de-res4
        up_conv3 = mx.symbol.UpSampling(up_conv4_out, scale=2, sample_type="nearest")
        #bn_up_conv3 = mx.sym.BatchNorm(data=up_conv3, fix_gamma=False, eps=eps, use_global_stats=USE_GLOBAL_STATS, momentum=0.9, name='up_conv3_bn1')
        conv2_1 = mx.symbol.Convolution(
            data=conv2, kernel=(1, 1), pad=(0, 0), num_filter=filter_list[4], name='up_stage3_conv_1x1')
        #bn_conv2_1 = mx.sym.BatchNorm(data=conv2_1, fix_gamma=False, eps=eps, use_global_stats=USE_GLOBAL_STATS, momentum=0.9, name='conv2_1_bn1')
        up_conv3_ = up_conv3 + conv2_1
        up_conv3_out = mx.symbol.Convolution(
            data=up_conv3_, kernel=(3, 3), pad=(1, 1), num_filter=filter_list[4], name='up_stage3_conv_3x3')
        
        # de-res3
        up_conv2 = mx.symbol.UpSampling(up_conv3_out, scale=2, sample_type="nearest") 
        #bn_up_conv2 = mx.sym.BatchNorm(data=up_conv2, fix_gamma=False, eps=eps, use_global_stats=USE_GLOBAL_STATS, momentum=0.9, name='up_conv2_bn1')
        conv1_1 = mx.symbol.Convolution(
            data=conv1, kernel=(1, 1), pad=(0, 0), num_filter=filter_list[4], name='up_stage2_conv_1x1')
        #bn_conv1_1 = mx.sym.BatchNorm(data=conv1_1, fix_gamma=False, eps=eps, use_global_stats=USE_GLOBAL_STATS, momentum=0.9, name='conv1_1_bn1')
        up_conv2_ = up_conv2 + conv1_1
        up_conv2_out = mx.symbol.Convolution(
            data=up_conv2_, kernel=(3, 3), pad=(1, 1), num_filter=filter_list[4], name='up_stage2_conv_3x3')
        
        output = []
        output.append(up_conv2_out)#stride:4
        output.append(up_conv3_out)#stride:8
        output.append(up_conv4_out)#stride:16
        output.append(up_conv5_out)#stride:32
        output.append(up_conv6_out)#stride:64

        return output

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

 

########################
        depth = 5
        rcnn_depth =4
        # config alias for convenient
        num_classes = cfg.dataset.NUM_CLASSES
        num_reg_classes = (2 if cfg.CLASS_AGNOSTIC else num_classes)
        num_anchors_p2 = cfg.network.p2_NUM_ANCHORS
        num_anchors_p3 = cfg.network.p3_NUM_ANCHORS
        num_anchors_p4 = cfg.network.p4_NUM_ANCHORS
        num_anchors_p5 = cfg.network.p5_NUM_ANCHORS
        num_anchors_p6 = cfg.network.p6_NUM_ANCHORS
        num_anchors = []
        num_anchors.append(num_anchors_p2)
        num_anchors.append(num_anchors_p3)
        num_anchors.append(num_anchors_p4)
        num_anchors.append(num_anchors_p5)
        num_anchors.append(num_anchors_p6)

        fpn_feat_stride = []
        fpn_feat_stride.append(cfg.network.p2_RPN_FEAT_STRIDE)
        fpn_feat_stride.append(cfg.network.p3_RPN_FEAT_STRIDE)
        fpn_feat_stride.append(cfg.network.p4_RPN_FEAT_STRIDE)
        fpn_feat_stride.append(cfg.network.p5_RPN_FEAT_STRIDE)
        fpn_feat_stride.append(cfg.network.p6_RPN_FEAT_STRIDE)
        anchor_scales = []
        anchor_scales.append(cfg.network.p2_ANCHOR_SCALES)
        anchor_scales.append(cfg.network.p3_ANCHOR_SCALES)
        anchor_scales.append(cfg.network.p4_ANCHOR_SCALES)
        anchor_scales.append(cfg.network.p5_ANCHOR_SCALES)
        anchor_scales.append(cfg.network.p6_ANCHOR_SCALES)
        anchor_ratios = []
        anchor_ratios.append(cfg.network.p2_ANCHOR_RATIOS)
        anchor_ratios.append(cfg.network.p3_ANCHOR_RATIOS)
        anchor_ratios.append(cfg.network.p4_ANCHOR_RATIOS)
        anchor_ratios.append(cfg.network.p5_ANCHOR_RATIOS)
        anchor_ratios.append(cfg.network.p6_ANCHOR_RATIOS)


        # input init
        if is_train:
            data = mx.sym.Variable(name="data")
            im_info = mx.sym.Variable(name="im_info")
            gt_boxes = mx.sym.Variable(name="gt_boxes")
            
            fpn_bbox_weight = []
            fpn_bbox_weight_p2 = mx.sym.Variable(name='bbox_weight/p2')   
            fpn_bbox_weight_p3 = mx.sym.Variable(name='bbox_weight/p3')            
            fpn_bbox_weight_p4 = mx.sym.Variable(name='bbox_weight/p4')
            fpn_bbox_weight_p5 = mx.sym.Variable(name='bbox_weight/p5')
            fpn_bbox_weight_p6 = mx.sym.Variable(name='bbox_weight/p6')
            
            fpn_bbox_weight.append(fpn_bbox_weight_p2)
            fpn_bbox_weight.append(fpn_bbox_weight_p3)
            fpn_bbox_weight.append(fpn_bbox_weight_p4)
            fpn_bbox_weight.append(fpn_bbox_weight_p5)
            fpn_bbox_weight.append(fpn_bbox_weight_p6)
            fpn_label =[]
            fpn_label_p2 = mx.sym.Variable(name='label/p2')
            fpn_label_p3 = mx.sym.Variable(name='label/p3')
            fpn_label_p4 = mx.sym.Variable(name='label/p4')
            fpn_label_p5 = mx.sym.Variable(name='label/p5')
            fpn_label_p6 = mx.sym.Variable(name='label/p6')
            fpn_label.append(fpn_label_p2)
            fpn_label.append(fpn_label_p3)
            fpn_label.append(fpn_label_p4)
            fpn_label.append(fpn_label_p5)
            fpn_label.append(fpn_label_p6)

            fpn_bbox_target = []
            fpn_bbox_target_p2 = mx.sym.Variable(name='bbox_target/p2')   
            fpn_bbox_target_p3 = mx.sym.Variable(name='bbox_target/p3')           
            fpn_bbox_target_p4 = mx.sym.Variable(name='bbox_target/p4')
            fpn_bbox_target_p5 = mx.sym.Variable(name='bbox_target/p5')
            fpn_bbox_target_p6 = mx.sym.Variable(name='bbox_target/p6')
            
            fpn_bbox_target.append(fpn_bbox_target_p2)
            fpn_bbox_target.append(fpn_bbox_target_p3)
            fpn_bbox_target.append(fpn_bbox_target_p4)
            fpn_bbox_target.append(fpn_bbox_target_p5)
            fpn_bbox_target.append(fpn_bbox_target_p6)
        else:
            data = mx.sym.Variable(name="data")
            im_info = mx.sym.Variable(name="im_info")

  ###########top-down
        fpn_p = self.get_fpn_resnet_conv(data,'50')
     
        
        rois_list =[]
        label_list = []
        bbox_weight_list = []
        bbox_target_list = []
        fpn_cls_prob = []
        fpn_bbox_loss = []
        score_list = []
        fpn_cls_act_reshapes_list = []
        fpn_bbox_pred_list = []
        for i in range(depth):
            fpn_conv = mx.sym.Convolution(
                data=fpn_p[i], kernel=(3, 3), pad=(1, 1), weight = fpn_conv_weight,bias =fpn_conv_bias, num_filter=256, name="fpn_conv_3x3"+str(i+2))     
            fpn_relu = mx.sym.Activation(data=fpn_conv, act_type="relu", name="fpn_relu"+str(i+2))
            fpn_cls_score = mx.sym.Convolution(
                data=fpn_relu, weight = fpn_cls_weight, bias = fpn_cls_bias,kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors[i], name="fpn_cls_score"+str(i+2))
          
            fpn_bbox_pred = mx.sym.Convolution(
                data=fpn_relu,weight = fpn_bbox_pred_weight,bias = fpn_bbox_pred_bias, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors[i], name="fpn_bbox_pred"+str(i+2))

            fpn_cls_score_reshape = mx.sym.Reshape(
                 data=fpn_cls_score, shape=(0, 2, -1, 0), name="fpn_cls_score_reshape"+str(i+2)) 
            if is_train:
                fpn_cls_prob_ = mx.sym.SoftmaxOutput(data=fpn_cls_score_reshape, label=fpn_label[i], multi_output=True,
                                                 normalization='valid', use_ignore=True, ignore_label=-1,
                                                 name="fpn_cls_prob"+str(i+2))                
                fpn_cls_prob_ = mx.sym.Reshape(data = fpn_cls_prob_, shape=(1,2,-1)) 
              
                fpn_bbox_loss_ = fpn_bbox_weight[i] * mx.sym.smooth_l1(name='fpn_bbox_loss_aaa'+str(i+2), scalar=3.0,
                                                                data=(fpn_bbox_pred - fpn_bbox_target[i]))
                                                                
                fpn_bbox_loss_temp = mx.sym.MakeLoss(name='fpn_bbox_loss'+str(i+2), data=fpn_bbox_loss_,
                                            grad_scale=1.0 / cfg.TRAIN.RPN_BATCH_SIZE)
                fpn_bbox_loss_temp = mx.sym.Reshape(data = fpn_bbox_loss_temp,shape=(1,4 * num_anchors[i],-1))
                fpn_cls_act = mx.sym.SoftmaxActivation(
                     data=fpn_cls_score_reshape, mode="channel", name="fpn_cls_act"+str(i+2))
                fpn_cls_act_reshape = mx.sym.Reshape(
                     data=fpn_cls_act, shape=(0, 2 * num_anchors[i], -1, 0), name='fpn_cls_act_reshape'+str(i+2))

                fpn_cls_act_reshapes_list.append(mx.symbol.flatten(data=fpn_cls_act_reshape, name='flatten_rpn_cls_act_%d'%(i+2)))#shape:[1,2*3*h*w]
                fpn_bbox_pred_list.append(mx.symbol.flatten(data=fpn_bbox_pred, name='flatten_rpn_bbox_pred_%d'%(i+2)))#shape:[1,4*3*h*w]
                fpn_cls_prob.append(fpn_cls_prob_)
                fpn_bbox_loss.append(fpn_bbox_loss_temp)
            else:
                fpn_cls_prob = mx.sym.SoftmaxActivation(data=fpn_cls_score_reshape, mode="channel", name="fpn_cls_prob"+str(i+2))
                fpn_cls_prob_reshape = mx.sym.Reshape(
                  data=fpn_cls_prob, shape=(0, 2 * num_anchors[i], -1, 0), name='fpn_cls_prob_reshape')
                fpn_cls_act_reshapes_list.append(mx.symbol.flatten(data=fpn_cls_prob_reshape, name='flatten_fpn_cls_act_%d'%(i+2)))
                fpn_bbox_pred_list.append(mx.symbol.flatten(data=fpn_bbox_pred, name='flatten_fpn_bbox_pred_%d'%(i+2)))
   
        concat_flat_fpn_cls_act = mx.symbol.Concat(fpn_cls_act_reshapes_list[0],fpn_cls_act_reshapes_list[1],fpn_cls_act_reshapes_list[2],fpn_cls_act_reshapes_list[3],fpn_cls_act_reshapes_list[4],dim=1,name='concat_fpn_cls_act')
        concat_flat_fpn_bbox_pred = mx.symbol.Concat(fpn_bbox_pred_list[0],fpn_bbox_pred_list[1],fpn_bbox_pred_list[2],fpn_bbox_pred_list[3],fpn_bbox_pred_list[4],dim=1,name='concat_pn_bbox_pred')

        if is_train:
            rois = mx.symbol.Custom(
                cls_prob=concat_flat_fpn_cls_act, bbox_pred=concat_flat_fpn_bbox_pred, im_info=im_info, name='rois_concat',
                op_type='roi_concat', feat_stride='4,8,16,32,64', p2 =fpn_p[0],p3=fpn_p[1],p4=fpn_p[2],p5=fpn_p[3],p6=fpn_p[4],
                scales=tuple(anchor_scales[0]), ratios=tuple(anchor_ratios[0]),
                rpn_pre_nms_top_n=cfg.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TRAIN.RPN_POST_NMS_TOP_N,
                threshold=cfg.TRAIN.RPN_NMS_THRESH, rpn_min_size='4,8,16,32,64')#[2000,4]
    
            gt_boxes_reshape = mx.sym.Reshape(data=gt_boxes, shape=(-1, 5), name='gt_boxes_reshape')  
            rois_all, label_all, bbox_target_all, bbox_weight_all = mx.sym.Custom(rois=rois ,gt_boxes=gt_boxes_reshape,data=data,
                                                                op_type='proposal_target',
                                                                num_classes=num_reg_classes,
                                                                batch_images=cfg.TRAIN.BATCH_IMAGES,
                                                                batch_rois=cfg.TRAIN.BATCH_ROIS,
                                                                cfg=cPickle.dumps(cfg),
                                                                fg_fraction=cfg.TRAIN.FG_FRACTION)  
 
            rois = mx.symbol.SliceChannel(data=rois_all, axis=0, num_outputs=rcnn_depth)
            label = mx.symbol.SliceChannel(data=label_all, axis=0, num_outputs = rcnn_depth)
            bbox_target = mx.symbol.SliceChannel(data=bbox_target_all, axis=0, num_outputs = rcnn_depth)
            bbox_weight = mx.symbol.SliceChannel(data=bbox_weight_all, axis=0, num_outputs = rcnn_depth)   
        else:
            rois = mx.symbol.Custom(
                cls_prob=concat_flat_fpn_cls_act, bbox_pred=concat_flat_fpn_bbox_pred, im_info=im_info, name='rois_concat',
                op_type='roi_concat', feat_stride='4,8,16,32,64', p2 =fpn_p[0],p3=fpn_p[1],p4=fpn_p[2],p5=fpn_p[3],p6=fpn_p[4],
                scales=tuple(anchor_scales[0]), ratios=tuple(anchor_ratios[0]),
                rpn_pre_nms_top_n=cfg.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TEST.RPN_POST_NMS_TOP_N,
                threshold=cfg.TEST.RPN_NMS_THRESH, rpn_min_size='4,8,16,32,64')#[2000,4]
####################rcnn share###########
        bbox_loss_list= []
        cls_prob_list = []
        rcnn_label_list = []
        bbox_pred_list = []
        roi_pool_list = []
        ro_list =[]
        l_list =[]
        weight_list =[]
        target_list =[]
        c_list = []
        b_list = []
        if is_train:
            for i in range(4):
                rois_pool = mx.symbol.ROIPooling(name='roi_pool_'+str(i), data=fpn_p[i], rois=rois[i], pooled_size=(14, 14), spatial_scale=1.0/fpn_feat_stride[i])
                fc_new_1 = mx.sym.FullyConnected(name='fc_new_1'+str(i),weight=fc_new_1_weight,bias =fc_new_1_bias, data=rois_pool, num_hidden=1024)

                fc_new_1_relu = mx.sym.Activation(data=fc_new_1, act_type='relu', name='fc_new_1_relu'+str(i))
                fc_new_1_relu = mx.symbol.Dropout(data=fc_new_1_relu, p=0.5, name="drop6_%d" %(i+2))
                fc_new_2 = mx.sym.FullyConnected(name='fc_new_2'+str(i),weight=fc_new_2_weight,bias =fc_new_2_bias, data=fc_new_1_relu, num_hidden=1024)
                fc_new_2_relu = mx.sym.Activation(data=fc_new_2, act_type='relu', name='fc_new_2_relu'+str(i))
            # cls_score/bbox_pred
                fc_new_2_relu = mx.symbol.Dropout(data=fc_new_2_relu, p=0.5, name="drop7_%d" %(i+2))
                cls_score = mx.sym.FullyConnected(name='cls_score'+str(i),weight=rcnn_cls_weight,bias =rcnn_cls_bias, data=fc_new_2_relu, num_hidden=num_classes)
                bbox_pred = mx.sym.FullyConnected(name='bbox_pred'+str(i),weight=rcnn_bbox_weight,bias =rcnn_bbox_bias,data=fc_new_2_relu, num_hidden=num_reg_classes * 4)
                cls_prob = mx.sym.SoftmaxOutput(name='cls_prob'+str(i), data=cls_score, label=label[i], normalization='valid',use_ignore=True, ignore_label=-1)
                bbox_loss_ = bbox_weight[i]* mx.sym.smooth_l1(name='bbox_loss_'+str(i), scalar=1.0,
                                                            data=(bbox_pred - bbox_target[i]))
                bbox_loss = mx.sym.MakeLoss(name='bbox_loss'+str(i), data=bbox_loss_, grad_scale=4.0 / cfg.TRAIN.BATCH_ROIS)
                rcnn_label = label[i]

                rcnn_label = mx.sym.Reshape(data=rcnn_label, shape=(cfg.TRAIN.BATCH_IMAGES, -1), name='label_reshape'+str(i))
                cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TRAIN.BATCH_IMAGES, -1, num_classes),
                                    name='cls_prob_reshape'+str(i))
                bbox_loss = mx.sym.Reshape(data=bbox_loss, shape=(cfg.TRAIN.BATCH_IMAGES, -1, 4 * num_reg_classes),
                                    name='bbox_loss_reshape'+str(i))
                bbox_loss_list.append(bbox_loss)
                rcnn_label_list.append(rcnn_label)
                cls_prob_list.append(cls_prob)
            bbox_loss_concat = mx.symbol.Concat(bbox_loss_list[0],bbox_loss_list[1],bbox_loss_list[2],bbox_loss_list[3],dim=1,name='bbox_loss_concat')
            cls_prob_concat = mx.symbol.Concat(cls_prob_list[0],cls_prob_list[1],cls_prob_list[2],cls_prob_list[3],dim=1,name='cls_prob_concat')
        
            rcnn_label_concat = mx.symbol.Concat(rcnn_label_list[0],rcnn_label_list[1],rcnn_label_list[2],rcnn_label_list[3],dim=1,name='rcnn_label_concat')
            fpn_cls_prob_concat = mx.symbol.Concat(fpn_cls_prob[0],fpn_cls_prob[1],fpn_cls_prob[2],fpn_cls_prob[3],fpn_cls_prob[4],dim=2,name='fpn_cls_prob_concat')
            fpn_bbox_loss_concat = mx.symbol.Concat(fpn_bbox_loss[0],fpn_bbox_loss[1],fpn_bbox_loss[2],fpn_bbox_loss[3],fpn_bbox_loss[4],dim=2,name='fpn_bbox_loss_concat')
            group = mx.sym.Group([fpn_cls_prob_concat,fpn_bbox_loss_concat,cls_prob_concat, bbox_loss_concat, mx.sym.BlockGrad(rcnn_label_concat)])
        else:
            rois_pool_concat,rois_as = mx.sym.Custom(p2=fpn_p[0],p3=fpn_p[1],p4=fpn_p[2],p5=fpn_p[3],
                                      rois = rois,data =data,op_type='assign_rois',rcnn_strides='(4,8,16,32)', pool_h=14, pool_w=14,name='rois_as')
            fc_new_1 = mx.sym.FullyConnected(name='fc_new_1',weight=fc_new_1_weight,bias =fc_new_1_bias, data=rois_pool_concat, num_hidden=1024)
            
 
            fc_new_1_relu = mx.sym.Activation(data=fc_new_1, act_type='relu', name='fc_new_1_relu')
            fc_new_1_relu = mx.symbol.Dropout(data=fc_new_1_relu, p=0.5, name="drop6_1")
            fc_new_2 = mx.sym.FullyConnected(name='fc_new_2',weight=fc_new_2_weight,bias =fc_new_2_bias, data=fc_new_1_relu, num_hidden=1024)

            fc_new_2_relu = mx.sym.Activation(data=fc_new_2, act_type='relu', name='fc_new_2_relu')
            fc_new_2_relu = mx.symbol.Dropout(data=fc_new_2_relu, p=0.5, name="drop6_1")

            cls_score = mx.sym.FullyConnected(name='cls_score',weight=rcnn_cls_weight,bias =rcnn_cls_bias, data=fc_new_2_relu, num_hidden=num_classes)
            bbox_pred = mx.sym.FullyConnected(name='bbox_pred',weight=rcnn_bbox_weight,bias =rcnn_bbox_bias,data=fc_new_2_relu, num_hidden=num_reg_classes * 4)

            cls_prob = mx.sym.SoftmaxActivation(name='cls_prob', data=cls_score)
            cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TEST.BATCH_IMAGES, -1, num_classes),
                                        name='cls_prob_reshape')
            cls_prob = mx.sym.SoftmaxActivation(name='cls_prob', data=cls_score)
            cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TEST.BATCH_IMAGES, -1, num_classes),
                                            name='cls_prob_reshape')
            bbox_pred = mx.sym.Reshape(data=bbox_pred, shape=(cfg.TEST.BATCH_IMAGES, -1, 4 * num_reg_classes),
                                            name='bbox_pred_reshape')

            output = [rois_as, cls_prob, bbox_pred]
            group = mx.symbol.Group(output)

        self.sym = group
        return group     


    def init_weight(self, cfg, arg_params, aux_params):

        arg_params['fpn_conv_weight'] = mx.random.normal(0, 0.001, shape=self.arg_shape_dict['fpn_conv_weight'])
        arg_params['fpn_conv_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fpn_conv_bias'])
        arg_params['fpn_cls_weight'] = mx.random.normal(0, 0.001, shape=self.arg_shape_dict['fpn_cls_weight'])
        arg_params['fpn_cls_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fpn_cls_bias'])
        arg_params['fpn_bbox_pred_weight'] = mx.random.normal(0, 0.001, shape=self.arg_shape_dict['fpn_bbox_pred_weight'])
        arg_params['fpn_bbox_pred_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fpn_bbox_pred_bias'])


        arg_params['stage5_conv_1x1_weight'] = mx.random.normal(0, 0.001, shape=self.arg_shape_dict['stage5_conv_1x1_weight'])
        arg_params['stage5_conv_1x1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['stage5_conv_1x1_bias'])
        arg_params['up_stage4_conv_1x1_weight'] = mx.random.normal(0, 0.001, shape=self.arg_shape_dict['up_stage4_conv_1x1_weight'])
        arg_params['up_stage4_conv_1x1_bias'] = mx.random.normal(0, 0.001, shape=self.arg_shape_dict['up_stage4_conv_1x1_bias'])
        arg_params['up_stage4_conv_3x3_weight'] = mx.random.normal(0, 0.001, shape=self.arg_shape_dict['up_stage4_conv_3x3_weight'])
        arg_params['up_stage4_conv_3x3_bias'] = mx.random.normal(0, 0.001, shape=self.arg_shape_dict['up_stage4_conv_3x3_bias'])
        arg_params['up_stage3_conv_1x1_weight'] = mx.random.normal(0, 0.001, shape=self.arg_shape_dict['up_stage3_conv_1x1_weight'])
        arg_params['up_stage3_conv_1x1_bias'] = mx.random.normal(0, 0.001, shape=self.arg_shape_dict['up_stage3_conv_1x1_bias'])    
        arg_params['up_stage3_conv_3x3_weight'] = mx.random.normal(0, 0.001, shape=self.arg_shape_dict['up_stage3_conv_3x3_weight'])
        arg_params['up_stage3_conv_3x3_bias'] = mx.random.normal(0, 0.001, shape=self.arg_shape_dict['up_stage3_conv_3x3_bias'])          
        arg_params['up_stage2_conv_1x1_weight'] = mx.random.normal(0, 0.001, shape=self.arg_shape_dict['up_stage2_conv_1x1_weight'])
        arg_params['up_stage2_conv_1x1_bias'] = mx.random.normal(0, 0.001, shape=self.arg_shape_dict['up_stage2_conv_1x1_bias']) 
        arg_params['up_stage2_conv_3x3_weight'] = mx.random.normal(0, 0.001, shape=self.arg_shape_dict['up_stage2_conv_3x3_weight'])
        arg_params['up_stage2_conv_3x3_bias'] = mx.random.normal(0, 0.001, shape=self.arg_shape_dict['up_stage2_conv_3x3_bias'])     
        arg_params['stage6_conv_3*3_weight'] = mx.random.normal(0, 0.001, shape=self.arg_shape_dict['stage6_conv_3*3_weight'])
        arg_params['stage6_conv_3*3_bias'] = mx.random.normal(0, 0.001, shape=self.arg_shape_dict['stage6_conv_3*3_bias'])



     
# SHARE 

        arg_params['fc_new_1_weight'] = mx.random.normal(0, 0.001, shape=self.arg_shape_dict['fc_new_1_weight'])
        arg_params['fc_new_1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fc_new_1_bias'])
        arg_params['fc_new_2_weight'] =mx.random.normal(0, 0.001, shape=self.arg_shape_dict['fc_new_2_weight'])
        arg_params['fc_new_2_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fc_new_2_bias'])
        arg_params['rcnn_cls_weight'] = mx.random.normal(0, 0.001, shape=self.arg_shape_dict['rcnn_cls_weight'])
        arg_params['rcnn_cls_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rcnn_cls_bias'])
        arg_params['rcnn_bbox_weight'] =  mx.random.normal(0, 0.001, shape=self.arg_shape_dict['rcnn_bbox_weight'])
        arg_params['rcnn_bbox_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rcnn_bbox_bias'])




     




