"""
Proposal Operator transform anchor coordinates into ROI coordinates with prediction results on
classification probability and bounding box prediction results, and image size and scale information.
"""

import os,sys
import mxnet as mx
import numpy as np
import numpy.random as npr
from distutils.util import strtobool
import copy


from bbox.bbox_transform import bbox_pred, clip_boxes
from rpn.generate_anchor import generate_anchors
from nms.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper

import time
# from rcnn.processing.bbox_transform import bbox_pred, clip_boxes
# from rcnn.processing.generate_anchor import generate_anchors
# from rcnn.processing.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper

DEBUG = False

class RoIConcatOperator(mx.operator.CustomOp):
    def __init__(self, feat_stride, scales, ratios, output_score,
                 rpn_pre_nms_top_n, rpn_post_nms_top_n, threshold, rpn_min_size):
        super(RoIConcatOperator, self).__init__()
        self._feat_stride = feat_stride
        self._scales = np.fromstring(scales[1:-1], dtype=float, sep=',')
        self._ratios = np.fromstring(ratios[1:-1], dtype=float, sep=',')
        self._output_score = output_score
        self._rpn_pre_nms_top_n = rpn_pre_nms_top_n
        self._rpn_post_nms_top_n = rpn_post_nms_top_n
        self._threshold = threshold
        self._rpn_min_size = rpn_min_size

        if DEBUG:
            print 'feat_stride: {}'.format(self._feat_stride)
            print 'anchors:'
            print self._anchors

    def forward(self, is_train, req, in_data, out_data, aux):
        nms = gpu_nms_wrapper(self._threshold, 0)
        batch_size = in_data[0].shape[0]
        if batch_size > 1:
            raise ValueError("Sorry, multiple images each device is not implemented")

        # for each (H, W) location i
        #   generate A anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the A anchors
        # clip predicted boxes to image
        # remove predicted boxes with either height or width < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)

        pre_nms_topN = self._rpn_pre_nms_top_n
        post_nms_topN = self._rpn_post_nms_top_n
        min_size = self._rpn_min_size

        # the first set of anchors are background probabilities
        # keep the second part
        scores_list = in_data[0].asnumpy()#[1,n]
        #print 'score_list shape:',scores_list.shape
        bbox_deltas_list = in_data[1].asnumpy()#[1,n*2]
        im_info = in_data[2].asnumpy()[0, :]
        p2_shape = in_data[3].asnumpy().shape
        p3_shape = in_data[4].asnumpy().shape
        p4_shape = in_data[5].asnumpy().shape
        p5_shape = in_data[6].asnumpy().shape
        p6_shape = in_data[7].asnumpy().shape
        feat_shape = []
        feat_shape.append(p2_shape)
        feat_shape.append(p3_shape)
        feat_shape.append(p4_shape)
        feat_shape.append(p5_shape)
        feat_shape.append(p6_shape)       
        #t = time.time()
        #print 'feat_shape:', feat_shape
        num_feat = len(feat_shape)#[1,5,4]
        score_index_start=0
        bbox_index_start=0
        keep_proposal = []
        keep_scores = []
    
        #t_1 = time.time()
        for i in range(num_feat):
            feat_stride = int(self._feat_stride[i])#4,8,16,32,64
            #print 'feat_stride:', feat_stride
            anchor = generate_anchors(feat_stride, scales=self._scales, ratios=self._ratios)
            num_anchors = anchor.shape[0]#3
           
            height = feat_shape[i][2]
            width = feat_shape[i][3]
   

            shift_x = np.arange(0, width) * feat_stride
            shift_y = np.arange(0, height) * feat_stride
            shift_x, shift_y = np.meshgrid(shift_x, shift_y)
            shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
            A = num_anchors#3
            K = shifts.shape[0]#height*width
            anchors = anchor.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
            anchors = anchors.reshape((K * A, 4))#3*height*widht,4
            scores = (scores_list[0,int(score_index_start):int(score_index_start+K*A*2)]).reshape((1,int(2*num_anchors),-1,int(width)))#1,2*3,h,w
            scores = scores[:,num_anchors:,:,:]#1,3,h,w
            bbox_deltas = (bbox_deltas_list[0,int(bbox_index_start):int(bbox_index_start+K*A*4)]).reshape((1,int(4*num_anchors),-1,int(width)))#1,4*3,h,w
            score_index_start += K*A*2
            bbox_index_start += K*A*4
            bbox_deltas = self._clip_pad(bbox_deltas, (height, width))
            bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))#[1,h,w,12]--->[1*h*w*3,4]
            scores = self._clip_pad(scores, (height, width))
            scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))#[1,h,w,3]--->[1*h*w*3,1]
            proposals = bbox_pred(anchors, bbox_deltas)#debug here, corresponding?
            proposals = clip_boxes(proposals, im_info[:2])
            keep = self._filter_boxes(proposals, min_size[i] * im_info[2])
            keep_proposal.append(proposals[keep, :])
            keep_scores.append(scores[keep])

        proposals = keep_proposal[0]
        scores = keep_scores[0]
        for i in range(1,num_feat):
            proposals=np.vstack((proposals, keep_proposal[i]))
            scores=np.vstack((scores, keep_scores[i]))
        #print 'roi concate t_1 spends :{:.4f}s'.format(time.time()-t_1)
        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN (e.g. 6000)
        #t_2 = time.time()
        order = scores.ravel().argsort()[::-1]
        if pre_nms_topN > 0:
            order = order[:pre_nms_topN]
        proposals = proposals[order, :]
        scores = scores[order]
        #print 'roi concate t_2_1_1 spends :{:.4f}s'.format(time.time()-t_2)
        # 6. apply nms (e.g. threshold = 0.7)
        # 7. take after_nms_topN (e.g. 300)
        # 8. return the top proposals (-> RoIs top)
        #t_nms = time.time()
        det = np.hstack((proposals, scores)).astype(np.float32)
        keep = nms(det)
        #print 'roi concate nms spends :{:.4f}s'.format(time.time()-t_nms)

        if post_nms_topN > 0:
            keep = keep[:post_nms_topN]
        # pad to ensure output size remains unchanged
        if len(keep) < post_nms_topN:
            try:
                pad = npr.choice(keep, size=post_nms_topN - len(keep))
            except:
                proposals = np.zeros((post_nms_topN, 4), dtype=np.float32)
                proposals[:,2] = 16
                proposals[:,3] = 16
                batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
                blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
                self.assign(out_data[0], req[0], blob)

                if self._output_score:
                    self.assign(out_data[1], req[1], scores.astype(np.float32, copy=False))
                return
            keep = np.hstack((keep, pad))
        proposals = proposals[keep, :]
        scores = scores[keep]
        #print 'roi concate t_2 spends :{:.4f}s'.format(time.time()-t_2)
        # Output rois array
        # Our RPN implementation only supports a single input image, so all
        # batch inds are 0
        batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
        blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
        self.assign(out_data[0], req[0], blob)

        if self._output_score:
            self.assign(out_data[1], req[1], scores.astype(np.float32, copy=False))
        #print 'roi concate spends :{:.4f}s'.format(time.time()-t)
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)
        self.assign(in_grad[2], req[2], 0)
        self.assign(in_grad[3], req[0], 0)
        self.assign(in_grad[4], req[0], 0)
        self.assign(in_grad[5], req[0], 0)
        self.assign(in_grad[6], req[0], 0)
        self.assign(in_grad[7], req[0], 0)
    @staticmethod
    def _filter_boxes(boxes, min_size):
        """ Remove all boxes with any side smaller than min_size """
        ws = boxes[:, 2] - boxes[:, 0] + 1
        hs = boxes[:, 3] - boxes[:, 1] + 1
        keep = np.where((ws >= min_size) & (hs >= min_size))[0]
        return keep

    @staticmethod
    def _clip_pad(tensor, pad_shape):
        """
        Clip boxes of the pad area.
        :param tensor: [n, c, H, W]
        :param pad_shape: [h, w]
        :return: [n, c, h, w]
        """
        H, W = tensor.shape[2:]
        h, w = pad_shape

        if h < H or w < W:
            tensor = tensor[:, :, :h, :w].copy()

        return tensor


@mx.operator.register("roi_concat")
class RoIConcatProp(mx.operator.CustomOpProp):
    def __init__(self, feat_stride='4,8,16,32,64', scales='(8)', ratios='(0.5, 1, 2)', output_score='False',
                 rpn_pre_nms_top_n='2000', rpn_post_nms_top_n='512', threshold='0.3', rpn_min_size='4,8,16,32,64'):
        super(RoIConcatProp, self).__init__(need_top_grad=False)
        self._feat_stride = [int(i) for i in feat_stride.split(',')]
        self._scales = scales
        self._ratios = ratios
        self._output_score = strtobool(output_score)
        self._rpn_pre_nms_top_n = int(rpn_pre_nms_top_n)
        self._rpn_post_nms_top_n = int(rpn_post_nms_top_n)
        self._threshold = float(threshold)
        self._rpn_min_size = [int(i) for i in rpn_min_size.split(',')]
    def list_arguments(self):
        return ['cls_prob', 'bbox_pred', 'im_info', 'p2','p3','p4','p5','p6']
    def list_outputs(self):
        if self._output_score:
            return ['output', 'score']
        else:
            return ['output']

    def infer_shape(self, in_shape):
        flatten_cls_prob_shape = in_shape[0]
        flatten_bbox_pred_shape = in_shape[1]
        im_info_shape = in_shape[2]


        assert flatten_cls_prob_shape[0] == flatten_bbox_pred_shape[0], 'ROI number does not equal in cls and reg'

        

        output_shape = (self._rpn_post_nms_top_n, 5)
        score_shape = (self._rpn_post_nms_top_n, 1)

        if self._output_score:
            return [flatten_cls_prob_shape, flatten_bbox_pred_shape, im_info_shape, feat_shape_shape], [output_shape, score_shape]
        else:
            return [flatten_cls_prob_shape, flatten_bbox_pred_shape, im_info_shape, in_shape[3],in_shape[4],in_shape[5],in_shape[6],in_shape[7]], [output_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return RoIConcatOperator(self._feat_stride, self._scales, self._ratios, self._output_score,
                                self._rpn_pre_nms_top_n, self._rpn_post_nms_top_n, self._threshold, self._rpn_min_size)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
