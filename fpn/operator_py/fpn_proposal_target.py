"""
Proposal Target Operator selects foreground and background roi and assigns label, bbox_transform to .
"""

import mxnet as mx
import numpy as np
from distutils.util import strtobool


from rcnn_get_batch import sample_rois
DEBUG = False
import time

class FPNProposalTargetOperator(mx.operator.CustomOp):
    def __init__(self, num_classes, batch_images, batch_rois, fg_fraction):
        super(FPNProposalTargetOperator, self).__init__()
        self._num_classes = num_classes
        self._batch_images = batch_images
        self._batch_rois = batch_rois#512
        self._fg_fraction = fg_fraction

        if DEBUG:
            self._count = 0
            self._fg_num = 0
            self._bg_num = 0

    def forward(self, is_train, req, in_data, out_data, aux):
        assert self._batch_rois % self._batch_images == 0, \
            'BATCHIMAGES {} must devide BATCH_ROIS {}'.format(self._batch_images, self._batch_rois)
        rois_per_image = self._batch_rois / self._batch_images#512
        fg_rois_per_image = np.round(self._fg_fraction * rois_per_image).astype(int)
        if DEBUG:
            pydevd.settrace('10.98.39.247', port=10001, stdoutToServer=True, stderrToServer=True)
        all_rois = in_data[0].asnumpy()#2000
        gt_boxes = in_data[1].asnumpy()
        #t = time.time()
        #all_rois[all_rois[:,0]!=0,:] = [0,0,0,0,0] #avoid Proposal bug(64stride feature map has no 400 anchor)
        #print 1
        if gt_boxes[0, 4] != -1:
            # Include ground-truth boxes in the set of candidate rois
            zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
            all_rois = np.vstack((all_rois, np.hstack((zeros, gt_boxes[:, :-1]))))
        # Sanity check: single batch opudbnly
        
        # for i in range(all_rois.shape[0]):
        #     print all_rois[i,:]
        # print all_rois.shape
        assert np.all(all_rois[:, 0] == 0), 'Only single item batches are supported'

        rois, labels, bbox_targets, bbox_weights, layer_indexs = \
            sample_rois(all_rois, fg_rois_per_image, rois_per_image, self._num_classes, gt_boxes=gt_boxes, sample_type='fpn', k0 = 4)
        #print 'sample rois spends :{:.4f}s'.format(time.time()-t)
        #print 3
        if DEBUG:
            pydevd.settrace('10.98.39.247', port=10001, stdoutToServer=True, stderrToServer=True)
  
        rois_all = np.zeros((self._batch_rois*4, 5), dtype=rois.dtype)
        labels_all = np.ones((self._batch_rois*4, ), dtype=labels.dtype)*-1
        bbox_targets_all = np.zeros((self._batch_rois*4, self._num_classes * 4), dtype=bbox_targets.dtype)
        bbox_weights_all = np.zeros((self._batch_rois*4, self._num_classes * 4), dtype=bbox_weights.dtype)
        for i in range(4):
            index = (layer_indexs == (i + 2))
            num_index = sum(index)
            start = self._batch_rois*i
            end = start+num_index
            index_range = range(start, end)
            rois_all[index_range, :] = rois[index, :]
            labels_all[index_range] = labels[index]  
            bbox_targets_all[index_range,:] = bbox_targets[index, :]
            bbox_weights_all[index_range,:] = bbox_weights[index, :]


        if DEBUG:
            print "labels=", labels_all
            print 'num fg: {}'.format((labels_all > 0).sum())
            print 'num bg: {}'.format((labels_all == 0).sum())
        #print 7
        if DEBUG:
            pydevd.settrace('10.98.39.247', port=10001, stdoutToServer=True, stderrToServer=True)
        for ind, val in enumerate([rois_all, labels_all, bbox_targets_all, bbox_weights_all]):
            self.assign(out_data[ind], req[ind], val)
        #print 'fpn proposal target spends :{:.4f}s'.format(time.time()-t)
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)


@mx.operator.register('fpn_proposal_target')
class FPNProposalTargetProp(mx.operator.CustomOpProp):
    def __init__(self, num_classes, batch_images, batch_rois, fg_fraction='0.25'):
        super(FPNProposalTargetProp, self).__init__(need_top_grad=False)
        self._num_classes = int(num_classes)
        self._batch_images = int(batch_images)
        self._batch_rois = int(batch_rois)
        self._fg_fraction = float(fg_fraction)

    def list_arguments(self):
        return ['rois', 'gt_boxes']

    def list_outputs(self):
        return ['rois_output', 'label', 'bbox_target', 'bbox_weight']

    def infer_shape(self, in_shape):
        rpn_rois_shape = in_shape[0]
        gt_boxes_shape = in_shape[1]

        output_rois_shape = (self._batch_rois*4, 5)#128*4,5
        label_shape = (self._batch_rois*4, )#128*4
        bbox_target_shape = (self._batch_rois*4, self._num_classes * 4)
        bbox_weight_shape = (self._batch_rois*4, self._num_classes * 4)

        return [rpn_rois_shape, gt_boxes_shape], \
               [output_rois_shape, label_shape, bbox_target_shape, bbox_weight_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return FPNProposalTargetOperator(self._num_classes, self._batch_images, self._batch_rois, self._fg_fraction)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []

