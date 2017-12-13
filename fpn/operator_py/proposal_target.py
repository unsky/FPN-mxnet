
"""
Proposal Target Operator selects foreground and background roi and assigns label, bbox_transform to them.
"""

import mxnet as mx
import numpy as np
from distutils.util import strtobool
from easydict import EasyDict as edict
import cPickle
from bbox.bbox_transform import bbox_pred, clip_boxes

from core.rcnn import sample_rois
from utils import image
DEBUG = False



def vis_all_detection(im_array, detections, class_names, scale):
    """
    visualize all detections in one image
    :param im_array: [b=1 c h w] in rgb
    :param detections: [ numpy.ndarray([[x1 y1 x2 y2 score]]) for j in classes ]
    :param class_names: list of names in imdb
    :param scale: visualize the scaled image
    :return:
    """
    import matplotlib  
    matplotlib.use('Agg') 
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import savefig  
    import random
    a =  [103.06 ,115.9 ,123.15]
    a = np.array(a)
    im = image.transform_inverse(im_array,a)
    plt.imshow(im)
    for j in range(len(class_names)):
        if class_names[j] == 0:
            continue
        color = (random.random(), random.random(), random.random())  # generate a random color
        dets = detections[j]
        det =dets
        bbox = det[0:] 
        score = det[0]
        rect = plt.Rectangle((bbox[0], bbox[1]),
                                 bbox[2] - bbox[0],
                                 bbox[3] - bbox[1], fill=False,
                                 edgecolor=color, linewidth=3.5)
        plt.gca().add_patch(rect)
        plt.gca().text(bbox[0], bbox[1] - 2,
                           '{:s} {:.3f}'.format(str(class_names[j]), score),
                           bbox=dict(facecolor=color, alpha=0.5), fontsize=12, color='white')
    plt.show()
    name = np.mean(im)
    savefig ('vis/'+str(name)+'.png')
    plt.clf()
    plt.cla()

    plt. close(0)


class ProposalTargetOperator(mx.operator.CustomOp):
    def __init__(self, num_classes, batch_images, batch_rois, cfg, fg_fraction):
        super(ProposalTargetOperator, self).__init__()
        self._num_classes = num_classes
        self._batch_images = batch_images
        self._batch_rois = batch_rois
        self._cfg = cfg
        self._fg_fraction = fg_fraction
        

        if DEBUG:
            self._count = 0
            self._fg_num = 0
            self._bg_num = 0

    def forward(self, is_train, req, in_data, out_data, aux):
        assert self._batch_rois == -1 or self._batch_rois % self._batch_images == 0, \
            'batchimages {} must devide batch_rois {}'.format(self._batch_images, self._batch_rois)
        all_rois = in_data[0].asnumpy()
        gt_boxes = in_data[1].asnumpy()
        im = in_data[2].asnumpy()

        if self._batch_rois == -1:
            rois_per_image = all_rois.shape[0] + gt_boxes.shape[0]
            fg_rois_per_image = rois_per_image
        else:
            rois_per_image = self._batch_rois / self._batch_images
            fg_rois_per_image = np.round(self._fg_fraction * rois_per_image).astype(int)


        # Include ground-truth boxes in the set of candidate rois
        zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
        all_rois = np.vstack((all_rois, np.hstack((zeros, gt_boxes[:, :-1]))))
        # Sanity check: single batch only
        assert np.all(all_rois[:, 0] == 0), 'Only single item batches are supported'

        rois, labels, bbox_targets, bbox_weights ,layer_indexs= \
            sample_rois(all_rois, fg_rois_per_image, rois_per_image, self._num_classes, self._cfg, gt_boxes=gt_boxes,sample_type='fpn', k0 = 4)
    
        vis = False
        if vis:
            ind = np.where(labels!=0)[0]
            im_shape = im.shape
            pred_boxes = bbox_pred(rois[:,1:], bbox_targets)
            pred_boxes = clip_boxes(pred_boxes, im_shape[-2:])
            l =labels[ind]
            ro = rois[ind,1:]
            b = bbox_targets[ind,:]
            p = pred_boxes[ind,:]*bbox_weights[ind,:]
            r = []
            for i in range(p.shape[0]):
                r.append(p[i,l[i]*4:l[i]*4+4])
            r_ =  np.vstack(r)
            
            vis_all_detection(im, r_, l, 1)
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
            print "labels=", labels
            print 'num fg: {}'.format((labels > 0).sum())
            print 'num bg: {}'.format((labels == 0).sum())
            self._count += 1
            self._fg_num += (labels > 0).sum()
            self._bg_num += (labels == 0).sum()
            print "self._count=", self._count
            print 'num fg avg: {}'.format(self._fg_num / self._count)
            print 'num bg avg: {}'.format(self._bg_num / self._count)
            print 'ratio: {:.3f}'.format(float(self._fg_num) / float(self._bg_num))
        for ind, val in enumerate([rois_all, labels_all, bbox_targets_all, bbox_weights_all]):
            self.assign(out_data[ind], req[0], val)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[0], 0)
        self.assign(in_grad[2], req[0], 0)


@mx.operator.register('proposal_target')
class ProposalTargetProp(mx.operator.CustomOpProp):
    def __init__(self, num_classes, batch_images, batch_rois, cfg, fg_fraction='0.25'):
        super(ProposalTargetProp, self).__init__(need_top_grad=False)
        self._num_classes = int(num_classes)
        self._batch_images = int(batch_images)
        self._batch_rois = int(batch_rois)
        self._cfg = cPickle.loads(cfg)
        self._fg_fraction = float(fg_fraction)

    def list_arguments(self):
        return ['rois', 'gt_boxes','data']

    def list_outputs(self):
        return ['rois_output', 'label', 'bbox_target', 'bbox_weight']

    def infer_shape(self, in_shape):
        rpn_rois_shape = in_shape[0]
        gt_boxes_shape = in_shape[1]

        rois = rpn_rois_shape[0] + gt_boxes_shape[0] if self._batch_rois == -1 else self._batch_rois

        output_rois_shape = (rois*4, 5)
        label_shape = (rois*4, )
        bbox_target_shape = (rois*4, self._num_classes * 4)
        bbox_weight_shape = (rois*4, self._num_classes * 4)

        return [rpn_rois_shape, gt_boxes_shape,in_shape[2]], \
               [output_rois_shape, label_shape, bbox_target_shape, bbox_weight_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return ProposalTargetOperator(self._num_classes, self._batch_images, self._batch_rois, self._cfg, self._fg_fraction)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []