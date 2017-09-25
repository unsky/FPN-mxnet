"""
RPN:
data =
    {'data': [num_images, c, h, w],
     'im_info': [num_images, 4] (optional)}
label =
    {'gt_boxes': [num_boxes, 5] (optional),
     'label': [batch_size, 1] <- [batch_size, num_anchors, feat_height, feat_width],
     'bbox_target': [batch_size, num_anchors, feat_height, feat_width],
     'bbox_weight': [batch_size, num_anchors, feat_height, feat_width]}
"""

import numpy as np
import numpy.random as npr

from utils.image import get_image, tensor_vstack
from generate_anchor import generate_anchors
from bbox.bbox_transform import bbox_overlaps, bbox_transform

import time
def get_rpn_testbatch(roidb, cfg):
    """
    return a dict of testbatch
    :param roidb: ['image', 'flipped']
    :return: data, label, im_info
    """
    # assert len(roidb) == 1, 'Single batch only'
    print roidb
    imgs, roidb = get_image(roidb, cfg)

    im_array = imgs
    im_info = [np.array([roidb[i]['im_info']], dtype=np.float32) for i in range(len(roidb))]

    data = [{'data': im_array[i],
            'im_info': im_info[i]} for i in range(len(roidb))]
    label = {}

    return data, label, im_info


def get_rpn_batch(roidb, cfg):
    """
    prototype for rpn batch: data, im_info, gt_boxes
    :param roidb: ['image', 'flipped'] + ['gt_boxes', 'boxes', 'gt_classes']
    :return: data, label
    """
    assert len(roidb) == 1, 'Single batch only'
    imgs, roidb = get_image(roidb, cfg)
    im_array = imgs[0]
    im_info = np.array([roidb[0]['im_info']], dtype=np.float32)
#    print roidb[0]
    # gt boxes: (x1, y1, x2, y2, cls)
    if roidb[0]['gt_classes'].size > 0:
        gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
        gt_boxes = np.empty((roidb[0]['boxes'].shape[0], 5), dtype=np.float32)
        gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :]
        gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
    else:
        gt_boxes = np.empty((0, 5), dtype=np.float32)

    data = {'data': im_array,
            'im_info': im_info}
    label = {'gt_boxes': gt_boxes}

    return data, label


def assign_anchor(feat_shape_p3,feat_shape_p4,feat_shape_p5,feat_shape_p6,
                  gt_boxes, im_info, cfg,
                  feat_stride_p3=4,scales_p3=(8,), ratios_p3=(0.75, 1, 1.5),
                  feat_stride_p4=8,scales_p4=(8,), ratios_p4=(0.75, 1, 1.5),
                  feat_stride_p5=16,scales_p5=(8,), ratios_p5=(0.75, 1, 1.5),
                  feat_stride_p6=16,scales_p6=(8,), ratios_p6=(0.75, 1, 1.5),
                  allowed_border=1000):
    
    """
    assign ground truth boxes to anchor positions
    :param feat_shape: list of infer output shape
    :param gt_boxes: assign ground truth:[n, 5]
    :param im_info: filter out anchors overlapped with edges
    :param feat_stride: anchor position step
    :param scales: used to generate anchors, affects num_anchors (per location)
    :param ratios: aspect ratios of generated anchors
    :param allowed_border: filter out anchors with edge overlap > allowed_border
    :return: dict of label
    'label': of shape (batch_size, 1) <- (batch_size, num_anchors, feat_height, feat_width)
    'bbox_target': of shape (batch_size, num_anchors * 4, feat_height, feat_width)
    'bbox_inside_weight': *todo* mark the assigned anchors
    'bbox_outside_weight': used to normalize the bbox_loss, all weights sums to RPN_POSITIVE_WEIGHT
    """
    allowed_border=1000
    feat_shape = [feat_shape_p3,feat_shape_p4,feat_shape_p5,feat_shape_p6]
    feat_stride=[4,8,16,32]
    scales=(8,10,12)
    ratios=(0.5, 1, 2)
    

    def _unmap(data, count, inds, fill=0, allowed_border=allowed_border):
        """" unmap a subset inds of data into original data of size count """
        if allowed_border:
            return data
        if len(data.shape) == 1:
            ret = np.empty((count,), dtype=np.float32)
            ret.fill(fill)
            ret[inds] = data
        else:
            ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
            ret.fill(fill)
            ret[inds, :] = data
        return ret

    DEBUG = False
    debug = True
    im_info = im_info[0]
    #print 'im_info: ', im_info
    scales = np.array(scales, dtype=np.float32)
    if len(feat_stride) != len(feat_shape):
        assert('length of feat_stride is not equal to length of feat_shape')
    all_anchors_list = []
    anchors_counter = []
    total_anchors = 0
    t = time.time()
    #print 'length of feat_shape: ',len(feat_shape)
    for i in range(len(feat_shape)):
        base_anchors = generate_anchors(base_size=feat_stride[i], ratios=list(ratios), scales=scales)
        num_anchors = base_anchors.shape[0]#3
        #print feat_shape[i]
        feat_height, feat_width = (feat_shape[i])[-2:]

        if DEBUG:
            print 'anchors:'
            print base_anchors
            print 'anchor shapes:'
            print np.hstack((base_anchors[:, 2::4] - base_anchors[:, 0::4],
                             base_anchors[:, 3::4] - base_anchors[:, 1::4]))
            print 'im_info', im_info
            print 'height', feat_height, 'width', feat_width
            print 'gt_boxes shape', gt_boxes.shape
            print 'gt_boxes', gt_boxes

        # 1. generate proposals from bbox deltas and shifted anchors
        shift_x = np.arange(0, feat_width) * feat_stride[i]
        shift_y = np.arange(0, feat_height) * feat_stride[i]
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = num_anchors#3
        K = shifts.shape[0]#h*w
        i_all_anchors = base_anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
        i_all_anchors = i_all_anchors.reshape((K * A, 4))#(k*A,4) in the original image
        all_anchors_list.append(i_all_anchors)
        i_total_anchors = int(K * A)#3*w*h
        total_anchors += i_total_anchors
        anchors_counter.append(total_anchors)

        # only keep anchors inside the image, but in FPN, author allowed anchor outside of image
        # inds_inside = np.where((all_anchors[:, 0] >= -allowed_border) &
        #                        (all_anchors[:, 1] >= -allowed_border) &
        #                        (all_anchors[:, 2] < im_info[1] + allowed_border) &
        #                        (all_anchors[:, 3] < im_info[0] + allowed_border))[0]
        if DEBUG:
            print 'total_anchors', i_total_anchors
            #print 'inds_inside', len(inds_inside)

        # keep only inside anchors
        #anchors = all_anchors[inds_inside, :]
        if DEBUG:
            print 'anchors shape', anchors.shape

    all_anchors = np.array(all_anchors_list[0])#(3*h1*w1,4)
    for i_anchors in all_anchors_list[1:]:
        all_anchors = np.vstack((all_anchors, i_anchors))
    #all_anchors:[total_anchors,4]
    # label: 1 is positive, 0 is negative, -1 is dont care
    labels = np.empty((total_anchors,), dtype=np.float32)
    labels.fill(-1)
    #print 'get anchors spends :{:.4f}s'.format(time.time()-t)
    t_1 = time.time()
    if gt_boxes.size > 0:
        # overlap between the anchors and the gt boxes
        # overlaps (ex, gt)
        #t = time.time()
        overlaps = bbox_overlaps(all_anchors.astype(np.float), gt_boxes.astype(np.float))
        #print 'bbox overlaps spends :{:.4f}s'.format(time.time()-t)
        argmax_overlaps = overlaps.argmax(axis=1)
        max_overlaps = overlaps[np.arange(total_anchors), argmax_overlaps]
        gt_argmax_overlaps = overlaps.argmax(axis=0)
        gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

        if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels first so that positive labels can clobber them
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        # fg label: for each gt, anchor with highest overlap
        labels[gt_argmax_overlaps] = 1

        # fg label: above threshold IoU
        labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

        if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels last so that negative labels can clobber positives
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
    else:
        labels[:] = 0
    t_1_1 = time.time()
    # subsample positive labels if we have too many
    num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCH_SIZE)
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg:
        disable_inds = npr.choice(fg_inds, size=(len(fg_inds) - num_fg), replace=False)
        if DEBUG:
            disable_inds = fg_inds[:(len(fg_inds) - num_fg)]
        labels[disable_inds] = -1

    # subsample negative labels if we have too many
    num_bg = cfg.TRAIN.RPN_BATCH_SIZE - np.sum(labels == 1)
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = npr.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        if DEBUG:
            disable_inds = bg_inds[:(len(bg_inds) - num_bg)]
        labels[disable_inds] = -1

    bbox_targets = np.zeros((total_anchors, 4), dtype=np.float32)
    if gt_boxes.size > 0:
        bbox_targets[:] = bbox_transform(all_anchors, gt_boxes[argmax_overlaps, :4])

    bbox_weights = np.zeros((total_anchors, 4), dtype=np.float32)
    bbox_weights[labels == 1, :] = np.array(cfg.TRAIN.RPN_BBOX_WEIGHTS)

    if DEBUG:
        _sums = bbox_targets[labels == 1, :].sum(axis=0)
        _squared_sums = (bbox_targets[labels == 1, :] ** 2).sum(axis=0)
        _counts = np.sum(labels == 1)
        means = _sums / (_counts + 1e-14)
        stds = np.sqrt(_squared_sums / _counts - means ** 2)
        print 'means', means
        print 'stdevs', stds
    #print 'choose labels spends :{:.4f}s'.format(time.time()-t_1_1)
    #print 'sort labels spends :{:.4f}s'.format(time.time()-t_1)
    # map up to original set of anchors
    t_2 = time.time()
    labels_list = []
    bbox_targets_list = []
    bbox_weights_list = []
    labels_list.append(_unmap(labels[:anchors_counter[0]], anchors_counter[0], range(anchors_counter[0]), fill=-1))
    bbox_targets_list.append(_unmap(bbox_targets[range(anchors_counter[0]),:], anchors_counter[0], range(anchors_counter[0]), fill=0))
    bbox_weights_list.append(_unmap(bbox_weights[range(anchors_counter[0]),:], anchors_counter[0], range(anchors_counter[0]), fill=0))
    for i in range(1, len(feat_shape)):
        count = anchors_counter[i]-anchors_counter[i-1]
        labels_list.append(_unmap(labels[anchors_counter[i-1]:anchors_counter[i]], count, range(count), fill=-1)) 
        bbox_targets_list.append(_unmap(bbox_targets[anchors_counter[i-1]:anchors_counter[i],:], count, range(count), fill=0))
        bbox_weights_list.append(_unmap(bbox_weights[anchors_counter[i-1]:anchors_counter[i],:], count, range(count), fill=0))
    if DEBUG:
#         print 'rpn: max max_overlaps', np.max(max_overlaps)
        print 'rpn: num_positives', np.sum(labels == 1)
        print 'rpn: num_negatives', np.sum(labels == 0)
        _fg_sum = np.sum(labels == 1)
        _bg_sum = np.sum(labels == 0)
        _count = 1
        print 'rpn: num_positive avg', _fg_sum / _count
        print 'rpn: num_negative avg', _bg_sum / _count
    feat_heights = []
    feat_widths = []
    for i in range(len(feat_shape)):
        feat_heights.append(feat_shape[i][-2])
        feat_widths.append(feat_shape[i][-1])
    #print '_unmap spends :{:.4f}s'.format(time.time()-t_2)
    label1 = labels_list[0].reshape((1, feat_heights[0], feat_widths[0], A)).transpose(0, 3, 1, 2)
    labels1 = label1.reshape((1, A * feat_heights[0] * feat_widths[0]))
    bbox_targets1 = bbox_targets_list[0].reshape((1, feat_heights[0], feat_widths[0], A * 4)).transpose(0, 3, 1, 2)
    bbox_weights1 = bbox_weights_list[0].reshape((1, feat_heights[0], feat_widths[0], A * 4)).transpose((0, 3, 1, 2))

    label2 = labels_list[1].reshape((1, feat_heights[1], feat_widths[1], A)).transpose(0, 3, 1, 2)
    labels2 = label2.reshape((1, A * feat_heights[1] * feat_widths[1]))
    bbox_targets2 = bbox_targets_list[1].reshape((1, feat_heights[1], feat_widths[1], A * 4)).transpose(0, 3, 1, 2)
    bbox_weights2 = bbox_weights_list[1].reshape((1, feat_heights[1], feat_widths[1], A * 4)).transpose((0, 3, 1, 2))

    label3 = labels_list[2].reshape((1, feat_heights[2], feat_widths[2], A)).transpose(0, 3, 1, 2)
    labels3 = label3.reshape((1, A * feat_heights[2] * feat_widths[2]))
    bbox_targets3 = bbox_targets_list[2].reshape((1, feat_heights[2], feat_widths[2], A * 4)).transpose(0, 3, 1, 2)
    bbox_weights3 = bbox_weights_list[2].reshape((1, feat_heights[2], feat_widths[2], A * 4)).transpose((0, 3, 1, 2))

    if len(feat_shape)>3:
        label4 = labels_list[3].reshape((1, feat_heights[3], feat_widths[3], A)).transpose(0, 3, 1, 2)
        labels4 = label4.reshape((1, A * feat_heights[3] * feat_widths[3]))
        bbox_targets4 = bbox_targets_list[3].reshape((1, feat_heights[3], feat_widths[3], A * 4)).transpose(0, 3, 1, 2)
        bbox_weights4 = bbox_weights_list[3].reshape((1, feat_heights[3], feat_widths[3], A * 4)).transpose((0, 3, 1, 2))

    if len(feat_shape)>4:
        label5 = labels_list[4].reshape((1, feat_heights[4], feat_widths[4], A)).transpose(0, 3, 1, 2)
        labels5 = label5.reshape((1, A * feat_heights[4] * feat_widths[4]))
        bbox_targets5 = bbox_targets_list[4].reshape((1, feat_heights[4], feat_widths[4], A * 4)).transpose(0, 3, 1, 2)
        bbox_weights5 = bbox_weights_list[4].reshape((1, feat_heights[4], feat_widths[4], A * 4)).transpose((0, 3, 1, 2))
    if len(feat_shape)>5:
        assert ('RPN anchorloader only support max number of feature map of 5!')
      #  'label/p4': labels2, 'label/p5': labels3,
      #, 'bbox_target/p4': bbox_targets2, 'bbox_target/p5': bbox_targets3,
      #, 'bbox_weight/p4': bbox_weights2, 'bbox_weight/p5': bbox_weights3
    if len(feat_shape) == 3:
        label = {'label/p3': labels1, 'label/p4': labels2, 'label/p5': labels3,
          'bbox_target/p3': bbox_targets1,'bbox_target/p4': bbox_targets2, 'bbox_target/p5': bbox_targets3,
             'bbox_weight/p3': bbox_weights1,'bbox_weight/p4': bbox_weights2, 'bbox_weight/p5': bbox_weights3,
            }
    elif len(feat_shape) == 4:
        label = {'label/p3': labels1, 'label/p4': labels2, 'label/p5': labels3, 'label/p6': labels4,
                    'bbox_target/p3': bbox_targets1, 'bbox_target/p4': bbox_targets2, 'bbox_target/p5': bbox_targets3, 'bbox_target/p6': bbox_targets4,
                    'bbox_weight/p3': bbox_weights1, 'bbox_weight/p4': bbox_weights2, 'bbox_weight/p5': bbox_weights3, 'bbox_weight/p6': bbox_weights4}
    elif len(feat_shape) == 5:
        label = {'label1': labels1, 'label2': labels2, 'label3': labels3, 'label4': labels4, 'label5': labels5,
            'bbox_target1': bbox_targets1, 'bbox_target2': bbox_targets2, 'bbox_target3': bbox_targets3, 'bbox_target4': bbox_targets4, 'bbox_target5': bbox_targets5,
            'bbox_weight1': bbox_weights1, 'bbox_weight2': bbox_weights2, 'bbox_weight3': bbox_weights3, 'bbox_weight4': bbox_weights4, 'bbox_weight5':bbox_weights5}
    #print 'get labels spends :{:.4f}s'.format(time.time()-t_2)
    return label
