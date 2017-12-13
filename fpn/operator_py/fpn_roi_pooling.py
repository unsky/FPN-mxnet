"""
fpn roi pooling
"""

import mxnet as mx
import numpy as np
DEBUG = False
import numpy.random as npr
from mxnet import autograd
from utils import image
num =0

def _unmap(data, count, inds):
    """" unmap a subset inds of data into original data of size count """
    assert len(inds) == count
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=np.float32)
        ret[inds] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
        ret[inds, :] = data
    return ret

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

        bbox = det[1:] 
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

    

class FPNROIPoolOperator(mx.operator.CustomOp):
    def __init__(self, rcnn_strides, pool_h, pool_w):
        super(FPNROIPoolOperator, self).__init__()
        self._feat_stride_fpn = np.fromstring(rcnn_strides[1:-1], dtype=int, sep=',')
        self._pool_h = int(pool_h)
        self._pool_w = int(pool_w)
        self.num =0
        
    def forward(self, is_train, req, in_data, out_data, aux):
        fpn_feat_pyramid = {}
        
        fpn_feat_pyramid.update({'stride4':in_data[0]})
        fpn_feat_pyramid.update({'stride8':in_data[1]})
        fpn_feat_pyramid.update({'stride16':in_data[2]})
        fpn_feat_pyramid.update({'stride32':in_data[3]})



        rois = in_data[4]
        num = rois.asnumpy().shape[0]
        label = in_data[5]
        bbox_target = in_data[6]
        bbox_weight = in_data[7]

        im = in_data[8]
        
        
        num_rois = rois.shape[0]
        rois_x1 = mx.nd.slice(rois, begin=(0, 1), end=(num_rois, 2))
        rois_y1 = mx.nd.slice(rois, begin=(0, 2), end=(num_rois, 3))
        rois_x2 = mx.nd.slice(rois, begin=(0, 3), end=(num_rois, 4))
        rois_y2 = mx.nd.slice(rois, begin=(0, 4), end=(num_rois, 5))
     
        area =  (rois_y2 - rois_y1) * (rois_x2 - rois_x1)
        area = area.asnumpy()
        print '-----------------------------------------------------------------------------',len(area[area<0])
        
        area[area<0]=0
        area = mx.nd.array(area)
        rois_area = mx.nd.sqrt(area)
        rois_area = rois_area.asnumpy()
      #  print rois_area
    

        if DEBUG:
            print 'rois_area shape:', rois_area.shape

        feat_dict = {}
        for stride in self._feat_stride_fpn:
            feat_dict.update({'stride%s'%stride:fpn_feat_pyramid['stride%s'%stride]})

        area_threshold = {'stride32':[np.inf, 448],
                          'stride16':[448,    224],
                          'stride8' :[224,    112],
                          'stride4' :[112,      -1*np.inf]}

       # area_threshold.update({'stride%s'%self._feat_stride_fpn[-1]:[area_threshold['stride%s'%self._feat_stride_fpn[-1]][0], 0]})

        # Assign to levels
        roi_pool_list = []
        index_list = []
        label_list =[]
        bbox_target_list = []
        bbox_weight_list = []
        rois_list ={}
        index_list = []
        rois_ =[]
        for s in self._feat_stride_fpn:
            thd = area_threshold['stride%s'%s]
            index = np.where(np.logical_and(thd[1] <= rois_area, rois_area < thd[0]))[0]
            
           # print len(index)
            
       #     print "stride: %s, num rois: %d" % (s, len(index))

            if len(index) > 0:
                index = mx.nd.array(index)
                if DEBUG:
                    print 'Context:'
                    print 'feat:', feat_dict['stride%s'%s].context
                    print 'rois:', rois.context
                    print 'index:', index.context
                _rois = mx.nd.take(rois, index)
                rois_list.update({'stride%s'%s:_rois})
                _label = mx.nd.take(label, index)
                _bbox_target = mx.nd.take(bbox_target, index)
                _bbox_weight = mx.nd.take(bbox_weight, index)
                _index = index[:]
                _index[:] = s 
                rois_.append(_rois)
                
    
                index_list.append(_index)
     
                roi_pool= mx.nd.ROIPooling(feat_dict['stride%s'%s], _rois, (self._pool_h, self._pool_w), 1.0 / float(s))
              #  print '***********************',rois_area.shape,index.shape,roi_pool.shape
                roi_pool_list.append(roi_pool)
                label_list.append(_label)
                bbox_target_list.append(_bbox_target)
                bbox_weight_list.append(_bbox_weight)
        fpn_roi_pool = mx.nd.concatenate(roi_pool_list, axis=0)
        label = mx.nd.concatenate(label_list, axis=0)
        rois = mx.nd.concatenate(rois_, axis=0)

        vis = False
        if vis:
            vis_all_detection(im.asnumpy(),rois.asnumpy(), label.asnumpy(), 1)

        index_ = mx.nd.concatenate(index_list, axis=0)
        self.rois_list = rois_list
        self.index_ = index_

        bbox_target = mx.nd.concatenate(bbox_target_list, axis=0)
        bbox_weight = mx.nd.concatenate(bbox_weight_list, axis=0)

            
        

        print 'forward mean:', fpn_roi_pool.asnumpy().mean()      
        for ind, val in enumerate([fpn_roi_pool, label, bbox_target, bbox_weight]):
            self.assign(out_data[ind], req[0], val)
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
            p2 = in_data[0][:]
            
        #    print self.rois_list
            index = self.index_.asnumpy().astype('int')
     
            grad = out_grad[0].asnumpy()[:]
            batch_size = grad.shape[0]
            
         #   print index
            if 'stride4' in self.rois_list:
                 p2.attach_grad()
                 rois_p2 = self.rois_list['stride4'][:]
                 with autograd.record():
                      roi_pool= mx.nd.ROIPooling(data=p2, rois=rois_p2,  pooled_size=(self._pool_h, self._pool_w), spatial_scale=1.0 /4.0)
             #    print len(index[index==4])
                
                 grad_p2 = mx.nd.array(grad[np.where(index==4)])
              #   print grad_p2
                 num_grad = grad_p2.shape[0]
                 scale = (batch_size*1.0)/num_grad
                 roi_pool.backward(grad_p2)
                 self.assign(in_grad[0], req[0], p2.grad*scale)
            else :
                 self.assign(in_grad[0], req[0], 0)
            
            p3 = in_data[1][:]
           
            if 'stride8' in self.rois_list:
                 p3.attach_grad()
                 rois_p3 = self.rois_list['stride8'][:]
                 with autograd.record():
                     roi_pool= mx.nd.ROIPooling(data=p3, rois=rois_p3,  pooled_size=(self._pool_h, self._pool_w), spatial_scale=1.0 /8.0)
                 grad_p3 = mx.nd.array(grad[np.where(index==8)])
                 num_grad = grad_p3.shape[0]
                 scale = (batch_size*1.0)/num_grad
                 roi_pool.backward(grad_p3)

                 self.assign(in_grad[1], req[0], p3.grad*scale)
            else :
                 self.assign(in_grad[1], req[0], 0)
           
            
            p4 = in_data[2][:]
          
            if 'stride16' in self.rois_list:
                 p4.attach_grad()
                 rois_p4 = self.rois_list['stride16'][:]
                 with autograd.record():
                     roi_pool= mx.nd.ROIPooling(data=p4, rois=rois_p4,  pooled_size=(self._pool_h, self._pool_w), spatial_scale=1.0 /16.0)
                 grad_p4 = mx.nd.array(grad[np.where(index==16)])
                 roi_pool.backward(grad_p4)    
                 num_grad = grad_p4.shape[0]
                 scale = (batch_size*1.0)/num_grad                 
                 self.assign(in_grad[2], req[0], p4.grad*scale)

            else :
                 self.assign(in_grad[2], req[0], 0)

                 
            p5 = in_data[3][:]
            
            if 'stride32' in self.rois_list:
                 p5.attach_grad()
                 rois_p5 = self.rois_list['stride32'][:]
                 with autograd.record():
                     roi_pool= mx.nd.ROIPooling(data=p5, rois=rois_p5,  pooled_size=(self._pool_h, self._pool_w), spatial_scale=1.0 /32.0)
                 grad_p5 = mx.nd.array(grad[np.where(index==32)])
                 roi_pool.backward(grad_p5)
                 num_grad = grad_p5.shape[0]
                 scale = (batch_size*1.0)/num_grad
                 self.assign(in_grad[3], req[0], p5.grad*scale)

            else :
                 self.assign(in_grad[3], req[0], 0)
        #    print 'mean:', in_grad[0].asnumpy().mean(),in_grad[1].asnumpy().mean(),in_grad[2].asnumpy().mean(),in_grad[3].asnumpy().mean()
      #      print out_grad[0][0:len(rois_p2)].shape
            self.assign(in_grad[4], req[0], 0)
            self.assign(in_grad[5], req[0], 0)
            self.assign(in_grad[6], req[0], 0)
            self.assign(in_grad[7], req[0], 0)
            self.assign(in_grad[8], req[0], 0)
            
@mx.operator.register("fpn_roi_pool")
class FPNROIPoolProp(mx.operator.CustomOpProp):
    def __init__(self, rcnn_strides='(4,8,16,32)', pool_h='7', pool_w='7'):
        super(FPNROIPoolProp, self).__init__(need_top_grad=True)
        self._pool_h = int(pool_h)
        self._pool_w = int(pool_w)
        self._rcnn_strides = rcnn_strides
   

    def list_arguments(self):
        args_list = []
        args_list.append('p2')
        args_list.append('p3')
        args_list.append('p4')
        args_list.append('p5')
        args_list.append('rois')
        args_list.append('label')
        args_list.append('bbox_target')
        args_list.append('bbox_weight')
        args_list.append('data')
        return args_list
    def list_outputs(self):
        return ['rois_pool_concat','label','bbox_target', 'bbox_weight']

    def infer_shape(self, in_shape):
        return in_shape, [[in_shape[5][0],256,self._pool_h,self._pool_h],in_shape[5],in_shape[6],in_shape[7]], []

    def create_operator(self, ctx, shapes, dtypes):
        return FPNROIPoolOperator(self._rcnn_strides, self._pool_h, self._pool_w)


