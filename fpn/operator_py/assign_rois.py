"""
fpn roi pooling
"""
from utils import image
import mxnet as mx
import numpy as np
DEBUG = False
import numpy.random as npr
from mxnet import autograd

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


def vis_all_detection(im_array, detections,  scale):
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
    for j in range(len(detections)):
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

    plt.show()
    name = np.mean(im)
    savefig ('vis/'+str(name)+'.png')
    plt.clf()
    plt.cla()
    

    plt. close(0)

class FPNROIOperator(mx.operator.CustomOp):
    def __init__(self, rcnn_strides, pool_h, pool_w):
        super(FPNROIOperator, self).__init__()
        self._feat_stride_fpn = np.fromstring(rcnn_strides[1:-1], dtype=int, sep=',')
        self._pool_h = int(pool_h)
        self._pool_w = int(pool_w)
        
    def forward(self, is_train, req, in_data, out_data, aux):
        fpn_feat_pyramid = {}
        
        fpn_feat_pyramid.update({'stride4':in_data[0]})
        fpn_feat_pyramid.update({'stride8':in_data[1]})
        fpn_feat_pyramid.update({'stride16':in_data[2]})
        fpn_feat_pyramid.update({'stride32':in_data[3]})
        rois = in_data[4][:]
        ro = in_data[4].asnumpy()[:]
        im = in_data[5]
        k0 =4
        w = (ro[:,3]-ro[:,1])
        h = (ro[:,4]-ro[:,2])
        s = w * h
        s[s<=0]=1e-6
        layer_index = np.floor(k0+np.log2(np.sqrt(s)/224))
        layer_index[layer_index<2]=2
        layer_index[layer_index>5]=5
        # Assign to levels
        roi_pool_list = []

        rois_list =[]
        for s in self._feat_stride_fpn:
                index =  np.where(layer_index==int(np.log2(int(s))))[0]
                if len(index)>0:
                    index = mx.nd.array(index, rois.context)
                    _rois = mx.nd.take(rois,index)   
                    roi_pool= mx.nd.ROIPooling(fpn_feat_pyramid['stride%s'%s], _rois, (self._pool_h, self._pool_w), 1.0 / float(s))
                    roi_pool_list.append(roi_pool)
                    rois_list.append(_rois)
        fpn_roi_pool = mx.nd.concatenate(roi_pool_list, axis=0)
        fpn_rois = mx.nd.concatenate(rois_list, axis=0)  
        vis = False
        if vis:
            vis_all_detection(im.asnumpy(),fpn_rois.asnumpy() ,1)
        print 'forward mean:', fpn_roi_pool.asnumpy().mean(),fpn_roi_pool.shape      
        self.assign(out_data[0], req[0], fpn_roi_pool)
        self.assign(out_data[1], req[0], fpn_rois)
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[0], 0)
        self.assign(in_grad[2], req[0], 0)
        self.assign(in_grad[3], req[0], 0)
        self.assign(in_grad[4], req[0], 0)
        self.assign(in_grad[5], req[0], 0)          
@mx.operator.register("assign_rois")
class FPNROIProp(mx.operator.CustomOpProp):
    def __init__(self, rcnn_strides='(4,8,16,32)', pool_h='7', pool_w='7'):
        super(FPNROIProp, self).__init__(need_top_grad=True)
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
        args_list.append('data')
        return args_list
    def list_outputs(self):
        return ['rois_pool_concat','rois']

    def infer_shape(self, in_shape):
        return in_shape, [[in_shape[4][0],256,self._pool_h,self._pool_w],[in_shape[4][0],5]], []

    def create_operator(self, ctx, shapes, dtypes):
        return FPNROIOperator(self._rcnn_strides, self._pool_h, self._pool_w)


