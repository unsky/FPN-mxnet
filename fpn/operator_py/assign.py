"""
Proposal Target Operator selects foreground and background roi and assigns label, bbox_transform to them.
"""

import mxnet as mx
import numpy as np
import numpy.random as npr




DEBUG = False


class AssignOperator(mx.operator.CustomOp):
    def __init__(self,rois_num,layer_num):
        super(AssignOperator, self).__init__()
        self.rois_num = int(rois_num/layer_num)
        self.layer_num = layer_num

    def forward(self, is_train, req, in_data, out_data, aux):

        rois = in_data[0].asnumpy()
        score = in_data[1].asnumpy()
        lenght = len(rois)
        # print lenght
        k0=4
        w = rois[:,3]-rois[:,1]
        h = rois[:,4]-rois[:,2]
        s = w * h
        s[s<=0] = 1e-6
        layer_indexs = np.floor(k0+np.log2(np.sqrt(s)/124))
        layer_indexs[layer_indexs<3] = 3
        layer_indexs[layer_indexs>6] = 6

        rois_list = []

        nms_topN = self.rois_num
        for i in range(self.layer_num):
            layer = i+3
            if len(rois[layer_indexs==layer])==0:
                if layer==3:
                    layer = 5
                if len(rois[layer_indexs==layer-1])==0:
                    layer = layer-1
                rois_ = rois[layer_indexs==layer-1][:]
                score_ = score[layer_indexs==layer-1][:]
            else:
                rois_ = rois[layer_indexs==layer]
                score_ = score[layer_indexs==layer]
            print layer,rois_.shape,score_.shape

            order = score_.ravel().argsort()[::-1]


            if len(score_) > nms_topN:
                order = order[:nms_topN]
                rois_ = rois_[order]
                score_ = score_[order]
            if len(score_) < nms_topN:
                keep6= np.where(score_>-100)
                pad = npr.choice(keep6[0], size=nms_topN - len(keep6[0]))
                keep_ = np.hstack((keep6[0], pad))
                rois_ = rois_[keep_, :]
                score = score_[keep_, :]
            rois_list.append(rois_)

   
  #      if len(score[3])==0:
   #         score[3] = score[2][0:10,:]
   #         rois[3] = rois[2][0:10,:]
   #     if len(score[0]) < nms_topN:
   #         keep6= np.where(score[0]>-100)
   #         pad = npr.choice(keep6[0], size=nms_topN - len(keep6[0]))
   #         keep_ = np.hstack((keep6[0], pad))
   #         rois[0] = rois[0][keep_, :]

    #    if len(score[3]) < nms_topN:
    #        keep6= np.where(score[3]>-100)
    #        pad = npr.choice(keep6[0], size=nms_topN - len(keep6[0]))
    #        keep_ = np.hstack((keep6[0], pad))
    #        rois[3] = rois[3][keep_, :]
        
        rois = np.vstack((rois_list[0],rois_list[1],rois_list[2],rois_list[3]))
 

        for ind, val in enumerate([rois]):
            self.assign(out_data[ind], req[ind], val)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)

    


@mx.operator.register('assign')
class AssignProp(mx.operator.CustomOpProp):
    def __init__(self,rois_num='128',layer_num='4'):
        super(AssignProp, self).__init__(need_top_grad=False)
        self.rois_num = int(rois_num)
        self.layer_num = int(layer_num)


    def list_arguments(self):
        return ['rois','score']

    def list_outputs(self):
        return ['rois_as']

    def infer_shape(self, in_shape):
        rois_shape = in_shape[0]
        score_shape = in_shape[1]
        rois_out_shape = [self.rois_num,5]
        
        return [rois_shape,score_shape], \
               [rois_out_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return AssignOperator(self.rois_num,self.layer_num)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
