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
        self.rois_num = rois_num
        self.layer_num = layer_num

    def forward(self, is_train, req, in_data, out_data, aux):

        rois = in_data[0].asnumpy()
        score = in_data[1].asnumpy()

    #    print len(rois)
        k0=4
        w = rois[:,3]-rois[:,1]
        h = rois[:,4]-rois[:,2]
        s = w * h
        s[s<=0] = 1e-6
        layer_indexs = np.floor(k0+np.log2(np.sqrt(s)/63))
        layer_indexs[layer_indexs<3] = 3
   
        layer_indexs[layer_indexs>6] = 6

        print len(np.where(layer_indexs==3)[0])
        print len(np.where(layer_indexs==4)[0])
        print len(np.where(layer_indexs==5)[0])
        print len(np.where(layer_indexs==6)[0])


        rois3 = rois[layer_indexs==3]
        rois4 = rois[layer_indexs==4]
        rois5 = rois[layer_indexs==5]
        rois6 = rois[layer_indexs==6]

        score3 = score[layer_indexs==3]
        score4 = score[layer_indexs==4]
        score5 = score[layer_indexs==5]
        score6 = score[layer_indexs==6]
        # print len(np.where(layer_indexs==3)[0]),len(rois3),len(score3)
        # print len(np.where(layer_indexs==4)[0])
        # print len(np.where(layer_indexs==5)[0])
        # print len(np.where(layer_indexs==6)[0])       



        nms_topN = self.rois_num
        order = score3.ravel().argsort()[::-1]
        if len(score3) > nms_topN:
            order = order[:nms_topN]
        rois3 = rois3[order]
        score3 = score3[order]


        order = score4.ravel().argsort()[::-1]
        if len(score4) > nms_topN:
            order = order[:nms_topN]
        rois4 = rois4[order]
        score4 = score4[order]

        order = score5.ravel().argsort()[::-1]
        if len(score5) > nms_topN:
            order = order[:nms_topN]
        rois5 = rois5[order]
        score5 = score5[order]

        order = score6.ravel().argsort()[::-1]
        if len(score6) > nms_topN:
            order = order[:nms_topN]
        rois6 = rois6[order]
        score6 = score6[order]

        if len(score3) < nms_topN:
            keep3= np.where(score3>-100)
            pad = npr.choice(keep3[0], size=nms_topN - len(keep3[0]))
            keep_ = np.hstack((keep3[0], pad))
            rois3 = rois3[keep_, :]
        if len(score4) < nms_topN:
            keep4= np.where(score4>-100)
            pad = npr.choice(keep4[0], size=nms_topN - len(keep4[0]))
            keep_ = np.hstack((keep4[0], pad))
            rois4 = rois4[keep_, :]
        if len(score5) < nms_topN:
            keep5= np.where(score5>-100)
            pad = npr.choice(keep5[0], size=nms_topN - len(keep5[0]))
            keep_ = np.hstack((keep5[0], pad))
            rois5 = rois5[keep_, :]
        if len(score6) < nms_topN:
            keep6= np.where(score6>-100)
            pad = npr.choice(keep6[0], size=nms_topN - len(keep6[0]))
            keep_ = np.hstack((keep6[0], pad))
            rois6 = rois6[keep_, :]

     #   print len(rois3),len(rois4),len(rois5),len(rois6)
        rois= np.vstack((rois3,rois4,rois5,rois6))
        # print len(rois)

        for ind, val in enumerate([rois]):
            self.assign(out_data[ind], req[ind], val)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)

    


@mx.operator.register('assign')
class AssignProp(mx.operator.CustomOpProp):
    def __init__(self,rois_num='1000',layer_num='4'):
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
        rois_out_shape = [self.rois_num*self.layer_num,5]
        
        return [rois_shape,score_shape], \
               [rois_out_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return AssignOperator(self.rois_num,self.layer_num)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
