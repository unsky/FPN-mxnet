"""
Proposal Target Operator selects foreground and background roi and assigns label, bbox_transform to them.
"""

import mxnet as mx
import numpy as np
import numpy.random as npr




DEBUG = False


class AssignRoisOperator(mx.operator.CustomOp):
    def __init__(self):
        super(AssignRoisOperator, self).__init__()


    def forward(self, is_train, req, in_data, out_data, aux):

        rois = in_data[0].asnumpy()
        labels = in_data[1].asnumpy()
        bbox_target = in_data[2].asnumpy()
        bbox_weight = in_data[3].asnumpy()
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

        label3 = labels[layer_indexs==3]
        label4 = labels[layer_indexs==4]
        label5 = labels[layer_indexs==5]
        label6 = labels[layer_indexs==6]

        bbox_target3 = bbox_target[layer_indexs==3]
        bbox_target4 = bbox_target[layer_indexs==4]
        bbox_target5 = bbox_target[layer_indexs==5]
        bbox_target6 = bbox_target[layer_indexs==6]

        bbox_weight3 = bbox_weight[layer_indexs==3]
        bbox_weight4 = bbox_weight[layer_indexs==4]
        bbox_weight5 = bbox_weight[layer_indexs==5]
        bbox_weight6 = bbox_weight[layer_indexs==6]
        # print len(label3),len(bbox_target3),len(bbox_weight3)
        # print len(label4),len(bbox_target4),len(bbox_weight4)
        # print len(label5),len(bbox_target5),len(bbox_weight5)
        # print label3
        # print label4
        # print label5


        post_nms_topN = len(rois)/4
      #  print post_nms_topN

#########################


        if len(label3) > post_nms_topN:
            label3 = label3[0:-1*(len(label3)-post_nms_topN)]
            rois3 = rois3[0:-1*(len(rois3)-post_nms_topN)]
            bbox_target3 = bbox_target3[0:-1*(len(bbox_target3)-post_nms_topN)]
            bbox_weight3 = bbox_weight3[0:-1*(len(bbox_weight3)-post_nms_topN)]
        #    print len(label3),len(rois3),len(bbox_target3),len(bbox_weight3)

    
        if len(label4) > post_nms_topN:
            label4 = label4[0:-1*(len(label4)-post_nms_topN)]
            rois4 = rois4[0:-1*(len(rois4)-post_nms_topN)]
            bbox_target4 = bbox_target4[0:-1*(len(bbox_target4)-post_nms_topN)]
            bbox_weight4 = bbox_weight4[0:-1*(len(bbox_weight4)-post_nms_topN)]
      #      print len(label4),len(rois4),len(bbox_target4),len(bbox_weight4)
        
        if len(label5) > post_nms_topN:
            label5 = label5[0:-1*(len(label5)-post_nms_topN)]
            rois5 = rois5[0:-1*(len(rois5)-post_nms_topN)]
            bbox_target5 = bbox_target5[0:-1*(len(bbox_target5)-post_nms_topN)]
            bbox_weight5 = bbox_weight5[0:-1*(len(bbox_weight5)-post_nms_topN)]


        
        if len(label6) > post_nms_topN:
            label6 = label6[0:-1*(len(label6)-post_nms_topN)]
            rois6 = rois6[0:-1*(len(rois6)-post_nms_topN)]
            bbox_target6 = bbox_target6[0:-1*(len(bbox_target6)-post_nms_topN)]
            bbox_weight6 = bbox_weight6[0:-1*(len(bbox_weight6)-post_nms_topN)]
        #    print len(label5),len(rois5),len(bbox_target5),len(bbox_weight5)
##################
        if len(label3)==0:
              rois3 = rois[0:2]
              label3 = labels[0:2]
              bbox_target3= bbox_target[0:2]
              bbox_weight3 = bbox_weight[0:2]
        if len(label4)==0:
              rois4 = rois[0:2]
              label4 = labels[0:2]
              bbox_target4= bbox_target[0:2]
              bbox_weight4 = bbox_weight[0:2]
        if len(label5)==0:
              rois5 = rois[0:2]
              label5 = labels[0:2]
              bbox_target5= bbox_target[0:2]
              bbox_weight5 = bbox_weight[0:2]
        if len(label6)==0:
              rois6 = rois[0:2]
              label6 = labels[0:2]
              bbox_target6= bbox_target[0:2]
              bbox_weight6 = bbox_weight[0:2]
        if len(label3) < post_nms_topN:
            keep3= np.where(label3>-100)
            pad = npr.choice(keep3[0], size=post_nms_topN - len(keep3[0]))
            keep_ = np.hstack((keep3[0], pad))
            rois3 = rois3[keep_, :]
            label3 = label3[keep_]
            bbox_target3 = bbox_target3[keep_]
            bbox_weight3 = bbox_weight3[keep_]

        if len(label4) < post_nms_topN:
            keep4= np.where(label4>-100)
            pad = npr.choice(keep4[0], size=post_nms_topN - len(keep4[0]))
            keep_ = np.hstack((keep4[0], pad))
            rois4 = rois4[keep_, :]
            label4 = label4[keep_]
            bbox_target4 = bbox_target4[keep_]
            bbox_weight4 = bbox_weight4[keep_]

        if len(label5) < post_nms_topN:
            keep5= np.where(label5>-100)
            pad = npr.choice(keep5[0], size=post_nms_topN - len(keep5[0]))
            keep_ = np.hstack((keep5[0], pad))
            rois5 = rois5[keep_, :]
            label5 = label5[keep_]
            bbox_target5 = bbox_target5[keep_]
            bbox_weight5 = bbox_weight5[keep_]
        if len(label6) < post_nms_topN:
            keep6= np.where(label6>-100)
            pad = npr.choice(keep6[0], size=post_nms_topN - len(keep6[0]))
            keep_ = np.hstack((keep6[0], pad))
            rois6 = rois6[keep_, :]
            label6 = label6[keep_]
            bbox_target6 = bbox_target6[keep_]
            bbox_weight6 = bbox_weight6[keep_]

        for ind, val in enumerate([rois3, label3, bbox_target3, bbox_weight3,rois4, label4, bbox_target4, bbox_weight4,rois5, label5, bbox_target5, bbox_weight5,rois6, label6, bbox_target6, bbox_weight6]):
            self.assign(out_data[ind], req[ind], val)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)
        self.assign(in_grad[2], req[2], 0)
        self.assign(in_grad[3], req[3], 0)
    


@mx.operator.register('assign_rois')
class AssignRoisProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(AssignRoisProp, self).__init__(need_top_grad=False)


    def list_arguments(self):
        return ['rois','label','bbox_target','bbox_weight']

    def list_outputs(self):
        return ['rois3','label3','bbox_target3','bbox_weight3','rois4','label4','bbox_target4','bbox_weight4','rois5','label5','bbox_target5','bbox_weight5','rois6','label6','bbox_target6','bbox_weight6']

    def infer_shape(self, in_shape):
        rois_shape = in_shape[0]
        label_shape = in_shape[1]
        bbox_target_shape = in_shape[2]
        bbox_weight_shape = in_shape[3]
        rois3_shape = [rois_shape[0]/4,rois_shape[1]]
    

        label3_shape = [label_shape[0]/4,]
        bbox_weight3_shape = [bbox_weight_shape[0]/4,bbox_weight_shape[1]]
        bbox_target3_shape = [bbox_target_shape[0]/4,bbox_target_shape[1]]
        return [rois_shape,label_shape,bbox_target_shape,bbox_weight_shape], \
               [rois3_shape,label3_shape,bbox_target3_shape,bbox_weight3_shape,rois3_shape,label3_shape,bbox_target3_shape,bbox_weight3_shape,rois3_shape,label3_shape,bbox_target3_shape,bbox_weight3_shape,rois3_shape,label3_shape,bbox_target3_shape,bbox_weight3_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return AssignRoisOperator()

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
