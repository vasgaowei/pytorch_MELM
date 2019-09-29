# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.visualization import draw_bounding_boxes

from ops.roi_pool import RoIPool
from ops.roi_align import RoIAlign
from ops.roi_ring_pool import RoIRingPool

from model.config import cfg
from utils.bbox import bbox_overlaps
import tensorboardX as tb

from scipy.misc import imresize

class Network(nn.Module):
  def __init__(self):
    nn.Module.__init__(self)
    self._predictions = {}
    self._losses = {}
    self._layers = {}
    self._gt_image = None
    self._event_summaries = {}
    self._image_gt_summaries = {}
    self._device = 'cuda'
    self.RoIPool = RoIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1./16)
    self.RoIAlign = RoIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1./16)
    self.RoIRingPool = RoIRingPool(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1./16, 0., 1.0)
    self.RoIRingPool_context = RoIRingPool(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1./16, 1.0, 1.8)
    self.RoIRingPool_frame = RoIRingPool(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1./16, 1.0/1.8, 1.0)
    

  def _add_gt_image(self):
    # add back mean
    image = self._image_gt_summaries['image'] + cfg.PIXEL_MEANS
    image = imresize(image[0], self._im_info[:2] / self._im_info[2])
    # BGR to RGB (opencv uses BGR)
    self._gt_image = image[np.newaxis, :,:,::-1].copy(order='C')

  def _add_gt_image_summary(self):
    # use a customized visualization function to visualize the boxes
    self._add_gt_image()
    image = draw_bounding_boxes(\
                      self._gt_image, self._image_gt_summaries['gt_boxes'], self._image_gt_summaries['im_info'])

    return tb.summary.image('GROUND_TRUTH', image[0].astype('float32').swapaxes(1,0).swapaxes(2,0)/255.0)

  def _add_losses(self, sigma_rpn=3.0):
    det_cls_prob = self._predictions['det_cls_prob']
    det_cls_prob = det_cls_prob.view(-1)
    label = self._image_level_label.view(-1)
    
    det_cls_product = self._predictions['det_cls_prob_product']
    
    refine_prob_1 = self._predictions['refine_prob_1']
    refine_prob_2 = self._predictions['refine_prob_2']
    #refine_prob_3 = self._predictions['refine_prob_3']
    
    
    #caculating the loss of the first branch

    roi_labels, roi_weights ,keep_inds = get_refine_supervision(det_cls_product, self._image_gt_summaries['ss_boxes'][self.ss_boxes_indexes, :],
                                                                     self._image_gt_summaries['image_level_label'])

   
    
    roi_weights = torch.tensor(roi_weights).cuda()
    roi_labels = torch.tensor(roi_labels, dtype=roi_weights.dtype).cuda()  
    refine_loss_1 = - torch.sum(torch.mul(roi_labels, torch.log(refine_prob_1[keep_inds]))) / roi_labels.shape[0]
    
    #caculating the loss of the second branch
    roi_labels, roi_weights, keep_inds = get_refine_supervision(refine_prob_1, self._image_gt_summaries['ss_boxes'][self.ss_boxes_indexes, :],
                                                                     self._image_gt_summaries['image_level_label'])
    
    roi_weights = torch.tensor(roi_weights).cuda()
    roi_labels = torch.tensor(roi_labels, dtype=roi_weights.dtype).cuda()
    refine_loss_2 = - torch.sum(torch.mul(roi_labels, torch.log(refine_prob_2[keep_inds]))) / roi_labels.shape[0]
    
    
    #roi_labels, roi_weights, keep_inds = self.get_refine_supervision(refine_prob_2, self._image_gt_summaries['ss_boxes'],
    #                                                                 self._image_gt_summaries['image_level_label'])
    
    #roi_weights = torch.tensor(roi_weights).cuda()
    #roi_labels = torch.tensor(roi_labels, dtype=roi_weights.dtype).cuda()
    #roi_labels = torch.mul(roi_labels, roi_weights)
    #refine_loss_3 = - torch.sum(torch.mul(roi_labels, torch.log(refine_prob_3[keep_inds]))) / roi_labels.shape[0]
    
    self._losses['refine_loss_1'] = refine_loss_1
    self._losses['refine_loss_2'] = refine_loss_2
    #self._losses['refine_loss_3'] = refine_loss_3
    
    label = torch.tensor(label, dtype=det_cls_prob.dtype, device=det_cls_prob.device)
    zeros = torch.zeros(det_cls_prob.shape, dtype=det_cls_prob.dtype, device=det_cls_prob.device)
    max_zeros = torch.max(zeros, 1-torch.mul(label, det_cls_prob))
    cls_det_loss = torch.sum(max_zeros)
    self._losses['cls_det_loss'] = cls_det_loss / 20
    
    loss = cls_det_loss / 20 + refine_loss_1*0.1 + refine_loss_2*0.1
    self._losses['total_loss'] = loss
    for k in self._losses.keys():
      self._event_summaries[k] = self._losses[k]

    return loss

  def _region_classification(self, fc7_roi, fc7_context, fc7_frame):
    alpha = cfg.TRAIN.MIL_RECURRECT_WEIGHT
    
    
    refine_score_1 = self.refine_net_1(fc7_roi)
    refine_score_2 = self.refine_net_2(fc7_roi)
    #refine_score_3 = self.refine_net_3(fc7)
    
  
    cls_score = self.cls_score_net(fc7_roi)
    context_score = self.det_score_net(fc7_context)
    frame_score = self.det_score_net(fc7_frame)
    det_score = frame_score - context_score
    
    cls_prob = F.softmax(cls_score, dim=1)   #num x class_num
    det_prob = F.softmax(det_score, dim=0)   #num x class_num
    
    refine_prob_1 = F.softmax(refine_score_1, dim=1)  #num x class_num+1
    refine_prob_2 = F.softmax(refine_score_2, dim=1)  #num x class_num+1
    #refine_prob_3 = F.softmax(refine_score_3, dim=1)  #num x class_num+1
    
    det_cls_prob_product = torch.mul(cls_score, det_prob)  #num x class_num
    det_cls_prob = torch.sum(det_cls_prob_product, 0) #1 x class_num or just a one dim vector whose size is class_num
    # bbox_pred = self.bbox_pred_net(fc7)
    bbox_pred = torch.zeros(cls_prob.shape[0], 80)
    
    self._predictions['refine_prob_1'] = refine_prob_1
    self._predictions['refine_prob_2'] = refine_prob_2
    #self._predictions['refine_prob_3'] = refine_prob_3
    
    self._predictions["bbox_pred"] = bbox_pred
    
    self._predictions['det_cls_prob_product'] = det_cls_prob_product
    self._predictions['det_cls_prob'] = det_cls_prob

    return cls_prob, det_prob, bbox_pred, det_cls_prob_product, det_cls_prob

  def _image_to_head(self):
    raise NotImplementedError

  def _head_to_tail(self, pool5):
    raise NotImplementedError

  def create_architecture(self, num_classes, tag=None,
                          anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
    self._tag = tag

    self._num_classes = num_classes
    self._anchor_scales = anchor_scales
    self._num_scales = len(anchor_scales)

    self._anchor_ratios = anchor_ratios
    self._num_ratios = len(anchor_ratios)

    self._num_anchors = self._num_scales * self._num_ratios

    assert tag != None

    # Initialize layers
    self._init_modules()

  def _init_modules(self):
    self._init_head_tail()
    self.cls_score_net = nn.Linear(self._fc7_channels, self._num_classes)
    self.det_score_net = nn.Linear(self._fc7_channels, self._num_classes)
    self.bbox_pred_net = nn.Linear(self._fc7_channels, self._num_classes * 4)
    
    self.refine_net_1 = nn.Linear(self._fc7_channels, self._num_classes + 1)
    self.refine_net_2 = nn.Linear(self._fc7_channels, self._num_classes + 1)
    #self.refine_net_3 = nn.Linear(self._fc7_channels, self._num_classes + 1)
    
    self.init_weights()

  def _run_summary_op(self, val=False):
    """
    Run the summary operator: feed the placeholders with corresponding newtork outputs(activations)
    """
    summaries = []
    # Add image gt
    summaries.append(self._add_gt_image_summary())
    # Add event_summaries
    for key, var in self._event_summaries.items():                   #__event_summaries is equal to loss itmes
      summaries.append(tb.summary.scalar(key, var.item()))
    self._event_summaries = {}
    return summaries

  def _predict(self, ss_boxes):
    # This is just _build_network in tf-faster-rcnn
    torch.backends.cudnn.benchmark = False
    net_conv = self._image_to_head()   
    ss_rois  = torch.from_numpy(ss_boxes).to(self._device)
    rois = ss_rois
    self._predictions["rois"] = rois
    pool5_roi = self.RoIRingPool(net_conv, rois)
    pool5_context = self.RoIRingPool_context(net_conv, rois)
    pool5_frame = self.RoIRingPool_frame(net_conv, rois)

    if self._mode == 'TRAIN':
      torch.backends.cudnn.benchmark = True # benchmark because now the input size are fixed
    fc7_roi = self._head_to_tail(pool5_roi)
    fc7_context = self._head_to_tail(pool5_context)
    fc7_frame = self._head_to_tail(pool5_frame)
    
    cls_prob, det_prob, bbox_pred ,cls_det_prob_product ,det_cls_prob = self._region_classification(fc7_roi, fc7_context, fc7_frame)
    
      
    return rois, cls_prob, det_prob, bbox_pred, cls_det_prob_product, det_cls_prob

  def forward(self, image, image_level_label ,im_info, gt_boxes=None, ss_boxes=None, mode='TRAIN'):
    #print('forward ss_boxes ', ss_boxes.shape)
    self._image_gt_summaries['image'] = image
    self._image_gt_summaries['image_level_label'] = image_level_label
    self._image_gt_summaries['gt_boxes'] = gt_boxes
    self._image_gt_summaries['im_info'] = im_info
    self._image_gt_summaries['ss_boxes'] = ss_boxes

    self._image = torch.from_numpy(image.transpose([0,3,1,2]).copy()).to(self._device)
    self._image_level_label = torch.from_numpy(image_level_label) if image_level_label is not None else None
    self._im_info = im_info # No need to change; actually it can be an list
    self._gt_boxes = torch.from_numpy(gt_boxes).to(self._device) if gt_boxes is not None else None

    self._mode = mode
    
    self.ss_boxes_indexes = self.return_ss_boxes(np.arange(ss_boxes.shape[0]), mode)
    rois, cls_prob, det_prob, bbox_pred ,cls_det_prob_product ,det_cls_prob = self._predict(ss_boxes[self.ss_boxes_indexes, :])

    
    bbox_pred = bbox_pred[:,:80]
    
    if mode == 'TEST':
      stds = bbox_pred.data.new(cfg.TRAIN.BBOX_NORMALIZE_STDS).repeat(self._num_classes).unsqueeze(0).expand_as(bbox_pred)
      means = bbox_pred.data.new(cfg.TRAIN.BBOX_NORMALIZE_MEANS).repeat(self._num_classes).unsqueeze(0).expand_as(bbox_pred)
      self._predictions["bbox_pred"] = bbox_pred.mul(stds).add(means)
    else:
      self._add_losses() # compute losses

      
  def return_ss_boxes(self, boxes_index, mode='TRAIN'):
        if mode == 'TEST':
            return boxes_index
        box_num = min(1000, len(boxes_index))
        indexes = np.random.choice(boxes_index, size=box_num, replace=False)
        return indexes
    
  def init_weights(self):
    def normal_init(m, mean, stddev, truncated=False):
      """
      weight initalizer: truncated normal and random normal.
      """
      # x is a parameter
      if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
      else:
        m.weight.data.normal_(mean, stddev)
      m.bias.data.zero_()
      
    #normal_init(self.rpn_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
    #normal_init(self.rpn_cls_score_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
    #normal_init(self.rpn_bbox_pred_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
    normal_init(self.cls_score_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
    normal_init(self.det_score_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
    normal_init(self.bbox_pred_net, 0, 0.001, cfg.TRAIN.TRUNCATED)
    normal_init(self.refine_net_1, 0, 0.01, cfg.TRAIN.TRUNCATED)
    normal_init(self.refine_net_2, 0, 0.01, cfg.TRAIN.TRUNCATED)
    #normal_init(self.refine_net_3, 0, 0.01, cfg.TRAIN.TRUNCATED)
    
    
  # Extract the head feature maps, for example for vgg16 it is conv5_3
  # only useful during testing mode
  def extract_head(self, image):
    feat = self._layers["head"](torch.from_numpy(image.transpose([0,3,1,2])).to(self._device))
    return feat

  # only useful during testing mode
  def test_image(self, image, im_info, ss_boxes):
    self.eval()
    with torch.no_grad():
      self.forward(image, None, im_info, None, ss_boxes, mode='TEST')
    bbox_pred, rois , det_cls_prob, det_cls_prob_product, refine_prob_1, refine_prob_2 = self._predictions['bbox_pred'].data.cpu().numpy(), \
                                                     self._predictions['rois'].data.cpu().numpy(), \
                                                     self._predictions['det_cls_prob'].data.cpu().numpy(), \
                                                     self._predictions['det_cls_prob_product'].data.cpu().numpy(),\
                                                     self._predictions['refine_prob_1'].data.cpu().numpy(), \
                                                     self._predictions['refine_prob_2'].data.cpu().numpy(), \
                                                    
                                                     
                                                     
                                                     
    return bbox_pred, rois, det_cls_prob, det_cls_prob_product, refine_prob_1[:,1:], refine_prob_2[:,1:]
  def delete_intermediate_states(self):
    # Delete intermediate result to save memory
    for d in [self._losses, self._predictions]:
      for k in list(d):
        del d[k]

  def get_summary(self, blobs):
    self.eval()
    self.forward(blobs['data'], blobs['image_level_labels'], blobs['im_info'], blobs['gt_boxes'], blobs['ss_boxes'])
    self.train()
    summary = self._run_summary_op(True)

    return summary

  def train_step(self, blobs, train_op):
    self.forward(blobs['data'], blobs['image_level_labels'], blobs['im_info'], blobs['gt_boxes'], blobs['ss_boxes'])
    cls_det_loss, refine_loss_1, refine_loss_2, loss = self._losses['cls_det_loss'].item(), \
                                      self._losses['refine_loss_1'].item(),  \
                                      self._losses['refine_loss_2'].item(),  \
                                      self._losses['total_loss'].item(), \
                                      
                                    
                                                                        
    train_op.zero_grad()
    self._losses['total_loss'].backward()
    train_op.step()

    self.delete_intermediate_states()

    return cls_det_loss, refine_loss_1, refine_loss_2, loss

  def train_step_with_summary(self, blobs, train_op):
    self.forward(blobs['data'], blobs['image_level_labels'],blobs['im_info'], blobs['gt_boxes'], blobs['ss_boxes'])
    cls_det_loss, refine_loss_1, refine_loss_2, loss = self._losses["cls_det_loss"].item(), \
                                                       self._losses['refine_loss_1'].item(), \
                                                       self._losses['refine_loss_2'].item(), \
                                                       self._losses['total_loss'].item()
    train_op.zero_grad()
    self._losses['total_loss'].backward()
    train_op.step()
    summary = self._run_summary_op()

    self.delete_intermediate_states()

    return cls_det_loss, refine_loss_1, refine_loss_2, loss, summary

  def train_step_no_return(self, blobs, train_op):
    self.forward(blobs['data'], blobs['im_info'], blobs['gt_boxes'], blobs['ss_boxes'])
    train_op.zero_grad()
    self._losses['total_loss'].backward()
    train_op.step()
    self.delete_intermediate_states()

  def load_state_dict(self, state_dict):
    """
    Because we remove the definition of fc layer in resnet now, it will fail when loading 
    the model trained before.
    To provide back compatibility, we overwrite the load_state_dict
    """
    nn.Module.load_state_dict(self, {k: state_dict[k] for k in list(self.state_dict())})

def get_refine_supervision(refine_prob, ss_boxes, image_level_label):
      '''
      refine_prob: num_box x 20 or num_box x 21
      ss_boxes; num_box x 4
      image_level_label: 1 dim vector with 20 elements
      '''
      
      cls_prob = refine_prob.data.cpu().numpy()
      
      
      if refine_prob.shape[1] == image_level_label.shape[1] + 1:
          cls_prob = cls_prob[:, 1:]
      roi_labels = np.zeros([refine_prob.shape[0], image_level_label.shape[1] + 1], dtype = np.int32)  # num_box x 21
      roi_labels[:,0] = 1                                                                        # the 0th elements is the bg
      roi_weights = np.zeros((refine_prob.shape[0], 1), dtype=np.float32)     # num_box x 1 weights of the rois
      
      max_score_box = np.zeros((0, 4), dtype = np.float32)
      max_box_score = np.zeros((0, 1), dtype = np.float32)
      max_box_classes = np.zeros((0, 1), dtype = np.int32)
      
      #print('ss_boxes ', ss_boxes[:5,:])
      for i in range(image_level_label.shape[1]):
          if image_level_label[0, i] == 1:
              cls_prob_tmp = cls_prob[:, i]
              max_index = np.argmax(cls_prob_tmp)
              
              max_score_box = np.concatenate((max_score_box, ss_boxes[max_index, 1:].reshape(1, -1)), axis=0)
              max_box_classes = np.concatenate((max_box_classes, (i+1)*np.ones((1, 1), dtype=np.int32)), axis=0)
              max_box_score = np.concatenate((max_box_score, cls_prob_tmp[max_index]*np.ones((1, 1), dtype=np.float32)), axis=0)
      overlaps = bbox_overlaps(ss_boxes[:,1:], max_score_box)
      gt_assignment = overlaps.argmax(axis=1)
      max_over_laps = overlaps.max(axis=1)
      roi_weights[:, 0] = max_box_score[gt_assignment, 0]
      labels = max_box_classes[gt_assignment, 0]
      
      fg_inds = np.where(max_over_laps > cfg.TRAIN.MIL_FG_THRESH)[0]
      
      roi_labels[fg_inds,labels[fg_inds]] = 1
      roi_labels[fg_inds, 0] = 0
      
      bg_inds = (np.array(max_over_laps >= cfg.TRAIN.MIL_BG_THRESH_LO, dtype=np.int32) + \
                 np.array(max_over_laps < cfg.TRAIN.MIL_BG_THRESH_HI, dtype=np.int32)==2).nonzero()[0]
      
      if len(fg_inds) > 0 and len(bg_inds) > 0:
          fg_rois_num = min(cfg.TRAIN.MIL_NUM_FG, len(fg_inds))
          fg_inds = fg_inds[np.random.choice(np.arange(0, len(fg_inds)), size=int(fg_rois_num), replace=False)]
          
          bg_rois_num = min(cfg.TRAIN.MIL_NUM_BG, len(bg_inds))
          bg_inds = bg_inds[np.random.choice(np.arange(0, len(bg_inds)), size=int(bg_rois_num), replace=False)]
      
      elif len(fg_inds) > 0:
          fg_rois_num = min(cfg.TRAIN.MIL_NUM_FG, len(fg_inds))
          fg_inds = fg_inds[np.random.choice(np.arange(0, len(fg_inds)), size=int(fg_rois_num), replace=False)]
      elif len(bg_inds) > 0:
          bg_rois_num = min(cfg.TRAIN.MIL_NUM_BG, len(bg_inds))
          bg_inds = bg_inds[np.random.choice(np.arange(0, len(bg_inds)), size=int(bg_rois_num), replace=False)]
      else:
          import pdb
          pdb.set_trace()
      
      # print(len(fg_inds), len(bg_inds))
      keep_inds = np.concatenate([fg_inds, bg_inds])
      
      return roi_labels[keep_inds, :], roi_weights[keep_inds,0].reshape(-1,1), keep_inds