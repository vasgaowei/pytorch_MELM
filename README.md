# pytorch_MLEM

This is the pytorch implementation for the paper《Min-Entropy Latent Model for Weakly Supervised Object Detection》,which is a accepted paper in [CVPR2018](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wan_Min-Entropy_Latent_Model_CVPR_2018_paper.pdf) and [TPAMI](https://ieeexplore.ieee.org/document/8640243). The MLEM is a novel and solid object detection framework under weakly supervised settings and attains state-of-the art performance in PASCAL VOC 2007 and PASCAL VOC 2012 datasets.

This implementation is based [Winfrand's](https://github.com/Winfrand/MELM) torch7 version.
# Prerequisites
* Nivdia GPU 1080Ti
* Ubuntu 16.04 LTS
* pytorch **0.4** is required and we will update a new version for pytorch **1.0** soon. 
* tensorflow **1.0**, tensorboard and [tensorboardX](https://github.com/lanpa/tensorboardX) for visualizing training and    validation curve.

# Installation
1. Clone the repository
   '''
   git clone https://github.com/vasgaowei/pytorch_MELM.git
   '''
