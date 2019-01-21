"""Validate a face recognizer on the "Labeled Faces in the Wild" dataset (http://vis-www.cs.umass.edu/lfw/).
Embeddings are calculated using the pairs from http://vis-www.cs.umass.edu/lfw/pairs.txt and the ROC curve
is calculated and plotted. Both the model metagraph and the model parameters need to exist
in the same directory, and the metagraph should have the extension '.meta'.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import lfw
import os
import sys
import math
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('lfw_dir', type=str,
        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('--lfw_batch_size', type=int,
        help='Number of images to process in a batch in the LFW test set.', default=100)
    parser.add_argument('model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--lfw_pairs', type=str,
        help='The file containing the pairs to use for validation.', default='data/pairs.txt')
    parser.add_argument('--lfw_file_ext', type=str,
        help='The file extension for the LFW dataset.', default='png', choices=['jpg', 'png'])
    parser.add_argument('--lfw_nrof_folds', type=int,
        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    return parser.parse_args(argv) #返回的是namespace {model = '...'}

def main(args):
    with tf.Graph().as_default(): #默认的计算流图
        with tf.Session() as sess:
            # Read the file containing the pairs used for testing
            pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs)) #将含有~或~user的directory作为用户目录

            # Get the paths for the corresponding images
            paths,actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir),pairs,args.lfw_file_ext)

            # Load the model
            facenet.load_model(args.model)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0") #embeddings是什么东东？？？
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            image_size=args.image_size
            embedding_size = embeddings.get_shape()[1] #embedding是什么东东？？？

            # Run forward pass to calculate embeddings
            print("running forward pass on LFW images")
            batch_size = args.lfw_batch_size
            nrof_images = len(paths) #图片总数
            nrof_batches = int(math.ceil(1.0 * nrof_images / batch_size)) #batch数量
            emb_array = np.zeros((nrof_images,embedding_size)) #???
            for i in range(nrof_batches):
                start_index = i * batch_size
                end_index = min((i+1) * batch_size,nrof_images)
                paths_batch = paths[start_index:end_index] #路径batch
                images =facenet.load_data(paths_batch,False,False,image_size) #下载图片
                feed_dict = {images_placeholder:images , phase_train_placeholder:False} #phase_train_placeholder???
                emb_array[start_index:end_index,:] = sess.run(embeddings,feed_dict=feed_dict) #embeddings为最终分类？？？

            tpr,fpr,accuracy,val,val_std,far = lfw.evaluate(emb_array,actual_issame,nrof_folds=args.lfw_nrof_folds) #actual_issame：实际的Image_name??? 为什么需要交叉验证？？？不是直接用lfw数据集验证model的准确率吗？？？
            print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy),np.std(accuracy)))#为什么accuracy还存在mean，std???
            print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val,val_std,far)) #far是什么意思

            auc = metrics.auc(fpr,tpr) #AUC曲线
            print('Area Under Curve (AUC): %1.3f' % auc)
            eer = brentq(lambda x:1. -x - interpolate.interp1d(fpr,tpr)(x),0.,1.) #brentq()标量函数求根；自变量区间为[0,1]，给定function，求f(x) = 0的解；公式f()含义？？？
            print('Equal Error Rate(EER): %1.3f' % eer) #interpolate.interp1d();定义函数  x:横坐标列表 y:纵坐标列表 kind:插值方式
            
if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

            









                

        












            
