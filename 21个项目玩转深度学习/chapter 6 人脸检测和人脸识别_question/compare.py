"""Performs face alignment and calculates L2 distance between the embeddings of images."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import argparse
import facenet
import align.detect_face

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('model', type=str,
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('image_files', type=str, nargs='+', help='Images to compare')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
                        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    return parser.parse_args(argv) #返回namespace

def main(args):
    images = load_and_align_data(args.image_files,args.image_size,args.margin,args.gpu_memory_fraction) #???检测对齐后的图像
    with tf.Graph().as_default():
        with tf.Session() as sess:
            model = facenet.load_model(args.model)

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            feed_dict = {image_placeholder:images,phase_train_placeholder:False)#因为不进行训练，所以phase_train_placeholder=Falser  相当于weight

            emb = sess.run(embeddings,feed_dict = feed_dict) #特征

    nrof_images = len(args.image_files)


    print('Images:')
    for i in range(nrof_images):
        print('%1d: %s' % (i,args.image_files[i]))
    print('')

    print('Distance matrix')
    print('    ',end='')
    for i in range(nrof_images):
        print('   %d   '% i,end='')
    print('')#相当于\n的作用，因为没有end=''，所以输出空字符串后会自动换行

    for i in range(nrof_images):
        print('%d  ' % i,end='') #dist接着这个输出
        for j in range(nrof_images):
            dist = np.sqrt(np.sum(np.square(np.subtract(emb[i,:],emb[j,:])))
            print('  %1.4f  ' % dist,end='')
        print('')

def load_and_align_data(image_paths,image_size,margin,gpu_memory_fraction):
    minsize = 20
    threshold = [0.6,0.7,0.7]
    factor = 0.709

    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,log_devicee_placement=False))
        with sess.as_default():
            pnet,rnet,onet = align.detect_face.create_mtcnn(sess,None) #构建MTCNN的3个网络

    nrof_samples = len(image_paths)
    img_list = [None] * nrof_samples
    for i in xrange(nrof_samples):
        img = misc.imread(os.path.expanduser(image_paths[i])) #将image读为array
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes,_ = align.detect_face.detect_face(img,minsize,pnet,rnet,onet,threshold,factor)
        det = np.squeeze(bounding_boxes[0,0:4])
        bb = np.zeros(4,dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin/2,0)
        bb[1] = np.maximum(det[0] - margin/2,0)
        bb[2] = np.minimum(det[2] + margin/2,img_size[1])
        bb[3] = np.minimum(det[3] + margin/2,img_size[0]) #横，纵坐标，height，width
        cropped = img[bb[1]:bb[3],bb[0]:bb[2]] #提取框中图像
        aligned = misc.imresize(cropped,(image_size,image_size),interp='bilinear')
        prewhitened = facenet.prewhiten(aligned) #???
        img_list[i] = prewhitened
    images = np.stack(img_list)
    return images #检测对齐后的图像

if __name__ == '__main__':
    main(parse_arguments(sys.args[1:]))
        
                           

            
    








                                 
