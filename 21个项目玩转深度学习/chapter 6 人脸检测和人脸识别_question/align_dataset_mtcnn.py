"""Performs face alignment and stores face thumbnails in the output directory."""

from __future__ import absolute_import #绝对引入，引入系统自带的.py文件
from __future__ import division #导入精确除法
from __future__ import print_function #兼容Python3 和 python2的print

from scipy import misc
import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import facenet #Functions for building the face recognition network.
import algin.detect_face #Tensorflow implementation of the face detection / alignment algorithm found at
#https://github.com/kpzhang93/MTCNN_face_detection_alignment
import random
from time import sleep

def parse_arguments(argv):
    parser = argparse.ArgumentParser() #创建实例

    parser.add_argument('input_dir',type=str,help='Directory with unaligned images.')#增加命令行；位置参数
    parser.add_argument('output_dir',type=str,help='Directory with aligned face thumbnails.')
    parser.add_argument('--image_size',type=int,help='Image size (height, width) in pixels.', default=182) #可选参数
    parser.add_argument('--margin', type=int,
                        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--random_order',
                        help='Shuffles the order of images to enable alignment using multiple processes.', action='store_true')
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    return parser.parse_args(argv) #为命令行赋值，并返回namespace(即：各个命令行的值）

def main(args):
    sleep(random.random()) #？？？
    output_dir = os.path.expanduser(args.output_dir) #把path中包含的"~"和"~user"转换成用户目录；用户目录？？？
    if not os.path.exists(output_dir):
        os.makedirs(output_dir) #如果指定目录不存在，则创建该目录

    src_path,_ = os.path.split(os.path.realpath(__file__)) #os.path.realpath()返回Path的真实路径(获取当前执行脚本的绝对路径）；os.path.split()把路径分为dirname和basename
    facenet.store_revision_info(src_path,output_dir,' '.join(sys.argv)) #???
    dataset = facenet.get_dataset(args.input_dir) #获取数据

    print('creating networks and loading parameters')

    with tf.Graph().as_default(): #创建默认计算图
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction= args.gpu_memory_fraction) #来设置tensorflow使用的GPU显存
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,log_device_placement=False)) #tf.ConfigProto()配置设备使用情况
        with sess.as_default(): #将sess设为默认会话，即使会话结束，仍然可以利用sess.run()
            pnet,rnet,onet = align.detect_face.create_mtcnn(sess,None) #??? 返回3中网络

    minsize = 20 #minimum size of face
    threshold = [0.6,0.7,0.7] #3种网络的限值
    factor = 0.709 #scale factor

    #Add a random key to the filename to allow alignment using multiple processes???
    random_key = np.random.randint(0,high=99999)
    bounding_boxes_filename = os.path.join(output_dir,'bounding_boxes_%05d.txt' % random_key)

    with open(bounding_boxes_filename,'w') as text_file: #向text_file中写入
        nrof_images_total = 0 #???
        nrof_successfully_aligned = 0 #???
        if args.random_order:
            random.shuffle(dataset) #打乱dataset中的数据顺序
        for cls in dataset:
            output_class_dir = os.path.join(output_dir,cls.name) #cls.name是文件夹的名称
            if not os.path.exists(output_class_dir):
                os.makedirs(output_class_dir) #如果没有该路径，创建
                if args.random_order:
                    random.shuffle(cls.image_paths) #打乱文件夹中各文件的顺序
            for image_path in cls.image_paths:
                nrof_images_total += 1 #图片数量
                filename = os.path.splitext(os.path.split(image_path)[1])[0] #splitext()是将文件路径和扩展名分开； split()是将文件路径和文件名分开；Image_path中，directory是input，不是output
                output_filename = os.path.join(output_class_dir,filename+'.png') #一张照片的完整路径
                print(image_path)#打印各个文件夹cls中，各个文件路径（图像？？？）
                if not os.path.exists(output_filename):
                    try:
                        img = misc.imread(image_path) #将读入图像转为array形式
                    except (IOError,ValueError,IndexError) as e: #如果try失败，引发错误，否则执行else
                        errorMessage = '{}:{}'.format(image_path,e)
                        print(errorMessage)
                    else:
                        if img.ndim < 2:
                            print('unable to align "%s"' % image_path)
                            text_file.write('%s\n' % (output_filename))
                            continue #继续下一个循环
                        if img.ndim == 2:
                            img = facenet.to_rgb(img) #将image格式转为RGB
                        img = img[:,:,0:3] #RGB

                        bounding_boxes,_ = align.detect_face.detect_face(img,minsize,pnet,rnet,onet,threshold,factor) #最终bounding_boxes
                        nrof_faces = bounding_boxes.shape[0] #???
                        if nrof_faces > 0:
                            det = bounding_boxes[:,0:4] #第一维是什么？？？
                            img_size = np.asarray(img.shape)[0:2] #各个维度的shape
                            if nrof_faces > 1:
                                bounding_box_size = (det[:,2] - det[:,0]) * (det[:,3] - det[:,1]) #提取框面积
                                img_center = img_size / 2 #img的中心点
                                offsets = np.vstack([(det[:,0] + det[:,2]) /2 - img_center[1],(det[:,1] + det[:,3]) /2 - img_center[0]])#提取框中心点相对img中心点的偏移量
                                offset_dist_squared = np.sum(np.power(offsets,2.0),0) #偏移距离
                                index = np.argmax(bounding_box_size -offset_dist_squared * 2.0) #选出bounding_box_size中最大的那个value对应的index
                                det = det[index,:]
                            det = np.squeeze(det)
                            bb = np.zeros(4,dtype =np.int32)
                            bb[0] = np.maximum(det[0] - args.margin/2,0)
                            bb[1] = np.maximum(det[1] - args.margin/2,0)
                            bb[2] = np.maximum(det[2] + args.margin/2,0)
                            bb[3] = np.maximum(det[3] + args.margin/2,0) #将得到人脸框缩小margin个像素

                            cropped = img[bb[1]:bb[3],bb[0]:bb[2],:] #截取提取框中image
                            scaled = misc.imresize(cropped,(args.image_size,args.image_size),interp='bilinear')
                            nrof_successfully_aligned +=1#成功检测对齐的人脸个数
                            misc.imsave(output_filename,scaled)
                            text_file.write('%s %d %d %d %d\n' % (output_filename,bb[0],bb[1],bb[2],bb[3]))
                        else:
                            print('unable to align "%s"' % image_path)
                            text_file.write('%s\n' % (output_filename))
    print('total number of images: %d' % nrof_images_total)
    print('numberof successfully aligned images: %d' % nrof_successfully_aligned)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

















        
                                
                                
                
             
            
















    
    
    
