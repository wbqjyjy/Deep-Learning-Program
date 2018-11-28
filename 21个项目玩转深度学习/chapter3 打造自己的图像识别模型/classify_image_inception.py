#对一个image进行分类

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import re
import sys
import tarfile #对文件进行压缩，解压

import numpy as np
from six.moves import urllib #???
import tensorflow as tf

FLAGS = None

class NodeLookup(object):
    def __init__(self,label_lookup_path=None):
        slef.node_lookup = self.load(label_lookup_path)

    def load(self,label_lookup_path):
        node_id_to_name = {}
        with open(label_lookup_path) as f:
            for index,line in enumerate(f):
                node_id_to_name[index]=line.strip()
        return node_id_to_name #返回每个image的标签信息

    def id_to_string(self,node_id):
        if node_id not in self.node_lookup:
            return ""
        return self.node_lookup[node_id] #返回image的label

def creat_graph():
        """Creates a graph from saved GraphDef file and returns a saver."""
        with tf.gfile.FastGFile(FLAGS.model_path,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read)) #GraphDef加载模型中的图
            _ = tf.import_graph_def(graph_def,name='') #在空白图中（GraphDef)加载模型中的图

def preprocess_for_eval(image,height,width,contral_fraction=0.875,scope=None):
        with tf.name_scope(scope,'eval_image',[image,height,width]):
            if image.dtype != tf.float32:
                image = tf.convert_image_dtype(image,dtype=tf.float32)
            if central_fraction:
                image =tf.image.central_crop(image,central_fraction=central_fraction) #基于图像中心进行裁剪

            if height and width:
                image = tf.expand_dims(image,0) #给image加一个维度，即：在各个维度加一个bias量
                image = tf.image.resize_bilinear(image,[height,widht],align_corners=False) #对图像进行双线性插值，使得image缩放到[height,width]范围内：为什么要先增加Image一个维度，然后在缩减一个维度？？？
                image = tf.squeeze(image,[0]) #去掉image的第1个维度（bias量）
            image = tf.subtract(image,0.5)
            image = tf.multiply(image,2.0)
            return image #image是2维

def run_inference_on_image(image):
    with tf.Graph().as_default():#将这个graph类实例作为运行环境的默认图，覆盖原来的默认图，直到with语句结束
        image_data = gfile.FastGFile(image,'rb').read()
        image_data = tf.image.decode_jpeg(image_data)
        image_data = preprocess_for_eval(image_data,299,299)
        image_data = tf.expand_dims(image_data,0) #个人理解：加入bias量1
        with tf.Session() as sess:
            image_data =sess.run(image_data)

    creat_graph() #这个graphdef存在默认图中

    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('InceptionV3/Logits/SpatialSqueeze:0') #调出默认图中某一变量的tensor，在这里是各个类别logit值对应的结点
        predictions = sess.run(softmax_tensor,{'input:0':image_data}) #以image_data为输入，求tensor： softmax_tensor的值
        predictions = np.squeeze(predictions)#将多余的维度去掉
        
        node_lookup = NodeLookup(FLAGS.label_path)#形成类实例

        top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1] #top_k是前k个prediction的index：将top_k个prediction按从大到小 给出他们的index

        for node_id in top_k:
            human_string = node_lookup.id_to_string(node_id) #人类可识别的label名称
            score = predictions[node_id] #各个label的概率值
            print('%s (score = %.5f)' % (human_string,score))


def main(_): 
    image = FLAGS.image_file
    run_inference_on_image(image)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',type=str,)
    parser.add_argument('--label_patn',type=str,)
    parser.add_argument('--image_file',type=str,default='',help='Absolute path to image file')
    parser.add_argument('--num_top_predictions',type=int,default=5,help='Display this many predictions.'))

    FLAGS,unparsed = parser.parse_known_args()
    tf.app.run(main=main,argv=[sys.argv[0]]+unparsed) #???不应该加argv吧？

    
            






















    





















            
