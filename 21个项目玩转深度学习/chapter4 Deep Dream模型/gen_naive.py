from __future__ import print_function #兼容python2,python3的print function
import os
from io import BytesIO
import numpy as np
from functools import partial
import PIL.Image
import scipy.misc
import tensorflow as tf

#step1: 创建计算图
graph = tf.Graph()
model_fn = 'tensorflow_inception_graph.pb'
#step2: 创建会话，使得计算图在会话中进行
sess = tf.InteractiveSession(graph=graph)

#将inception模型导入
with tf.gfile.FastGFile(model_fn,'rb') as f: #tf.gfile.FastGFile()打开inception
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read()) #将inception模型存入序列化图形graph_def中

t_input = tf.placeholder(np.float32,name='input') #创建占位符
imagenet_mean =117.0
t_preprocessed = tf.expand_dims(t_input - imagenet_mean,0) #给input在第一维增加一维，存batch

tf.import_graph_def(graph_def,{'input':t_preprocessed}) #将inception导入计算图中

#将input存储为图像
def savearray(img_array,img_name):
    scipy.misc.toimage(img_array).save(img_name)
    print('img saved :%s ' % img_name)

#求img
def render_naive(t_obj,img0,iter_n=20,step=1.0):
    t_score = tf.reduce_mean(t_obj) #t_score为优化目标；tf.reduce_mean()求t_obj的平均值
    t_grad = tf.gradients(t_score,t_input)[0] #求导数 t_score/t_input

    img = img0.copy()
    for i in range(iter_n):
        g,score = sess.run([t_grad,t_score],{t_input:img})#在会话中计算graident，及score
        g /= g.std() + 1e-8 #对gradient进行正规化？？？
        img += g *step
        print('score(mean) = %f' % (score))
    savearray(img,'naive.jpg')

#定义卷积层、通道数，并取出对应的tensor
name = 'mixed4d_3x3_bottleneck_pre_relu'
channel = 139
layer_output = graph.get_tensor_by_name('import/%s:0'% name)#layer的output值

#定义原始Img
img_noise = np.random.uniform(size=(224,224,3)) + 100.0

#调用render_naive函数渲染
render_naive(layer_output[:,:,:,channel],img_noise,iter_n=20)
        

