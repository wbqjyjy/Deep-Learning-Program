from __future__ import print_function
import os
from io import BytesIO
import numpy as np
from functools import partial
import PIL.Image
import scipy.misc
import tensorflow as tf

#创建计算图
graph = tf.Graph()

#导入图像识别模型inception
model_fn = 'tensorflow_inception_graph.pb'
with tf.gfile.FastGFile('model_fn','rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

#创建会话，用于执行计算图
sess = tf.InteractiveSession(graph=graph)

#创建占位符，放Input
t_input = tf.placeholder(np.float32,name='input')
imagenet_mean = 117.0
t_preprocessed = tf.expand_dims(t_input - imagenet_mean,0)

#将模型导入计算图
tf.import_graph_def(graph_def,{'input':t_preprocessed})

def savearray(img_array,img_name):
    scipy.misc.toimage(img_array).save(img_name)
    print('img saved: %s' % img_name)

#将图像进行缩放？？？
def resize_ratio(img,ratio):
    min = img.min()
    max = img.max()
    img = (img-min)/(max-min) * 255
    img = np.float32(scipy.misc.resize(img,ratio))
    img = img/255 * (max-min) + min
    return img

#计算score关于Img的gradient
def calc_grad_tiled(img,t_grad,tile_size =512):
    sz = tile_size #每次只对sz大小的图像求gradient
    h,w = img.shape[:2]
    #为了避免“边缘效应”，首先要将img进行任意size的roll
    sx,sy = np.random.randint(sz,size=2)
    img_shift = np.roll(np.roll(img,sx,1),sy,0) #np.roll(img,shift,axis)
    #对每个patch进行求导
    grad = np.zeros_like(img) #np.zeros_like(img)输出与img同shape的0 array
    for y in range(0,max(h-sz//2,sz),sz):
        for x in range(0,max(w-sz//2,sz),sz):
            sub =img_shift[y:y+sz,x:x+sz]
            g=sess.run(t_grad,{t_input:sub})#与render_multiscale()中的t_grad相呼应
            grad[y:y+sz,x:x+sz] = g
    return np.roll(np.roll(grad,-sx,1),-sy,0) #返回最后的gradient

#求渲染+放大后的图像
def render_multiscale(t_obj,img0,iter_n=10,step=1.0,octave_n=3,octave_scale=1.4):
    t_score = tf.reduce_mean(t_obj) #求t_obj的mean
    t_grad = tf.gradients(t_score,t_input)[0] ##

    img = img0.copy()
    for octave in range(octave_n):#每次将图片放大octave_scale倍
        if octave > 0:
            img = resize_ratio(img,octave_scale)
            for i in range(iter_n):
                g =calc_grad_tiled(img,t_grad)
                g /= g.std() + 1e-8 #对g进行正规化处理
                img += g*step
                print('.',end=' ')
    savearray(img,'multiscale.jpg')

if __name__ == '__main__': #如果该.py文件为主程序，执行下列操作
    name ='mixed4d_3x3_bottleneck_pre_relu'
    channel = 139
    img_noise = np.random.uniform(size=(224,224,3)) + 100.0
    layer_output = graph.get_tensor_by_name("import/%s:0"% name)
    render_multiscale(layer_output[:,:,:,channel],img_noise,iter_n=20)
    
    
         













    
