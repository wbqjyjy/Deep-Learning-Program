# coding:utf-8

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
#创建会话
sess = tf.InteractiveSession(graph=graph)
#导入模型inception
model_fn = 'tensorflow_inception_graph.pb'
with tf.gfile.FastGFile(model_fn,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

t_input = tf.placeholder(np.float32,name='input')
imagenet_mean = 117.0
t_preprocessed = tf.expand_dims(t_input - imagenet_mean,0)

#将模型导入计算图中
tf.import_graph_def(graph_def,{'input':t_preprocessed})

#保存图像
def savearray(img_array,img_name):
    scipy.misc.toimage(img_array).save(img_name)
    print('img saved as %s' % img_name)

#进行标准化
def visstd(a,s=0.1):
    return (a-a.mean())/max(a.std(),1e-4) * s + 0.5

#对图像进行缩放
def resize_ratio(img,ratio):
    min = img.min()
    max = img.max()
    img = (img - min)/(max - min) * 255
    img = np.float32(scipy.misc.imresize(img,ratio))
    img = img/255 * (max - min) + min
    return img

#将图像以一定(height,width)进行缩放
def resize(img,hw):
    min = img.min()
    max = img.max()
    img = (img - min)/(max - min) * 255
    img = np.float32(scipy.misc.imresize(img,hw))
    img = img/255 * (max - min) + min
    return img

#计算图像gradient
def calc_grad_tiled(img,t_grad,tile_size=512):
    sz = tile_size
    sx,sy = np.random.randint(sz,size=2)
    img_shift = np.roll(np.roll(img,sx,1),sy,0)
    h,w=img.shape[:2]
    grad = np.zeros_like(img)
    for y in range(0,max(h-sz//2,sz),sz):
        for x in range(0,max(w-sz//2,sz),sz):
            sub = img_shift[y:y+sz,x:x+sz]
            g = sess.run([t_grad],{t_input:sub})
            grad[y:y+sz,x:x+sz] = g
    return np.roll(np.roll(grad,-sx,1),-sy,0)

#将输入，输出都变为ndarray
def tffunc(*argtypes):
    placeholders = list(map(tf.placeholder,argtypes))
    def wrap(f):
        out = f(*placeholders)
        def wrapper(*args,**kw):
            return out.eval(dict(zip(placeholders,args)),session = kw.get('session'))
        return wrapper
    return wrap

#对图像进行渲染
def render_deepdream(t_obj,img0,iter_n=10,step=1.5,octave_n=4,octave_scale=1.4):
    t_score = tf.reduce_mean(t_obj)
    t_grad = tf.gradients(t_score,t_input)[0]

    img = img0

    octaves = []

    for i in range(octave_n -1):
        hw =img.shape[:2]
        lo=resize(img,np.int32(np.float32(hw)/octave_scale)) #低频成分
        hi = img - resize(lo,hw)#高频成分
        img = lo
        octaves.append(hi)


    #先生成低频的图像，在依次放大并加上高频
    for octave in range(octave_n):
        if octave > 0:
            hi =octaves[-octave]
            img = resize(img,hi.shape[:2]) + hi
        for i in range(iter_n):
            g = calc_grad_tiled(img,t_grad)
            img += g*(step/(np.abs(g).mean() + 1e-7)) #学习步长是随时间而变化的，随着时间的推移，g会变得越来越小，因此，step会变得越来越大
            print('.',end=' ')
     img = img.clip(0,255)
     savearray(img,'deepdream.jpg')

if __name__ == '__main__':
    img0 = PIL.Image.open('test.jpg')
    img0 = np.float32(img0)

    name = ‘mixed4d_3x3_bottleneck_pre_relu'
    channel = 139
    layer_output = graph.get_tensor_by_name("import/%s:0" % name)
    render_deepdream(layer_output[:,:,:,channel],img0) #也可使用layer_output[:,:,:,:]使用所有channel的mean，来渲染图形
    
            






















    
















