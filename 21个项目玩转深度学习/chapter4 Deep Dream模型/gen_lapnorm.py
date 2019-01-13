#coding:utf-8

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
model_fn = 'tensorflow_inception_graph.pb'
with tf.gfile.FastGFile('model_fn','rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

t_input = tf.placeholder(np.float32,name='input')
imagenet_mean = 117.0
t_preprocessed = tf.expand_dims(t_input - imagenet_mean,0)

tf.import_graph_def(graph_def,{'input':t_preprocessed})

#保存图像
def savearray(img_array,img_name):
    scipy.misc.toimage(img_array).save(img_name)
    print('img saved: %s' % img_name)

#对图像进行缩放
def resize_ratio(img,ratio):
    min = img.min()
    max = img.max()
    img = (img - min)/(max - min) *255 #由于scipy.misc.imresize()是在255的基础上进行缩放，所以要对img进行预处理，使其在255范围内
    img = np.float32(scipy.misc.imresize(img,ratio)
    img = img/255 *(max-min) + min
    return img

#求gradient
def calc_grad_tiled(img,t_grad,tile_size=512):
    sz = tile_size
    h,w = img.shape[:2]
    sx,sy = np.random.randint(sz,size=2)
    #为避免边缘效应，对img进行偏移
    img_shift = np.roll(np.roll(img,sx,1),sy,0)
    #初始化一个gradient数组
    grad = np.zeros_like(img)
    for y in range(0,max(h-sz//2,sz),sz):
        for x in range(0,max(w-sz//2,sz),sz):
            sub = img_shift[y:y+sy,x:x+sx]
            g = sess.run([t_grad],{t_input:sub})
            grad[y:y+sy,x:x+sx] = g
    return np.roll(np.roll(grad,-sx,1),-sy,0)

k = np.float32([1,4,6,4,1]) #???
k = np.outer(k,k) # k.shape = (5,5)
k5x5 = k[:,:,None,None]/k.sum() * np.eye(3,dtype=np.float32) #k[:,:,None,None].shape =(5,5,1,1)
#question??? 为什么 (5,5,1,1) * (3,3) = (5,5,3,3)

#将图像分为低频 和 高频
def lap_split(img):
    with tf.name_scope('split'): #可以让变量具有相同的命名
        #求img的低频分量
        lo = tf.nn.conv2d(img,k5x5,[1,2,2,1],'SAME') # k5x5 ???
        #将lo缩放为原img大小
        lo2 = tf.nn.conv2d_transpose(lo,k5x5 * 4,tf.shape(img),[1,2,2,1]) # k5x5 * 4???
        hi = img - lo2
    return lo,hi

#将图像img分成n层拉普拉斯金字塔
def lap_split_n(img,n):
    levels = []
    for i in range(n):
        img,hi = lap_split(img)
        levels.append(hi)
    levels.append(img)
    return levels[::-1]

#将拉普拉斯金字塔还原到原始图像
def lap_merge(levels):
    img = levels[0]
    for hi in levels[1:]:
        with tf.name_scope('merge'):
            img = tf.nn.conv2d_transpose(img,k5x5 * 4,tf.shape(hi),[1,2,2,1]) + hi
    return img

#对Img做标准化
def normalize_std(img,eps=1e-10):
    with tf.name_scope('normalize'):
        std = tf.sqrt(tf.reduce_mean(tf.square(img)))
        return img / tf.maximum(std,eps)

#拉普拉斯金字塔标准化
def lap_normalize(img,scale_n =4):
    img = tf.expand_dims(img,0)
    tlevels = lap_split_n(img,scale_n) #将img分解为scale_n层金字塔
    #对每一层都做一次normalize
    tlevels = list(map(normalize_std,tlevels))
    out = lap_merge(tlevels) #融合
    return out[0,:,:,:] #返回融合后的图形

#使进行normalize时，输入，输出均为array形式
#下面函数用到partial(function,arg)
def tffunc(*argtypes):
    placeholders = list(map(tf.placeholder,argtypes)) #定义占位符的数据类型
    def wrap(f):
        out = f(*placeholders) #function f中的placeholder为上述定义的占位符,*arg 说明可以是多个占位符，在lap_normalize()中，其中一个给scale_n，另外一个给img
        def wrapper(*args,**kw):
            return out.eval(dict(zip(placeholders,args)),session = kw.get('session')) #out为tensor ; 返回为tensor out计算结果
        return wrapper #返回的wrapper为ndarray，因为eval()返回的是ndarray
    return wrap

#渲染img
def render_lapnorm(t_obj,img0,iter_n=10,step=1.0,octave_n=3,octave_scale=1.4,lap_n=4):
    t_score = tf.reduce_mean(t_obj)
    t_grad = tf.gradients(t_score,t_input)[0]
    lap_norm_func = tffunc(np.float32)(partial(lap_normalize,scale_n=lap_n)) #说明wrapper()中没有参数，eval()中的参数均采用默认值

    img = img0.copy()
    for octave in range(octave_n):
        if octave > 0:
            img = resize_ratio(img,octave_scale)
        for i in range(iter_n):
            g = calc_grad_tiled(img,t_grad)
            g = lap_norm_func(g) #对于每一个缩放尺寸的img,把gradient分解为拉普拉斯图像，然后在进行融合；而在gen_deepdream.py中，是把分成拉普拉斯图像的img（低频img),先求其gradient，然后得到img += g*step，然后将低频img缩放加上下一层的高频img，通过这种方式来柔和img；可以看出，gen_lapnorm.py是通过一次性将gradient分解为拉普拉斯图像然后融合的方法，去柔和img。而gen_deepdream.py则是通过将img分解为拉普拉斯图像，然后求每一层img的gradient，img += g*step，然后将其缩放，在于下一层高频img进行相加，通过这种方法柔和img
            img += g * step
            print('.',end=' ')
    savearray(img,'lapnorm.jpg')

if __name__ == '__main__':
    name = 'mixed4d_3x3_bottleneck_pre_relu'
    channel = 139
    img_noise = np.random.uniform(size=(224,224,3)) + 100.0
    layer_output = graph.get_tensor_by_name("import/%s:0"%name)
    render_lapnorm(layer_output[:,:,:,channel],img_noise,iter_n=20)


    
                     





                     
    
    
        
    











    
