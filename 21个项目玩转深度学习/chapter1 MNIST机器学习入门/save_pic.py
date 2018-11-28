#从mnist中读取前20个数据，并将其转化为图片存储

#coding: utf-8
from tensorflow.examples.tutorials.mnist import input_data
import scipy.misc
import os

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

save_dir = 'MNIST_data/raw/'
if os.path.exists(save_dir) is False:
    os.makedirs(save_dir)

for i in range(20):
    image_array = mnist.train.images[i,:]
    image_array = image_array.reshape(28,28)
    filename = save_dir + 'mnist_train_%d.jpg'.%i
    #先用scipy.misc.toimage转换为图像，在调用save直接保存
    scipy.misc.toimage(image_array,cmin=0.0,cmax=1.0).save(filename) 

