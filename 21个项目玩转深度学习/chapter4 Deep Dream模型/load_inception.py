from __future__ import print_function #该语句是为了在python2,python3中可以使用互相兼容的print
import numpy as np
import tensorflow as tf

#创建图和Session
graph = tf.Graph() #生成新的计算图，不同计算图上的张量和运算都不会共享
sess = tf.InteractiveSession(graph=graph) #tf.InteractiveSession()是一种交互式的session方式，它让自己成为了默认的session，也就是说用户在不需要指明用哪个session运行的情况下，就可以运行起来，这就是默认的好处。这样的话就是run()和eval()函数可以不指明session啦。

#tensorflow_inception_graph.pb文件中，即存储了inception的网络结构，也存储了对应的数据
#使用下面语句将其导入
model_fn = 'tensorflow_inception_graph.pb'
with tf.gfile.FastGFile(model_fn,'rb') as f: #tf.gfile.FastGFile()实现对图片的读取
    graph_def = tf.GraphDef() #创建序列化图形
    graph_def.ParseFromString(f.read()) #将Inception模型导入该序列化图形

#定义t_input为我们输入的图像
t_input = tf.placeholder(np.float32,name='input') #创建占位符
imagenet_mean = 117.0 #inception模型中，input都经过去均值处理，因此，在deep dream中的input也要经过去均值处理

#由于Inception模型中input为(batch,height,width,channel)，因此，要将原始输入(height,width,channel)加入一维
t_preprocessed = tf.expand_dims(t_input - imagenet_mean,0) #在第一维插入batch
tf.import_graph_def(graph_def,{'input':t_preprocessed}) #将使用的图像识别模型graph_def，以及，inputimage导入

#找到所有卷积层
layers = [op.name for op in graph.get_operations() if op.type == 'Conv2D' and 'import/' in op.name] #利用graph.get_operations()即可找到计算图中的所有操作op

#输出卷积层层数
print('Number of layers',len(layers))

#特别地，输出mixed4d_3x3_bottleneck_pre_relu的形状
name = 'mixed4d_3x3_bottleneck_pre_relu'
print('shape of %s: %s' % (name,str(graph.get_tensor_by_name('import/' + name + ':0').get_shape())))#graph.get_tensor_by_name(name=the name of tensor to return)得到tensor; tensor.get_shape()返回tensor的形状
# question??? tensor_name中，为什么要加 ':0'？？？

#导入模型过程总结

#step0: 在导入模型前，首先要创建“计算图”和“会话”,使得计算图在给会话下执行
#graph = tf.Graph()
#sess = tf.InteractiveSession(graph=graph)

#step1：将模型导入到序列化的图形中，主要用到：
#tf.gfile.FastGFile() 打开模型
#tf.GraphDef() 创建序列化图形
#graph_def.ParseFromString(f.read()) 将二进制模型解析为code,导入创建的序列化图形

#step2：创建占位符，用于Input image的输入：t_input = tf.placeholder(np.float32,name='input')

#step3：将inception，以及input 导入到创建的graph(计算图）中：tf.import_graph_def()

#step4：根据计算图graph的函数graph.get_operations(),可以找到inception模型的所有layers

#step5： 根据layer名称，可以打印各层layer的tensor shape


