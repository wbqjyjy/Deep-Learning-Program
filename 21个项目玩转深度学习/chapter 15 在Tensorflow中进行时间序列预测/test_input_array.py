#读入Numpy的timeseries数据
#coding:utf-8
from __future__ import print_function #兼容python2,3的print
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.timeseries.python.timeseries import NumpyReader

#构造“时间序列数据”
x = np.array(range(1000))
noise = np.random.uniform(-0.2,0.2,1000)
y = np.sin(np.pi * x / 100) + x / 200. + noise
plt.plot(x,y)
plt.savefig('timeserise_y.jpg')

#读取数据
#step1
data = {
    tf.contrib.timeseries.TrainEvalFeatures.TIMES: x,
    tf.contrib.timeseries.TrainEvalFeatrues.VALUES: y,
    }
#上述也可以直接写为：data={times:x,values:y}
#step2
reader = NumpyReader(data)
#step3
with tf.Session()as sess:
    full_data = reader.read_full() #step4
    coord = tf.train.Coordinator() #step5：创建“管理线程”的类实例
    threads = tf.train.start_queue_runners(sess=sess,coord=coord) #step6：启动队列才能正常进行读取
    print(sess.run(full_data)) #不能直接print(sess.run(reader.read_full()))，需要启动线程后才可以
    coord.request_stop() #任务执行结束后，停止线程

#reader即为train_data_set，接下来创建train_batch
#step1
train_input_fn = tf.contrib.timeseries.RandomWindowInputFn(reader,batch_size=2,window_size=10) #batch大小为2条timeseries，一个timeseries共有window_size个时间对
#step2
with tf.Session() as sess:
    batch_data = train_input_fn.creat_batch() #对trainset创建batch
    coord = tf.train.Coordinator() #创建管理线程的类实例
    threads = tf.train.start_queue_runners(sess=sess,coord=coord) #创建线程执行任务
    one_batch =sess.run(batch_data[0]) #利用创建的线程执行任务
    coord.request_stop() #执行完任务要记得关掉线程

print('one_batch_data:',one_batch)


















