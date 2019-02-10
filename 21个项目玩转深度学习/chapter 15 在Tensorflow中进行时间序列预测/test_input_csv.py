#coding:utf-8
from __future__ import print_function #python2,3兼容print函数
import tensorflow as tf

csv_file_name = './data/period_trend.csv'
reader = tf.contrib.timeseries.CSVReader(csv_file_name) #读取csv文件

#读取数据
with tf.Session() as sess:
    data = reader.read_full()
    coord = tf.train.Coordinator() #创建管理线程的类实例
    threads = tf.train.start_queue_runners(sess=sess,coord=coord) #创建执行任务的线程
    print(sess.run(data)) #要创建线程后，才能执行此任务
    coord.request_stop() #执行完任务后，停止线程

#创建batch
train_input_fn = tf.contrib.timeserise.RandomWindowInputFn(reader,batch_size=4,window_size=16)

with tf.Session() as sess:
    data = train_input_fn.creat_batch() #创建batch
    coord = tf.train.Coordinator() #创建管理线程的类实例
    threads = tf.train.start_queue_runners(sess=sess,coord=coord) #创建之星任务的线程
    batch1 = sess.run(data[0])
    batch2 = sess.run(data[0])
    coord.request_stop() #任务结束后，结束线程

print('batch1:',batch1)
print('batch2:',batch2)
