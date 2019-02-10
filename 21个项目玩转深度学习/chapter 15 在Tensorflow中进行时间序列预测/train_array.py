#coding:utf-8
from __future__ import print_function #python2,3兼容print函数
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.timeseries.python.timeseries import NumpyReader

def main(_):
    #构建数据
    x = np.array(range(1000))
    noise = np.random.uniform(-0.2,0.2,1000)
    y = np.sin(np.pi * x /100) + x / 200. + noise
    plt.plot(x,y)
    plt.savefig('timeseries_y.jpg')

    #读取数据
    data = {
        tf.contrib.timeseries.TrainEvalFeatures.TIMES:x,
        tf.contrib.timeseries.TrainEvalFeatures.VALUES:y,
        }
    reader = NumpyReader(data)
    #创建用于训练的batch
    train_input_fn = tf.contrib.timeseries.RandomWindowInputFn(
        reader,batch_size = 16,window_size=40)

    #利用AR模型，进行时间序列建模
    ar = tf.contrib.timeseries.ARRegressor(
        periodicities=200,input_window_size=30,output_window_size=10,
        num_features=1,
        loss=tf.contrib.timeseries.ARModel.NORMAL_LIKELIHOOD_LOSS) #AR模型的输入为前30个时间对，输出为后10个时间对，二者之和刚好等于train_batch一个时间序列的长度；num_features：表示时间点value的维度；
    #periodicities：表示时间序列的周期

    #进行训练
    ar.train(input_fn=train_input_fn,steps=6000)#训练6000次

    #利用训练数据对训练好的模型进行 校验
    evaluation_input_fn = tf.contrib.timeseries.WholeDatasetInputFn(reader)

    evaluation = ar.evaluate(input_fn=evaluation_input_fn,steps=1) #evaluation有如下关键字[covariance,loss,mean,observed,start_tuple,times,global_step]，其中，evaluation[mean]：预测值（共1000个时间序列对，其中前30个是输入，后970为预测值）；利用evaluation[start_tuple]进行预测（为evaluation的最后30个预测时间对）；evaluation[times]：对应的时间点

    (predictions,) = tuple(ar.predict(
        input_fn=tf.contrib.timeseries.predict_contrinuation_input_fn(
            evaluation,steps=250))) #预测1000个时间序列对后另外250个时间对的值

    #作图
    plt.figure(figsize=(15,5))
    plt.plot(data['times'].reshape(-1),data['values'].reshape(-1),label='origin')
    plt.plot(evaluation['times'].reshape(-1),evaluation['mean'].reshape(-1),label='evaluation')
    plt.plot(predictions['times'].reshape(-1),predictions['mean'].reshape(-1),label='predicion')
    plt.xlabel('time_step')
    plt.ylabel('values')
    plt.legend(loc=4)
    plt.savefig('predict_result.jpg')


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run() #运行main(_)




















    
