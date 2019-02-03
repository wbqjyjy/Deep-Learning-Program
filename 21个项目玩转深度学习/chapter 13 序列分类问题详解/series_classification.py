# coding: utf-8
from __future__ import print_function #使python3,2兼容
import tensorflow as tf
import random
import numpy as np

#ToySequenceData()用于产生序列样本
class ToySequenceData(object):
    """ 生成序列数据。每个数量可能具有不同的长度。
    一共生成下面两类数据
    - 类别 0: 线性序列 (如 [0, 1, 2, 3,...])
    - 类别 1: 完全随机的序列 (i.e. [1, 3, 10, 7,...])
    注意:
    max_seq_len是最大的序列长度。对于长度小于这个数值的序列，我们将会补0。
    在送入RNN计算时，会借助sequence_length这个属性来进行相应长度的计算。
    """
    def __init__(self,n_samples=1000,max_seq_len=20,min_seq_len=3,max_value=1000):
        #n_samples：样本量；max_seq_len：最大序列长度；min_seq_len：最小序列长度；max_value：序列value最大值
        self.data = [] #序列数据
        self.labels = [] #数据标签[1,0]
        self.seqlen = [] #序列真实长度
        for i in range(n_smaples):
            #序列的长度是随机的，在min_seq_len和max_seq_len之间
            len = random.randint(min_seq_len,max_seq_len) #随机生成序列长度
            self.seqlen.append(len) #将序列长度添加到seqlen
            if random.random() < .5: #以50%概率，随机添加一个线性或随机的训练
                #生成一个线性序列
                rand_start = random.randint(0,max_value-len)
                s = [[float(i)/max_value] for i in range(rand_start,rand_start + len)]
                s += [[0.] for i in range(max_seq_len - len)] #长度不足max_seq_len补[0.]
                self.data.append(s)
                self.labels.append([1.,0.])
            else:
                s = [[float(random.randint(0,max_value))/max_value] for i in range(len)] #随机序列
                s += [[0.] for i in range(max_seq_len - len)]
                self.data.append(s) #将生成序列加入data中
                self.labels.append([0.,1.])
         self.batch_id = 0 #???

    def next(self,batch_size):
        """ 生成batch_size的样本。
            如果使用完了所有样本，会重新从头开始"""
        if self.batch_id == len(self.data):
            self.batch_id = 0

        batch_data = self.data[self.batch_id:min(self.batch_id + batch_size,len(self.data))] #批数据
        batch_labels = self.labels[self.batch_id:min(self.batch_id + batch_size, len(self.data))] #批label
        batch_seqlen = self.seqlen[self.batch_id:min(self.batch_id + batch_size, len(self.data))]

        self.batch_id = min(self.batch_id +batch_size, len(self.data))

        return batch_data,batch_labels,batch_seqlen

#运行的参数
learning_rate = 0.01
training_iters = 1000000
batch_size = 128
display_step = 10

#网络定义时的参数
seq_max_len = 20 #最大序列长度
n_hidden = 64 #隐层大小
n_classes = 2 #序列类别数

trainset = ToySequenceData(n_samples = 1000, max_seq_len = seq_max_len) #训练集
testset = ToySequenceData(n_samples = 500, max_seq_len = seq_max_len) #测试集

x = tf.placeholder("float",[None,seq_max_len,1]) #输入序列
y = tf.placeholder("float",[None,n_classes]) #输出类别
seqlen = tf.placeholder(tf.int32,[None]) #序列实际长度

#weights和bias在输出时会用到
weights = {
    'out':tf.Variable(tf.random_normal([n_hidden,n_classes]))
    }
biases = {
    'out':tf.Variable(tf.random_normal([n_classes]))
    }

def dynamicRNN(x,seqlen,weights,biases):
    #生成RNN单元
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
    #进行时间维度扩展
    outputs,states = tf.nn.dynamic_rnn(lstm_cell,x,dtype=tf.float32,sequence_length=seqlen)
    #生成output_batch中，各个序列的最终output_index
    batch_size = tf.shape(outputs)[0]
    index = tf.range(0,batch_size) * seq_max_len +(seqlen - 1)
    outputs = tf.gather(tf.reshape(outputs,[-1,n_hidden]),index) #产出各个output
    #形成output logits
    return tf.matmul(outputs,weights['out']) + bias['out']

#开始训练
pred = dynamicRNN(x,seqlen,weights,biases) #形成logits

#生成cost
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
#创建优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

#分类准确率
correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1)) #tf.equal(x,y)返回的是True or Flase about x=？y
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

#初始化
init = tf.global_variables_initializer()

#训练
with tf.Session() as sess:
    sess.run(init) #全局初始化
    step=1
    while step * batch_size < training_iters:
        batch_x,batch_y,batch_seqlen = trainset.next(batch_size) #产出x,y,seqlen
        sess.run(optimizer,feed_dict = {x:batch_x,y:batch_y,seqlen:batch_seqlen}) #进行优化
        if step % display_step == 0:
            acc = sess.run(accuracy,feed_dict = {x:batch_x,y:batch_y,seqlen:batch_seqlen})
            loss = sess.run(cost,feed_dict ={x:batch_x,y:batch_y,seqlen:batch_seqlen})
            print("Iter" + str(step*batch_size) + ",Minibatch Loss= " +\
                  "{:.6f}".format(loss) + ",Training Accuracy= " +\
                  "{:.5f}".format(acc))
        step += 1
    print("optimization finished!")

    #在测试集上计算一次精度
    test_data = testset.data
    test_label = testset.labels
    test_seqlen = testset.seqlen
    print("Testing Accuracy:",\
          sess.run(accuracy,feed_dict={x:test_data,y:test_label,seqlen:test_seqlen}))


         
                
        
