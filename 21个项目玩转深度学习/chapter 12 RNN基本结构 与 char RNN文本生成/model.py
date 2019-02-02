# coding: utf-8

from __futur=__ import print_function
import tensorflow as tf
import numpy as np
import time
import os

def pick_top_n(preds,vocab_size,top_n=5):
    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n]] = 0 #将除了top_n个预测值的位置都置为0
    p = p/np.sum(p) #归一化概率
    c = np.random.choice(vocab_size,1,p=p)[0] #随机选取一个字符
    return c

class CharRNN:
    def __init__(self,num_classes,num_seqs=64,num_steps=50,lstm_size=128,num_layers=2,learning_rate=0.001,grad_clip=5,sampling=False,train_keep_prob=0.5,use_embedding=False,embedding_size=128):
        if sampling is True:
            num_seqs, num_steps = 1,1
        else:
            num_seqs, num_steps = num_seqs, num_steps

        self.num_classes = num_classes #类别数
        self.num_seqs = num_seqs #batch size
        self.num_steps = num_steps #序列长度
        self.lstm_size = lstm_size #隐含单元h大小
        self.num_layers = num_layers #RNN叠加层数
        self.learning_rate = learning_rate #学习率
        self.grad_clip = grad_clip #???
        self.train_keep_prob = train_keep_prob #dropout 概率
        self.use_embedding = use_embedding #是否使用embedding
        self.embedding_size = embedding_size #???

        tf.reset_default_graph()
        self.build_inputs() #构建输入
        self.build_lstm() #构建网络
        self.build_loss() #构建损失
        self.build_optimizer() #构建优化器
        self.saver = tf.train.Saver() #保存实例

    def build_inputs(self):
        with tf.name_scope('inputs'):
            self.inputs = tf.placeholder(tf.int32,shape=(self.num_seqs,self.num_steps),name='inputs')
            self.targets = tf.placehoder(tf.int32,shape=(self.num_seqs,self.num_steps),name='target') #targets与inputs一样，第i个input的输出为第i+1个target；我们的目的是训练RNN模型，我们输入一个字符后能够生成一段话；
            self.keep_prob = tf.placeholder(tf.float32,name='keep_prob')

            #对于中文，需要使用embedding层
            #英文字母没有必要用embedding层
            if self.use_embedding is False:
                self.lstm_inputs = tf.one_hot(self.inputs,self.num_classes) #将Input的字母用one-hot表示，长度为num_classes
            else:
                with tf.device("/cpu:0"):
                    embedding = tf.get_variable('embedding',[self.num_classes,self.embedding_size]) #embedding可以进行训练
                    self.lstm_inputs = tf.nn.embedding_lookup(embedding,self.inputs) #self.inputs是一个序列，如何将各个字母分别输入不同的时间片？？？

    def build_lstm(self):
        #创建单个cell并堆叠多层
        def get_a_cell(lstm_size,keep_prob):
            lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size) #建立单层LSTM RNN
            drop = tf.nn.rnn_cell.DropoutWrapper(lstm,output_keep_prob=keep_prob) #对单层LSTM RNN设立dropout
            return drop

        with tf.name_scope('lstm'):
            cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell(self.lstm_size,self.keep_prob) for _ in range(self.num_layers)]) #将num_layer个RNN Cell进行堆叠
            self.initial_state = cell.zero_state(self.num_seqs,tf.float32) #第一个隐含单元h0的初值

            #通过dynamic_rnn对cell展开时间维度
            self.lstm_outputs,self.final_state = tf.nn.dynamic_rnn(cell,self.lstm_inputs,initial_state=self.initial_state) #以self.lstm_inputs的时间序列长度为依据，展开rnn时间维度；返回的是各个时间片的output和最终的隐含单元（注意：这里output为隐含单元）

            #通过lstm_outputs得到概率
            seq_output = tf.concat(self.lstm_outputs,1) #？？？
            x = tf.reshape(seq_output,[-1,self.lstm_size]) #通过shape变化的Output？？？

            with tf.variable_scope('softmax'):
                softmax_w = tf.Variable(tf.truncated_normal([self.lstm_size,self.num_classes],stddev=0.1)) #w的维数为：(隐含单元长度，类别数）
                softmax_b = tf.Variable(tf.zeros(self.num_classes)) #偏置

            self.logits = tf.matmul(x,softmax_w) + softmax_b
            self.proba_prediction = tf.nn.softmax(self.logits,name='predictions') #各个时间片，output不同类别概率值

    def build_loss(self):
        with tf.name_scope('loss'):
            y_one_hot = tf.one_hot(self.targets,self.num_classes) #进行One-hot向量转变，target为output值（等同于input)
            y_reshaped = tf.reshape(y_one_hot,self.logtis.get_shape()) #???与各个时间片的output shape一致
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logtis,labels=y_reshaped)
            self.loss = tf.reduce_mean(loss)

    def build_optimizer(self):
        #使用clipping gradients
        tvars = tf.trainable_variables() #要进行训练的所有变量
        grads,_ = tf.clip_by_global_norm(tf.gradients(self.loss,tvars),self.grad_clip) #求梯度
        train_op = tf.train.AdamOptimizer(self.learning_rate) #优化器选择
        self.optimizer = train_op.apply_gradients(zip(grads,tvars))

    def train(self,batch_generator,max_steps,save_path,save_every_n,log_every_n):
        self.session = tf.Session()
        with self.session as sess:
            sess.run(tf.global_variables_initializer()) #初始化全局变量
            #train network
            step = 0
            new_state = sess.run(self.initial_state) #h0
            for x,y in batch_generator:
                step += 1
                start = time.time()
                feed = {self.inputs:x,
                        self.targets:y,
                        self.keep_prob:self.train_keep_prob,
                        self.initial_state:new_state}
                batch_loss,new_state,_ = sess.run([self.loss, #损失
                                                   self.final_state, #最终的隐含单元value，将其作为下一个batch的初始隐含单元value
                                                   self.optimizer],feed_dict=feed) #采用的优化器
                end = time.time()
                if step % log_every_n == 0:
                    print('step: {}/{}...'.format(step,max_steps),
                          'loss: {:.4f}...'.format(batch_loss),
                          '{:.4f} sec/batch'.format((end - start)))
                if (step % save_every_n == 0):
                    self.saver.save(sess,os.path.join(save_path,'model'),global_step=step)
                if step >= max_steps:
                    break
            self.saver.save(sess,os.path.join(save_path,'model'),global_step=step)

    def sample(self,n_samples,prime,vocab_size): #在有限序列prime的基础上，在生成n_samples个字符
        samples = [c for c in prime] #最后生成的序列存在sample中
        sess = self.Session
        new_state = sess.run(self.initialize_state)
        preds = np.ones((vocab_size))
        for c in prime: 
            x = np.zeros((1,1)) #x为什么要以这种方式定义？？？
            #输入单个字符
            x[0,0] = c
            feed = {self.inputs: x,
                    self.keep_prob:1, #因为是 test，所以不dropout
                    self.initialize_state:new_state}
            preds , new_state = sess.run([self.proba_prediction,self.final_state],feed_dict = feed)
        
        #prime中的字符都在RNN中遍历以后，利用最后的output probability，从vocab_size中随机选出一个字符
        c = pick_top_n(preds,vocab_size)
        #添加字符到sample中
        samples.append(c)

        #不断生成字符，直到达到指定数目
        for i in range(n_samples):
            x = np.zeros((1,1))
            x[0,0] = c #以生成的字符为下一个input，输出output probability，然后，在随机选取一个字符
            feed = {self.inputs: x,
                    self.keep_prob:1.,
                    self.initial_state:new_state}
            preds, new_state = sess.run([self.proba_prediction, self.final_state],feed_dict = feed)

            c=pick_top_n(preds,vocab_size)
            samples.append(c)

        return np.array(samples)
            
    def load(self,checkpoint):
        self.session = tf.Session()
        self.saver.restore(self.session,checkpoint)
        print('restored from : {}'.format(checkpoint))
            
