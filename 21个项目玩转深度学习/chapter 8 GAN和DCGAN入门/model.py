from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils import *

def conv_out_size_name(size,stride): #？？？
    return int(math.ceil(float(size)/float(stride))) #math.ceil()返回大于参数的最小整数

class DCGAN(object):
    def __init__(self,sess,input_height=108,input_width=108,crop=True,
                 batch_size=64,sample_num=64,output_height=64,output_width=64,
                 y_dim=None,z_dim=100,gf_dim=64,df_dim=64,
                 gfc_dim=1024,dfc_dim=1024,c_dim=3,dataset_name='default',
                 input_fname_pattern='*.jpg',checkpoint_dir=None,sample_dir=None):
        """
    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
      y_dim: (optional) Dimension of dim for y. [None]
      z_dim: (optional) Dimension of dim for Z. [100]
      gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
      df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
      gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
      dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
      c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
    """
        self.sess = sess
        self.crop = crop

        self.batch_size = batch_size
        self.sample_num = sample_num

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.y_dim = y_dim
        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')#d_bn1为判别网络discriminator的layer1的batch normalization
        self.d_bn2 = batch_norm(name='d_bn2')

        if not self.y_dim:
            self.d_bn3 = batch_norm(name='d_bn3') #如果有label值的话，对layer3也进行batch normalization

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')

        if not self.y_dim:
            self.g_bn3 = batch_norm(name='g_bn3') #不太明白为什么Generator，有y也要batch normalization，最终不是一个图片吗，并没有涉及y啊？？？

        self.dataset_name = dataset_name
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir

        if self.dataset_name == 'mnist':
            self.data_X,self.data_y = self.load_mnist() #下载mnist数据
            self.c_dim = self.data_X[0].shape[-1] #不明白为什么c_dim 等于这个？？？
        else:
            self.data = glob(os.path.join("./data",self.dataset_name,self.input_fname_pattern)) #返回相匹配的文件名或目录
            imreadImg = imread(self.data[0])
            if len(imreadImg.shape) >= 3:
                self.c_dim = imread(self.data[0]).shape[-1]
            else:
                self.c_dim = 1

        self.grayscale = (self.c_dim == 1) #判断是否为彩色图

        self.build_model() #在__init__文件中，建立模型

    def build_model(self):
        if self.y_dim:
            self.y = tf.placeholder(tf.float32,[self.batch_size,self.y_dim],name='y') #如果是带label的训练集，则给label建立一个占位符

        if self.crop:
            image_dims = [self.output_height,self.output_width,self.c_dim]
        else:
            image_dims = [self.input_height,self.input_width,self.c_dim] #根据是否需要对Image进行修剪，来判断输出size

        self.inputs = tf.placeholder(tf.float32,[self.batch_size] + image_dims,name='real_images') #real image input(送到判别网络中的）

        inputs = self.inputs

        self.z = tf.placeholder(tf.float32,[None,self.z_dim],name='z') #noise 送到生成网络的Input

        self.z_sum = histogram_summary('z',self.z) #???

        if self.y_dim:
            self.G = self.generator(self.z,self.y) #如果有label，则生成网络的建立，需要同时输入z，y
            self.D,self.D_logits = self.discriminator(inputs,self.y,reuse=False) #这里，z,y,input均为占位符  self.G,self.D,self.D_logits分别代表什么？？？

            self.sampler = self.sampler(self.z,self.y) #self.sampler代表什么？？？
            self.D_,self.D_logists_ = self.discriminator(self.G,self.y,reuse=True) #self.D_，self.D_logists_分别代表什么
        else:
            self.G = self.generator(self.z) #生成的图片？？？
            self.D,self.D_logits = self.discriminator(inputs)#？？？

            self.sampler = self.sampler(self.z) #？？？
            self.D_,self.D_logits_ = self.discriminator(self.G,reuse=True)

        self.d_sum = histogram_summary("d",self.D) #???
        self.d__sum = histogram_summary("d",self.D_) #???
        self.G_sum = image_summary("G",self.G)#???
        #build_model()主要是得出了real,fake图片的概率值，以及self.D？？？ self.G???

        def sigmoid_corss_entropy_with_logits(x,y):
            try:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x,labels=y)
            except:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x,targets=y)

        self.d_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits,tf.ones_like(self.D))) #由于real image的label全为1（is real)，因此，要用tf.ones_like(self.D)将label=1，self.D_logits为概率值
        self.d_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_,tf.zeros_like(self.D_))) #generated_image为假的概率
        self.g_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_,tf.ones_like(self.D_))) #self.D_logits_为生成图片为真对概率；

        self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake #DCGAN的损失函数

        self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
        self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

        t_vars = tf.trainable_variables() #所有需要训练的参数list

        self.d_vars = [var for var in t_vars if 'd_' in var.name] #判别网络参数
        self.g_vars = [var for var in t_vars if 'g_' in var.name] #圣城网络参数

        self.saver = tf.train.Saver() #创建Saver的类实例

    def train(self,config):
        d_optim = tf.train.AdamOptimizer(config.learning_rate,betal=config.beta1).minimize(self.d_loss,var_list = self.d_vars) #优化判别网络
        g_optim = tf.train.AdamOptimizer(config.learning_rate,beta1=config.beta1).minimize(self.g_loss,var_list = self.g_vars) #优化生成网络
        try:
            tf.global_variables_initializer().run() #初始化全局变量
        except:
            tf.initializer_all_variables().run()

        self.g_sum = merge_summary([self.z_sum, self.d__sum,
          self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = merge_summary(
          [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = SummaryWriter("./logs", self.sess.graph) #把信息写到tensorboard上

        sample_z = np.random.uniform(-1,1,size=(self.sample_num,self.z_dim)) #???

        if config.dataset == 'mnist':
            sample_inputs = self.data_X[0:self.sample_num]
            sample_labels = self.data_y[0:self.sample_num]
        else:
            sample_files = self.data[0:self.sample_num]
            sample = [
                get_image(sample_file,
                    input_height=self.input_height,
                    input_width=self.input_width,
                    resize_height=self.output_height,
                    resize_width=self.output_width,
                    crop=self.crop,
                    grayscale=self.grayscale) for sample_file in sample_files]
           if (self.grayscale):#????不太明白为什么要进行如下操作
               sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
           else:
               sample_inputs = np.array(sample).astype(np.float32)

            counter = 1
            start_time = time.time()
            could_load,checkpoint_counter = self.load(self.checkpoint_dir)#下载model？？？，self.load()是神马？？？
            if could_load:
                counter = checkpoint_counter #counter存的是DCGAN网络？？？
                print("[*] Load SUCCESS")
            else:
                print("[!] Load failed...")

            for epoch in xrange(config.epoch):
                if config.dataset == 'mnist':
                    batch_idxs = min(len(self.data_X),config.train_size) #有几个batch
                else:
                    self.data = glob(os.path.join("./data",config.dataset,self.input_fname_pattern)) #取出所有相关文件名
                    batch_idxs = min(len(self.data),config.train_size)

                for idx in xrange(0,batch_idxs):
                    if config.dataset == 'mnist':
                        batch_images = self.data_X[idx*config.batch_size:(idx+1)*config.batch_size]#每批次取出的数据
                        batch_labels = self.data_y[idx*config.batch_size:(idx+1)*config.batch_size]
                    else:
                        batch_files = self.data[idx*config.batch_size:(idx+1)*config.batch_size]
                        batch = [
                            get_image(batch_file,
                                input_height=self.input_height,
                                input_width=self.input_width,
                                resize_height=self.output_height,
                                resize_width=self.output_width,
                                crop=self.crop,
                                grayscale=self.grayscale) for batch_file in batch_files]
                       if self.grayscale:
                           batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
                       else:
                           batch_images = np.array(batch).astype(np.float32)

                    batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
                        .astype(np.float32)#随机产生的噪声，数量与batchsize一致

                    if config.dataset == 'mnist':
                        #update D network
                        _,summary_str = self.sess.run([d_optim,self.d_sum], #网络结构在discriminator()中已经给出
                            feed_dict= {
                                self.inputs:batch_images,
                                self.z:batch_z,
                                self.y:batch_labels,
                                })
                        self.writer.add_summary(sumamry_str,counter)#将一些信息写入tensorboard。counter是什么？？？

                        #update G network
                        _,summary_str = self.sess.run([g_optim,self.g_sum],
                            feed_dict={
                                self.z:batch_z,
                                self.y:batch_labels,
                                })#不太明白为什么圣城网络还有label？？？
                        self.writer.add_summary(summary_str,counter)

                        # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                        _, summary_str = self.sess.run([g_optim, self.g_sum],
                            feed_dict={ self.z: batch_z, self.y:batch_labels })
                        self.writer.add_summary(summary_str, counter)

                        errD_fake = self.d_loss_fake.eval({ #tf.reduce_mean().eval()计算平均值
                            self.z:batch_z,
                            self.y:batch_labels
                            }) #self.d_loss_fake.eval()自定义函数？？？ 相当于是计算 判别网络中的 fake_loss
                        errD_real = self.d_loss_real.eval({
                            self.inputs:batch_images,
                            self.y:batch_labels
                            })
                        errG = self.g_loss.eval({
                            self.z:batch_z,
                            self.y:batch_labels
                            })
                    else:
                        # Update D network
                        _, summary_str = self.sess.run([d_optim, self.d_sum],
                            feed_dict={ self.inputs: batch_images, self.z: batch_z })
                        self.writer.add_summary(summary_str, counter)

                        # Update G network
                        _, summary_str = self.sess.run([g_optim, self.g_sum],
                            feed_dict={ self.z: batch_z })
                        self.writer.add_summary(summary_str, counter)

                        # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                        _, summary_str = self.sess.run([g_optim, self.g_sum],
                            feed_dict={ self.z: batch_z })
                        self.writer.add_summary(summary_str, counter)
          
                        errD_fake = self.d_loss_fake.eval({ self.z: batch_z })
                        errD_real = self.d_loss_real.eval({ self.inputs: batch_images })
                        errG = self.g_loss.eval({self.z: batch_z})

                    counter += 1
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f,d_loss:%.8f,g_loss:%.8f" % (epoch,idx,batch_idxs,time.time() -start_time,errD_fake+errD_real,errG))

                    if np.mod(counter, 100) == 1:#当epoch达到100时：
                        if config.dataset == 'mnist':
                            samples, d_loss, g_loss = self.sess.run(  #求 生成的sample??? ,d_loss，g_loss，为什么前边用的是eval()求，现在又改为sess.run()
                              [self.sampler, self.d_loss, self.g_loss],
                               feed_dict={
                                self.z: sample_z,
                                 self.inputs: sample_inputs,
                                  self.y:sample_labels,
                                 }
                                )
                            save_images(samples, image_manifold_size(samples.shape[0]),
                              './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
                            print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 
                        else:
                            try:
                                samples, d_loss, g_loss = self.sess.run(
                                  [self.sampler, self.d_loss, self.g_loss],
                                  feed_dict={
                                  self.z: sample_z,
                                  self.inputs: sample_inputs,
                                     },
                                     )
                                save_images(samples, image_manifold_size(samples.shape[0]),
                                   './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
                                print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 
                           except:
                                print("one pic error!...")

                 if np.mod(counter, 500) == 2:
                    self.save(config.checkpoint_dir, counter) #self.save()是自定义函数吗？？？ 这里保存的是model及参数？？？为什么用counter???counter是原来保存的模型及参数？？？但是，self.save()怎么和saver.save()联系起来？？？
                    
    def discriminator(self,image,y=None,reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables() #如果reuse=True，重用判别网络中的参数
            if not self.y_dim:
                h0 = lrelu(conv2d(image,self.df_dim,name='d_h0_conv')) #filter为要求的参数
                h1 = lrelu(self.d_bn1(conv2d(h0,self.df_dim*2,name='d_h1_conv'))) #在输入layer1的activation function之前，先进行batch normalization,防止梯度消失现象的发生
                h2 = lrelu(self.d_bn2(conv2d(h1,self.df_dim*4,name='d_h2_conv')))
                h3 = lrelu(self.d_bn3(conv2d(h2,self.df_dim*8,name='d_h3_conv')))
                h4 = linear(tf.reshape(h3,[self.batch_size,-1]),1,'d_h4_lin') #将H4 dimension调整为样本数量

                return tf.nn.sigmoid(h4),h4 #前者为probability，后者为net output???
            else:
                yb = tf.reshape(y,[self.batch_size,1,1,self.y_dim])
                x = conv_cond_concat(image,yb) #包含有y的imput???

                h0 = lrelu(conv2d(x,self.c_dim + self.y_dim,name='d_h0_conv'))#不太理解这个维度到底是怎么整的？？？
                h0 = conv_cond_concat(h0,yb)

                h1 = lrelu(self.d_bn1(conv2d(h0,self.df_dim + self.y_dim, name='d_h1_conv')))
                h1 = tf.reshape(h1,[self.batch_size,-1])
                h1 = concat([h1,y],1)

                h2 = lrelu(self.d_bn2(linear(h1,self.dfc_dim,'d_h2_lin')))
                h2 = concat([h2,y],1)

                h3 = linear(h2,1,'d_h3_lin')

                return tf.nn.sigmoid(h3),h3


    def generator(self,z,y=None):
        with tf.variable_scope("generator") as scope:
            if not self.y_dim:
                s_h,s_w = self.output_height,self.output_width
                s_h2,s_w2 = conv_out_size_same(s_h,2),conv_out_size_same(s_w,2)
                s_h4,s_w4 = conv_out_size_same(s_h2,2),conv_out_size_same(s_w2,2)
                s_h8,s_w8 = conv_out_size_same(s_h4,2),conv_out_size_same(s_w4,2)
                s_h16,s_w16 = conv_out_size_same(s_h8,2),conv_out_size_same(s_w8,2)

                #project z and reshape
                self.z_,self.h0_w,self.h0_b = linear(z,self.gf_dim*8*s_h16*s_w16,'g_h0_lin',with_w=True)
                self.h0 = tf.reshape(self.z_,[-1,s_h16,s_w16,self.gf_dim*8])
                h0=tf.nn.relu(self.g_bn0(self.h0)) #先batch normalization,在input

                self.h1,self.h1_w,self.h1_b = deconv2d(h0,[self.batch_size,s_h8,s_w8,self.gf_dim*4],name='g_h1',with_w=True)
                h1 = tf.nn.relu(self.g_bn1(self.h1)) #第一层反卷积，在relu之前，先要对卷积结果 batch norm

                h2,self.h2_w,self.h2_b = deconv2d(h1,[self.batch_size,s_h4,s_w4,self.gf_dim*2],name='g_h2',with_w=True)
                h2=tf.nn.relu(self.g_bn2(h2))

                h3, self.h3_w, self.h3_b = deconv2d(
                    h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3', with_w=True)
                h3 = tf.nn.relu(self.g_bn3(h3))

                h4, self.h4_w, self.h4_b = deconv2d(
                    h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)

                return tf.nn.tanh(h4) #output是个image，但是Tanh在-1,1之间，为了成为图像 ，要 （x+1)*127.5
            else:
                s_h,s_w = self.output_height,self.output_width
                s_h2,s_h4 = int(s_h/2),int(s_h/4)
                s_w2,s_w4 = int(s_w/2),int(s_w/4)

                # yb = tf.expand_dims(tf.expand_dims(y,1),2)
                yb = tf.reshape(y,[self.batch_size,1,1,self.y_dim])
                z = concat([z,y],1)

                h0 = tf.nn.relu(self.g_bn0(linear(z,self.gfc_dim,'g_h0_lin')))
                h0 = concat([h0,y],1)

                h1 = tf.nn.relu(self.g_bn1(linear(h0,self.gf_dim*2*s_h4*s_w4,'g_h1_lin')))
                h1 = tf.reshape(h1,[self.batch_size,s_h4,s_w4,self.gf_dim*2])

                h1 = conv_cond_concat(h1,yb)

                h2 = tf.nn.relu(self.g_bn2(deconv2d(h1,[self.batch_size,s_h2,s_w2,self.gf_dim*2],name='g_h2')))
                h2 = conv_cond_concat(h2,yb)

                return tf.nn.sigmoid(deconv2d(h2,[self.batch_size,s_h,s_w,self.c_dim],name='g_h3'))

    def sampler(self,z,y=None):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables() #加载生成器的参数

            if not self.y_dim:
                s_h,s_w = self.output_height,self.output_width
                s_h2,s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
                s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
                s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
                s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

                # project `z` and reshape
                h0 = tf.reshape(
                    linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin'),
                    [-1, s_h16, s_w16, self.gf_dim * 8])
                h0 = tf.nn.relu(self.g_bn0(h0, train=False))

                h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1')
                h1 = tf.nn.relu(self.g_bn1(h1, train=False))

                h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2')
                h2 = tf.nn.relu(self.g_bn2(h2, train=False))

                h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3')
                h3 = tf.nn.relu(self.g_bn3(h3, train=False))

                h4 = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4')

                return tf.nn.tanh(h4) #有一个疑问，加载入参数后，con2d等等操作，都没有显示的说要套用参数值，就可以被直接套进去吗？是的，因为，所有反卷积操作均是在tf.variable_scope("generator")形成。
            else:
                s_h, s_w = self.output_height, self.output_width
                s_h2, s_h4 = int(s_h/2), int(s_h/4)
                s_w2, s_w4 = int(s_w/2), int(s_w/4)

                # yb = tf.reshape(y, [-1, 1, 1, self.y_dim])
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                z = concat([z, y], 1)

                h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin'), train=False))
                h0 = concat([h0, y], 1)

                h1 = tf.nn.relu(self.g_bn1(
                    linear(h0, self.gf_dim*2*s_h4*s_w4, 'g_h1_lin'), train=False))
                h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
                h1 = conv_cond_concat(h1, yb)

                h2 = tf.nn.relu(self.g_bn2(
                    deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2'), train=False))
                h2 = conv_cond_concat(h2, yb)

                return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))

    def load_mnist(self):
        data_dir = os.path.join("./data", self.dataset_name)
    
        fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)

        fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        trY = loaded[8:].reshape((60000)).astype(np.float)

        fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)

        fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.float)

        trY = np.asarray(trY)
        teY = np.asarray(teY)
    
        X = np.concatenate((trX, teX), axis=0)
        y = np.concatenate((trY, teY), axis=0).astype(np.int)
    
        seed = 547
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(y)
    
        y_vec = np.zeros((len(y), self.y_dim), dtype=np.float)
        for i, label in enumerate(y):
            y_vec[i,y[i]] = 1.0 #？？？
    
        return X/255.,y_vec #所有的label=1 ??? 假设y为0-9的数，y_dim=10，y_vec.shape=(N,dimension) ,y_vec[i,y[i]]意思是，第i个行的第y[i]列=1，其它等于0

@property
  def model_dir(self):
    return "{}_{}_{}_{}".format(
        self.dataset_name, self.batch_size,
        self.output_height, self.output_width)
      
  def save(self, checkpoint_dir, step):
    model_name = "DCGAN.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step) #保存model参数信息

  def load(self, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name)) #将原来的model参数加载进来
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0)) #counter是表示一些时间信息，epoch信息吗？？？
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0





                

                








                        
                        


        






















                                          
