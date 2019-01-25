from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import json
import glob
import random
import collections
import math
import time

parser = argparse.ArgumentParser() #创建类实例
parser.add_argument("--input_dir",help = "path to folder containing images")
parser.add_argument("--mode",required=True,choices=["train","test","export"])
parser.add_argument("--output_dir",required=True,help="where to put output files")
parser.add_argument("--seed",type=int)
parser.add_argument("--checkpoint",default=None,help="directory with checkpoint to resume training from or use for testing")

parser.add_argument("--max_steps",type=int,help="number of training steps(0 to disable)")
parser.add_argument("--max_epochs", type=int, help="number of training epochs")
parser.add_argument("--summary_freq", type=int, default=100, help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
parser.add_argument("--display_freq", type=int, default=0, help="write current training images every display_freq steps")
parser.add_argument("--save_freq", type=int, default=5000, help="save model every save_freq steps, 0 to disable")

parser.add_argument("--aspect_ratio", type=float, default=1.0, help="aspect ratio of output images (width/height)")
parser.add_argument("--lab_colorization", action="store_true", help="split input image into brightness (A) and color (B)")
parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
parser.add_argument("--which_direction", type=str, default="AtoB", choices=["AtoB", "BtoA"])
parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")
parser.add_argument("--scale_size", type=int, default=286, help="scale images to this size before cropping to 256x256")
parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
parser.add_argument("--no_flip", dest="flip", action="store_false", help="don't flip images horizontally")
parser.set_defaults(flip=True)
parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")

parser.add_argument("--output_filetype",default="png",choices=["png","jpeg"]) #添加命令行参数
a = parser.parse_args() #返回namespace

EPS = 1e-12 #???
CROP_SIZE = 256

Examples = collections.namedtuple("Examples","paths,inputs,targets,count,steps_per_epoch") #返回一个名为Examples，包含属性：paths,inputs,targets,count,steps_per_epoch，的类
Model = collections.namedtuple("Model","outputs,predict_real,predict_fake,discrim_loss,discrim_grads_and_vars,gen_loss_GAN,gen_loss_L1,gen_grads_and_vars,train")

def preprocess(image): #???
    with tf.name_scope("preprocess"):
        #[0,1] => [-1,1]
        return image * 2 - 1

def deprocess(image): #？？？
    with tf.name_scope("deprocess"):
        #[-1,1] => [0,1]
        return (image + 1)/2 #image像素 范围

def preprocess_lab(lab): #???
    with tf.name_scope("preprocess_lab"):
        L_chan,a_chan,b_chan = tf.unstack(lab,axis=2)
        #L_chan: black and white with input range [0,100]
        #a_chan / b_chan : color channels with input range ~[-110,110],not exact
        #[0,100] => [-1,1], ~[-110,110] => [-1,1]
        return [L_chan / 50 - 1,a_chan / 110,b_chan /110]

def deprocess_lab(L_chan,a_chan,b_chan):#???
    with tf.name_scope("deprocess_lab"):
        # this is axis=3 instead of axis=2 because we process inidvidual images but deprocess batches
        return tf.stack([(L_chan + 1) /2 * 100,a_chan * 110,b_chan * 110],axis=3)

def augment(image,brightness): #将图像转换成彩色的？？？
    #(a,b)color channels,combinewith L channel and convert to rgb
    a_chan , b_chan = tf.unstack(image,axis=3)
    L_chan = tf.squeeze(brightness,axis=3)
    lab = deprocess_lab(L_chan,a_chan,b_chan)
    rgb = lab_to_rgb(lab)
    return rgb
    
def conv(batch_input,out_channels,stride):
    with tf.variable_scope("conv"):
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter",[4,4,in_channels,out_channels],dtype=tf.float32,initializer = tf.random_normal_initializer(0,0.02))
        # [batch,in_height,in_width,in_channels],[filter_widht,filter_height,in_channels,out_channels]
        # => [batch,out_height,out_width,out_channels]
        padded_input = tf.pad(batch_input,[[0,0],[1,1],[1,1],[0,0]],mode="CONSTANT")
        conv = tf.nn.conv2d(padded_input,filter,[1,stride,stride,1],padding="VALID")
        return conv

def lrelu(x,a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1+a)) * x + (0.5 * (1 - a)) * tf.abs(x)

def batchnorm(input): #batch normalization
    with tf.variable_scope("batchnorm"):
        # this block looks like it has 3 inputs on the graph unless we do this
        input = tf.identity(input)

        channels = input.get_shape()[3]
        offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
        variance_epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
        return normalized

def deconv(batch_input, out_channels):
    with tf.variable_scope("deconv"):
        batch, in_height, in_width, in_channels = [int(d) for d in batch_input.get_shape()]
        filter = tf.get_variable("filter", [4, 4, out_channels, in_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, out_channels, in_channels]
        #     => [batch, out_height, out_width, out_channels]
        conv = tf.nn.conv2d_transpose(batch_input, filter, [batch, in_height * 2, in_width * 2, out_channels], [1, 2, 2, 1], padding="SAME")
        return conv

def check_image(image): #确定image形式 3维
    assertion = tf.assert_equal(tf.shape(image)[-1], 3, message="image must have 3 color channels")
    with tf.control_dependencies([assertion]):
        image = tf.identity(image)

    if image.get_shape().ndims not in (3, 4):
        raise ValueError("image must be either 3 or 4 dimensions")

    # make the last dimension 3 so that you can unstack the colors
    shape = list(image.get_shape())
    shape[-1] = 3
    image.set_shape(shape)
    return image

# based on https://github.com/torch/image/blob/9f65c30167b2048ecbe8b7befdc6b2d6d12baee9/generic/image.c
def rgb_to_lab(srgb): #???
    with tf.name_scope("rgb_to_lab"):
        srgb = check_image(srgb)
        srgb_pixels = tf.reshape(srgb, [-1, 3])

        with tf.name_scope("srgb_to_xyz"):
            linear_mask = tf.cast(srgb_pixels <= 0.04045, dtype=tf.float32)
            exponential_mask = tf.cast(srgb_pixels > 0.04045, dtype=tf.float32)
            rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
            rgb_to_xyz = tf.constant([
                #    X        Y          Z
                [0.412453, 0.212671, 0.019334], # R
                [0.357580, 0.715160, 0.119193], # G
                [0.180423, 0.072169, 0.950227], # B
            ])
            xyz_pixels = tf.matmul(rgb_pixels, rgb_to_xyz)

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("xyz_to_cielab"):
            # convert to fx = f(X/Xn), fy = f(Y/Yn), fz = f(Z/Zn)

            # normalize for D65 white point
            xyz_normalized_pixels = tf.multiply(xyz_pixels, [1/0.950456, 1.0, 1/1.088754])

            epsilon = 6/29
            linear_mask = tf.cast(xyz_normalized_pixels <= (epsilon**3), dtype=tf.float32)
            exponential_mask = tf.cast(xyz_normalized_pixels > (epsilon**3), dtype=tf.float32)
            fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon**2) + 4/29) * linear_mask + (xyz_normalized_pixels ** (1/3)) * exponential_mask

            # convert to lab
            fxfyfz_to_lab = tf.constant([
                #  l       a       b
                [  0.0,  500.0,    0.0], # fx
                [116.0, -500.0,  200.0], # fy
                [  0.0,    0.0, -200.0], # fz
            ])
            lab_pixels = tf.matmul(fxfyfz_pixels, fxfyfz_to_lab) + tf.constant([-16.0, 0.0, 0.0])

        return tf.reshape(lab_pixels, tf.shape(srgb))

def lab_to_rgb(lab):
    with tf.name_scope("lab_to_rgb"):
        lab = check_image(lab)
        lab_pixels = tf.reshape(lab, [-1, 3])

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("cielab_to_xyz"):
            # convert to fxfyfz
            lab_to_fxfyfz = tf.constant([
                #   fx      fy        fz
                [1/116.0, 1/116.0,  1/116.0], # l
                [1/500.0,     0.0,      0.0], # a
                [    0.0,     0.0, -1/200.0], # b
            ])
            fxfyfz_pixels = tf.matmul(lab_pixels + tf.constant([16.0, 0.0, 0.0]), lab_to_fxfyfz)

            # convert to xyz
            epsilon = 6/29
            linear_mask = tf.cast(fxfyfz_pixels <= epsilon, dtype=tf.float32)
            exponential_mask = tf.cast(fxfyfz_pixels > epsilon, dtype=tf.float32)
            xyz_pixels = (3 * epsilon**2 * (fxfyfz_pixels - 4/29)) * linear_mask + (fxfyfz_pixels ** 3) * exponential_mask

            # denormalize for D65 white point
            xyz_pixels = tf.multiply(xyz_pixels, [0.950456, 1.0, 1.088754])

        with tf.name_scope("xyz_to_srgb"):
            xyz_to_rgb = tf.constant([
                #     r           g          b
                [ 3.2404542, -0.9692660,  0.0556434], # x
                [-1.5371385,  1.8760108, -0.2040259], # y
                [-0.4985314,  0.0415560,  1.0572252], # z
            ])
            rgb_pixels = tf.matmul(xyz_pixels, xyz_to_rgb)
            # avoid a slightly negative number messing up the conversion
            rgb_pixels = tf.clip_by_value(rgb_pixels, 0.0, 1.0)
            linear_mask = tf.cast(rgb_pixels <= 0.0031308, dtype=tf.float32)
            exponential_mask = tf.cast(rgb_pixels > 0.0031308, dtype=tf.float32)
            srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + ((rgb_pixels ** (1/2.4) * 1.055) - 0.055) * exponential_mask

        return tf.reshape(srgb_pixels, tf.shape(lab))

def load_examples():
    if a.input_dir is None or not os.path.exists(a.input_dir):
        raise Exception("input_dir does not exist")
    #获取input 文件
    input_paths = glob.glob(os.path.join(a.input_dir,"*.jpg")) #找出所有相关文件名
    decode = tf.image.decode_ipeg
    if len(input_paths) == 0:
        input_paths = glob.glob(os.path.join(a.input_dir,"*.png"))
        decode = tf.image.decode_png

    if len(input_paths) == 0: 
        raise Exception("input_dir contains no image files")

    def get_name(path):
        name,_ = os.path.splitext(os.path.basename(path))
        return name
    #将input 文件按不同类别分类
    if all(get_name(path).isdigit() for path in input_paths): #如果图片名为digit，则按digit分类
        input_paths = sorted(input_paths,key = lambda path: int(get_name(path)))
    else:
        input_paths = sorted(input_paths)#否则，按文件名分类
    #获取input_images
    with tf.name_scope("load_images"):
        path_queue = tf.train.string_input_producer(input_paths,shuffle=a.mode == "train") #根据Input_paths创建一个输入队列
        reader = tf.WholeFileReader()
        paths,contents = reader.read(path_queue)#读取文件
        raw_input = decode(contents)#解密文件
        raw_input = tf.image.convert_image_dtype(raw_input,dtype=tf.float32) #图片归一化，返回[0,1]浮点类型数据

        assertion = tf.assert_equal(tf.shape(raw_input)[2],3,message="image does not have 3 channels")
        with tf.control_dependencies([assertion]):#在有些机器学习程序中我们想要指定某些操作执行的依赖关系，这时我们可以使用tf.control_dependencies()来实现。 control_dependencies(control_inputs)返回一个控制依赖的上下文管理器，使用with关键字可以让在这个上下文环境中的操作都在control_inputs 执行
            raw_input = tf.identity(raw_input)#如果image有3channel，则执行该语句；返回一个一模一样的新的tensor

        raw_input.set_shape([None,None,3]) #???
        #将images分为A,B两类，并将A,B分到inputs，和，targets里边
        if a.lab_colorization:
            lab = rgb_to_lab(raw_input) #转为灰白图像？？？
            L_chan,a_chan,b_chan = preprocess_lab(lab) #???
            a_images = tf.expand_dims(L_chan,axis=2)
            b_images = tf.stack([a_chan,b_chan],axis=2)
        else:
            width = tf.shape(raw_input)[1]
            a_images = preprocess(raw_input[:,:width//2,:])#???
            b_images = preprocess(raw_input[:,width//2:,:])
    if a.which_direction == "AtoB":
        inputs,targets = [a_images,b_images]
    elif a.which_direction == "BtoA":
        inputs,targets = [b_images,a_images]
    else:
        raise Exception("invaild direction")

    seed = random.randint(0,2**31 - 1) #???
    def transform(image): #将image转到相同尺寸大小
        r = image
        if a.flip:
            r = tf.image.random_flip_left_right(r,seed=seed)
        r = tf.image.resize_images(r,[a.scale_size,a.scale_size],method=tf.image.ResizeMethod.AREA)

        offset = tf.cast(tf.floor(tf.random_uniform([2],0,a.scale_size - CROP_SIZE + 1,seed = seed)),dtype = tf.int32)
        if a.scale_size > CROP_SIZE:
            r = tf.image.crop_to_bounding_box(r,offset[0],offset[1],CROP_SIZE,CROP_SIZE)
        elif a.scale_size < CROP_SIZE:
            raise Exception("scale size cannot be less than crop size")
        return r
    #将inputs和targets中image size根据要求作统一
    with tf.name_scope("input_images"):
        imput_images = transform(inputs)

    with tf.name_scope("target_images"):
        target_images = transform(targets)
    #对inputs 和 targets 分批次
    paths_batch,inputs_batch,targets_path = tf.train.batch([paths,input_images,target_images],batch_size = a.batch_size)
    step_per_epoch = int(math.ceil(len(input_paths)/a.batch_size))
    #返回Examples类实例
    return Examples(paths=paths_batch, 
                    inputs=inputs_batch,
                    targets=targets_batch,
                    count=len(input_paths),
                    steps_per_epoch=steps_per_epoch,
                    )

def create_generator(generator_inputs,generator_outputs_channels):
    layers = []

    with tf.variable_scope("encoder_1"):
        output = conv(generator_inputs,a.ngf,stride=2)
        layers.append(output)

    layer_specs = [
        a.ngf * 2,
        a.ngf * 4,
        a.ngf * 8,
        a.ngf * 8,
        a.ngf * 8,
        a.ngf * 8,
        a.ngf * 8,
        ]

    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = lrelu(layers[-1],0.2)
            convolved = conv(rectified,out_channels,stride=2)
            output = batchnorm(convolved)
            layers.append(output)

    layer_specs = [
        (a.ngf * 8,0.5),
        (a.ngf * 8,0.5),
        (a.ngf * 8,0.5),
        (a.ngf * 8,0.0),
        (a.ngf * 4,0.0),
        (a.ngf * 2,0.0),
        (a.ngf,0.0),
        ]

    num_encoder_layers = len(layers)
    for decoder_layer,(out_channels,dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1 #???层间跳跃，引用RNN的一种处理方法，他是为了解决overfitting而设立的？？？ 为了防止梯度过饱和，如果网络优化到一定程度，没有再继续的必要，可以通过“跳跃”维持原状，具体细节，需要重新复习
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                input = layers[-1]
            else:
                input = tf.concat([layers[-1],layers[skip_layer]],axis=3)

            rectified = tf.nn.relu(input)
            output = deconv(rectified,out_channels)
            output = batchnorm(output)

            if dropout > 0.0: #防止过拟合的一种手段，即只对部分结点进行优化
                output = tf.nn.dropout(output,keep_prob = 1- dropout)
                
            layers.append(output) #先卷积，然后在反卷积

    with tf.variable_scope("decoder_1"):
        input = tf.concat([layers[-1],layers[0]],axis=3)
        rectified = tf.nn.relu(input)
        output = deconv(rectified,generator_outputs_channels)
        output = tf.tanh(output)
        layers.append(output)

    return layers[-1]

def create_model(inputs, targets):
    def create_discriminator(discrim_inputs, discrim_targets):#创建判别网络
        n_layers = 3
        layers = []

        # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
        input = tf.concat([discrim_inputs, discrim_targets], axis=3)#将目标图片，以及生成网络图片组合，作为判别网络的输入

        # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
        with tf.variable_scope("layer_1"):
            convolved = conv(input, a.ndf, stride=2)
            rectified = lrelu(convolved, 0.2)
            layers.append(rectified) #将卷积层 添加到 layers 中

        # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
        # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
        # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
        for i in range(n_layers):
            with tf.variable_scope("layer_%d" % (len(layers) + 1)): #进行3次卷积
                out_channels = a.ndf * min(2**(i+1), 8) #规定output 的channel
                stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
                convolved = conv(layers[-1], out_channels, stride=stride)
                normalized = batchnorm(convolved)
                rectified = lrelu(normalized, 0.2)
                layers.append(rectified)

        # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
        with tf.variable_scope("layer_%d" % (len(layers) + 1)): #在进行一次卷积，然后输出概率
            convolved = conv(rectified, out_channels=1, stride=1)
            output = tf.sigmoid(convolved)
            layers.append(output)

        return layers[-1] #返回概率值

    with tf.variable_scope("generator") as scope:
        out_channels = int(targets.get_shape()[-1])
        outputs = create_generator(inputs, out_channels) #创建“由生成网络生成的图片”，其图片大小与target一致，G(y) , x

    # create two copies of discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables
    with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator"):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_real = create_discriminator(inputs, targets) #判断y与x的相似度

    with tf.name_scope("fake_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_fake = create_discriminator(inputs, outputs) #判断y与G(y)的相似度

    with tf.name_scope("discriminator_loss"):
        # minimizing -tf.log will try to get inputs to 1
        # predict_real => 1
        # predict_fake => 0
        discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS))) #计算 判别网络损失

    with tf.name_scope("generator_loss"):
        # predict_fake => 1
        # abs(targets - outputs) => 0
        gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
        gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
        gen_loss = gen_loss_GAN * a.gan_weight + gen_loss_L1 * a.l1_weight #计算 生成网络损失

    with tf.name_scope("discriminator_train"):
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        discrim_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars) #对判别损失进行梯度优化

    with tf.name_scope("generator_train"): #对生成损失进行梯度优化
        with tf.control_dependencies([discrim_train]):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars) #计算梯度
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)#使用计算到的梯度来更新variable

    ema = tf.train.ExponentialMovingAverage(decay=0.99) #滑动平均
    update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])

    global_step = tf.contrib.framework.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)

    return Model(
        predict_real=predict_real,
        predict_fake=predict_fake,
        discrim_loss=ema.average(discrim_loss),
        discrim_grads_and_vars=discrim_grads_and_vars, #计算梯度，需要sess.run()
        gen_loss_GAN=ema.average(gen_loss_GAN),
        gen_loss_L1=ema.average(gen_loss_L1),
        gen_grads_and_vars=gen_grads_and_vars, #计算梯度,需要sess.run()
        outputs=outputs,
        train=tf.group(update_losses, incr_global_step, gen_train), #返回一个Model类； tf.group(input) input是一组operation,当tf.group()完成时，里边的operation也就完成了
    )

def save_images(fetches, step=None):
    image_dir = os.path.join(a.output_dir, "images") #image的保存路径，没有就创建
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filesets = []
    for i, in_path in enumerate(fetches["paths"]):#fetches中存有所有sample，每个sample又分为input,output,target3个image
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8"))) #文件名
        fileset = {"name": name, "step": step}
        for kind in ["inputs", "outputs", "targets"]:
            filename = name + "-" + kind + ".png" #文件名格式
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[kind] = filename #将input，output,target文件名分别存入key下
            out_path = os.path.join(image_dir, filename) #定义输出路径
            contents = fetches[kind][i] #取出kind下第I个image内容，并写入out_path
            with open(out_path, "wb") as f:
                f.write(contents)
        filesets.append(fileset) #将各个sample文件存入filesets
    return filesets #返回filesets； fileset共有5个key；filesets中存有所有sample信息；

def append_index(filesets, step=False):
    index_path = os.path.join(a.output_dir, "index.html")#将所有结果都整合到一个html文件中
    if os.path.exists(index_path):#如果已经建立了output_dir，则追加 info，否则，写入
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:#如果有step的信息，则写入Html文件
            index.write("<th>step</th>")
        index.write("<th>name</th><th>input</th><th>output</th><th>target</th></tr>")

    for fileset in filesets:
        index.write("<tr>")

        if step:
            index.write("<td>%d</td>" % fileset["step"])
        index.write("<td>%s</td>" % fileset["name"])#写入sample名字

        for kind in ["inputs", "outputs", "targets"]:
            index.write("<td><img src='images/%s'></td>" % fileset[kind])#写入具体的sample地址，应该是可以直接将image读入html中吧？？？

        index.write("</tr>")
    return index_path #返回这个html文件

def main():
    if tf.__version__.split('.')[0] != "1":
        raise Exception("Tensorflow version 1 required")

    if a.seed is None:#确定op的执行种子，使得各个op在同一状态执行，避免随机性
        a.seed = random.randint(0, 2**31 - 1)

    tf.set_random_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)

    if not os.path.exists(a.output_dir):#如果output_dir不存在，则创建
        os.makedirs(a.output_dir)

    if a.mode == "test" or a.mode == "export":
        if a.checkpoint is None:#如果此时要执行的是test操作，或者时export操作，而checkpoint没有，则报错
            raise Exception("checkpoint required for test mode")

        # load some options from the checkpoint
        options = {"which_direction", "ngf", "ndf", "lab_colorization"}
        with open(os.path.join(a.checkpoint, "options.json")) as f:
            for key, val in json.loads(f.read()).items():
                if key in options:
                    print("loaded", key, "=", val)#将checkpoint中的value值赋给各个option
                    setattr(a, key, val) #给a的属性key赋值
        # disable these features in test mode
        a.scale_size = CROP_SIZE
        a.flip = False

    for k, v in a._get_kwargs():#给a的属性赋值
        print(k, "=", v)

    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))#将python数据结构var(a)转为json结构

    if a.mode == "export": #如果想要export generator graph
        # export the generator to a meta graph that can be imported later for standalone generation
        if a.lab_colorization:
            raise Exception("export not supported for lab_colorization")

        input = tf.placeholder(tf.string, shape=[1]) #input
        input_data = tf.decode_base64(input[0])
        input_image = tf.image.decode_png(input_data)

        # remove alpha channel if present
        input_image = tf.cond(tf.equal(tf.shape(input_image)[2], 4), lambda: input_image[:,:,:3], lambda: input_image)
        # convert grayscale to RGB
        input_image = tf.cond(tf.equal(tf.shape(input_image)[2], 1), lambda: tf.image.grayscale_to_rgb(input_image), lambda: input_image)

        input_image = tf.image.convert_image_dtype(input_image, dtype=tf.float32)
        input_image.set_shape([CROP_SIZE, CROP_SIZE, 3]) #修正input的size
        batch_input = tf.expand_dims(input_image, axis=0)

        with tf.variable_scope("generator"): #返回的是：generator生成的output,需要sess.run()吧？??
            batch_output = deprocess(create_generator(preprocess(batch_input), 3)) 

        output_image = tf.image.convert_image_dtype(batch_output, dtype=tf.uint8)[0]
        if a.output_filetype == "png":
            output_data = tf.image.encode_png(output_image)
        elif a.output_filetype == "jpeg":
            output_data = tf.image.encode_jpeg(output_image, quality=80)
        else:
            raise Exception("invalid filetype")
        output = tf.convert_to_tensor([tf.encode_base64(output_data)]) #将output转为tensor???

        key = tf.placeholder(tf.string, shape=[1]) #key代表的是什么？？？ 占位符；input
        inputs = {
            "key": key.name, 
            "input": input.name
        }
        tf.add_to_collection("inputs", json.dumps(inputs))#将tf.add_to_collection(a,b)将元素b添加到列表a中；
        outputs = {
            "key":  tf.identity(key).name,
            "output": output.name,
        }
        tf.add_to_collection("outputs", json.dumps(outputs))

        init_op = tf.global_variables_initializer()
        restore_saver = tf.train.Saver() #加载saver
        export_saver = tf.train.Saver() #导入saver???

        with tf.Session() as sess:
            sess.run(init_op) #初始化全局变量
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint) #checkpoint路径
            restore_saver.restore(sess, checkpoint)#将checkpoint数据加载进来
            print("exporting model")
            export_saver.export_meta_graph(filename=os.path.join(a.output_dir, "export.meta"))#支持以json导出metagraphdef
            export_saver.save(sess, os.path.join(a.output_dir, "export"), write_meta_graph=False) #将数据保存到export中

        return #疑问？？？ key,input都是占位符，在利用他们进行运算时，为什么没有sess.run()，

    examples = load_examples() #下载sample
    print("examples count = %d" % examples.count)

    # inputs and targets are [batch_size, height, width, channels]
    model = create_model(examples.inputs, examples.targets) #建立model

    # undo colorization splitting on images that we use for display/output
    if a.lab_colorization:
        if a.which_direction == "AtoB":
            # inputs is brightness, this will be handled fine as a grayscale image
            # need to augment targets and outputs with brightness
            targets = augment(examples.targets, examples.inputs) #返回的是rgb???
            outputs = augment(model.outputs, examples.inputs) #返回的是rgb???
            # inputs can be deprocessed normally and handled as if they are single channel
            # grayscale images
            inputs = deprocess(examples.inputs) #input变为grayscale
        elif a.which_direction == "BtoA":
            # inputs will be color channels only, get brightness from targets
            inputs = augment(examples.inputs, examples.targets) #input变为rgb
            targets = deprocess(examples.targets)#grayscale
            outputs = deprocess(model.outputs)#grayscale
        else:
            raise Exception("invalid direction")
    else:
        inputs = deprocess(examples.inputs) #grayscale
        targets = deprocess(examples.targets)#grayscale
        outputs = deprocess(model.outputs)#grayscale

    def convert(image):
        if a.aspect_ratio != 1.0:
            # upscale to correct aspect ratio
            size = [CROP_SIZE, int(round(CROP_SIZE * a.aspect_ratio))]
            image = tf.image.resize_images(image, size=size, method=tf.image.ResizeMethod.BICUBIC)

        return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

    # reverse any processing on images so they can be written to disk or displayed to user
    with tf.name_scope("convert_inputs"):
        converted_inputs = convert(inputs) 

    with tf.name_scope("convert_targets"):
        converted_targets = convert(targets)

    with tf.name_scope("convert_outputs"):
        converted_outputs = convert(outputs)

    with tf.name_scope("encode_images"):
        display_fetches = {
            "paths": examples.paths,
            "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name="input_pngs"), #inputs中含有Input，input是一个placeholder
            "targets": tf.map_fn(tf.image.encode_png, converted_targets, dtype=tf.string, name="target_pngs"),
            "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name="output_pngs"), #outputs中包含key（placeholder),和，output（tensor)
        }#tf.map_fn() 映射函数：将png映射到image

    # summaries
    with tf.name_scope("inputs_summary"):
        tf.summary.image("inputs", converted_inputs)

    with tf.name_scope("targets_summary"):
        tf.summary.image("targets", converted_targets)

    with tf.name_scope("outputs_summary"):
        tf.summary.image("outputs", converted_outputs)

    with tf.name_scope("predict_real_summary"):
        tf.summary.image("predict_real", tf.image.convert_image_dtype(model.predict_real, dtype=tf.uint8))

    with tf.name_scope("predict_fake_summary"):
        tf.summary.image("predict_fake", tf.image.convert_image_dtype(model.predict_fake, dtype=tf.uint8))

    tf.summary.scalar("discriminator_loss", model.discrim_loss)
    tf.summary.scalar("generator_loss_GAN", model.gen_loss_GAN)
    tf.summary.scalar("generator_loss_L1", model.gen_loss_L1)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/values", var)

    for grad, var in model.discrim_grads_and_vars + model.gen_grads_and_vars:
        tf.summary.histogram(var.op.name + "/gradients", grad)

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=1)

    logdir = a.output_dir if (a.trace_freq > 0 or a.summary_freq > 0) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None) #管理模型训练过程
    with sv.managed_session() as sess:
        print("parameter_count =", sess.run(parameter_count))

        if a.checkpoint is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            saver.restore(sess, checkpoint)

        max_steps = 2**32
        if a.max_epochs is not None:
            max_steps = examples.steps_per_epoch * a.max_epochs
        if a.max_steps is not None:
            max_steps = a.max_steps

        if a.mode == "test":
            # testing
            # at most, process the test data once
            max_steps = min(examples.steps_per_epoch, max_steps)
            for step in range(max_steps):
                results = sess.run(display_fetches) #path,output,input,target,output ，所有需要的输入都在这里 ？？？creat_model()时已经输入
                filesets = save_images(results)
                for i, f in enumerate(filesets):
                    print("evaluated image", f["name"])
                index_path = append_index(filesets)

            print("wrote index at", index_path)
        else:
            # training
            start = time.time()

            for step in range(max_steps):
                def should(freq):
                    return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

                options = None
                run_metadata = None
                if should(a.trace_freq):
                    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE) #定义tensorflow运行选项
                    run_metadata = tf.RunMetadata() #定义tensorflow运行元信息

                fetches = {
                    "train": model.train,
                    "global_step": sv.global_step,
                }

                if should(a.progress_freq):
                    fetches["discrim_loss"] = model.discrim_loss
                    fetches["gen_loss_GAN"] = model.gen_loss_GAN
                    fetches["gen_loss_L1"] = model.gen_loss_L1

                if should(a.summary_freq):
                    fetches["summary"] = sv.summary_op

                if should(a.display_freq):
                    fetches["display"] = display_fetches

                results = sess.run(fetches, options=options, run_metadata=run_metadata) #sess.run()input...隐含？？？是的，creat_model()中已经输入

                if should(a.summary_freq):
                    print("recording summary")
                    sv.summary_writer.add_summary(results["summary"], results["global_step"])

                if should(a.display_freq):
                    print("saving display images")
                    filesets = save_images(results["display"], step=results["global_step"])
                    append_index(filesets, step=True)

                if should(a.trace_freq):
                    print("recording trace")
                    sv.summary_writer.add_run_metadata(run_metadata, "step_%d" % results["global_step"])

                if should(a.progress_freq):
                    # global_step will have the correct step count if we resume from a checkpoint
                    train_epoch = math.ceil(results["global_step"] / examples.steps_per_epoch)
                    train_step = (results["global_step"] - 1) % examples.steps_per_epoch + 1
                    rate = (step + 1) * a.batch_size / (time.time() - start)
                    remaining = (max_steps - step) * a.batch_size / rate
                    print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (train_epoch, train_step, rate, remaining / 60))
                    print("discrim_loss", results["discrim_loss"])
                    print("gen_loss_GAN", results["gen_loss_GAN"])
                    print("gen_loss_L1", results["gen_loss_L1"])

                if should(a.save_freq):
                    print("saving model")
                    saver.save(sess, os.path.join(a.output_dir, "model"), global_step=sv.global_step)

                if sv.should_stop():
                    break


main()

#总结一下：
#在main()中进行export , test, train 3种操作
#其中，export，用到saver.restore() ,saver.save(),不用sess.run(),不用return
#test，用到creat_model(),直接用sess.run(model.outputs)就可以，其中creat_model（）需要的input,targets，已经在main中表明creat_model(examples.input,examples.target)，因此，不需feed_dict
#train,依然是用到creat_model() 中的model.train(),因此，直接sess.run(),由于input已经在creat_model()中定义，不需feed_dict

#所得example均从shell中输入他的路径

#还有2点需要说明：
#一个是:
#creat_generator() 最后输出的是output image，在通篇code中，只书写过程，但是，没有sess.run()；同样的，在creat_model()中也没有sess.run()，所有的sess.run()均是在test和train()中进行的；

#creat_generator()只是定义了生成网络结构，以及，利用该结构，生成的output_image(return)
#creat_dicriminator() 同generator()
#creat_model() 将 generator()和discriminator()结合起来，计算pix2pix损失，返回一个 pix2pix Model，这个model带有各种需要sess.run()的属性

