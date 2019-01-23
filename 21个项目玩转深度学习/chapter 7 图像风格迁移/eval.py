#利用已经预训练好的模型进行图像的快速风格迁移
# coding:utf-8
from __future__ import print_function
import tensorflow as tf
from prerocessing import preprocessing_factory #引入了Tensorflow Slim中的一些预训练模型
import reader #将训练图片读入tensorflow
import model #用于定义图像生成网络
import time
import os

tf.app.flags.DEFINE_string('loss_model','vgg_16','The name of the architecture to evaluate.''You can view all the support models in nets/nets_factory.py')
tf.app.flags.DEFINE_integer('image_size',256,'Image size to train')
tf.app.flags.DEFINE_string('model_file','models.ckpt','')
tf.app.flags.DEFINE_string('image_file','a.jpg','') #tf.app.flags.DEFINE_*** 类似于parsers.add_arguments()，用于添加命令行参数

FLAGS = tf.app.flags.FLAGS #通过FLAGS.image_file可以读取 coder在命令行输入的参数；tf.app.flags.FLAGS 可以与 tf.app.run()联用，tf.app.run()默认运行main()；如果tf.app.run(test)，则运行test()，test的参数为FLAGS中的参数

def main(_): #main(_)表示什么意思？？？
    #get image width and height
    width = 0
    height = 0
    with open(FLAGS.image_file,'rb') as img:
        with tf.Session().as_default as sess:
            if FLAGS.image_file.lower().endswith('png'):
                image = sess.run(tf.image.decode_png(img.read()))
            else:
                image = sess.run(tf.image.decode_jpge(img.read()))
            height = image.shape[0]
            width = image.shape[1]
    tf.logging.info('image size is : %dx%d' % (height,width))

    with tf.Graph().as_default():
        with tf.Session.as_default() as sess:
            #Read image data
            image_preprocessing_fn,_ = preprocessing_factory.get_preprocessing(FLAGS.loss_model,is_training = False) #FLAGS.loss_model为用于评估的model，该function返回的是，在loss_model下，对image进行预处理的function
            image = reader.get_image(FLAGS.image_file,height,width,image_preprocessing_fn) #利用image_preprocessing_fn函数，对image进行预处理，返回预处理后的image

            #Add batch dimension
            image = np.expand_dims(image,0)

            generated = model.net(image,training = False) #形成图像风格迁移：返回的是一个需要sess.run()的model
            generated = tf.cast(generated,tf.uint8) #将generated的数据格式转为uint8

            #remove batch dimension
            generated = np.squeeze(generated,[0])

            #restore model variables??? 预训练模型，为什么还要在刷新其参数？？？
            saver = tf.train.saver(tf.global_variables(),write_version = tf.train.SaverDef.V1) #创建类实例
            #是否可去？？？： sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()]) #这里运行的是model.net中的图像生成网络？input image在最开始的时候，已经输入图像生成网络，这里是进行训练，但是，在model.net(train=False)，所以不需要进行训练，那么这里为什么还有在对参数进行初始化呢？不合逻辑，正常理解，应该是，重载模型参数，saver.restore()，然后，在用sess.run(generated)???

            #use absolute path
            FLAGS.model_file = os.path.abspath(FLAGS.model_file)
            saver.restore(sess,FLAGS.model_file) #重载模型参数，为什么要重载模型参数，不是已经风格迁移了吗？？？
            #上边这几句是不是写的反了？？？

            #make sure generated directory exists
            generated_file = 'generated/res.jpg'
            if not os.path.exists('generated'):
                os.makedirs('generated')

            #generate and write image data to file
            with open(generated_file,'wb') as img:
                start_time = time.time()
                img.write(sess.run(tf.encode_jpeg(generated)))
                end_time = time.time()
                tf.logging.info('Elapsed time: %fs' % (end_time - start_time))

                tf.logging.info('Done. Please check %s.' % generated_file)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO) #在屏幕显示INFO等级及其以上的log信息
    tf.app.run() #运行main（）函数
                






    
