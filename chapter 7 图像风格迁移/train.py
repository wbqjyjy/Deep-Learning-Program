#coding: utf-8

from __future__ import print_function
from __future__ import division
import tensorflow as tf
from nets import nets_factory #自定义模块
from preprocessing import preprocessing_factory #自定义模块
import reader #自定义模块
import model #自定义模块
import time
import losses #自定义模块
import utils #定义了一些convenient function
import os
import argparse

slim = tf.contrib.slim #使code更简洁？？？,完全没有用啊？？？可能在其他函数里用到了

def parse_args():
    parser = argparse.ArgumentParser()#建立类实例
    parser.add_argument('-c','--conf',default = 'conf/mosaic.yml',help = 'the path to the conf file')#添加命令行参数
    return parser.parse_args() #返回namespace
    
                                              
def main(FLAGS):
    style_features_t = losses.get_style_features(FLAGS) #style target的Gram

    #make sure the training path exists
    training_path = os.path.join(FLAGS.model_path,FLAGS.naming) #model/wave/ ；用于存放训练好的模型
    if not (os.path.exists(training_path)):
        os.makedirs(training_path)

    with tf.Graph().as_default(): #默认计算图
        with tf.Session() as sess:#没有as_default(),因此，走出with 语句，sess停止执行，不能在被用
            """build loss network"""
            network_fn =nets_factory.get_network_fn(FLAGS.loss_model,num_classes=1,is_training=False) #取出loss model，且该model不用训练
            #对要进入loss_model的content_image,和generated_image进行preprocessing
            image_preprocessing_fn,image_unpreprocessing_fn = preprocessing_factory.get_preprocessing(FLAGS.loss_model,is_training=False) #取出用于loss_model的，对image进行preprocessing和unpreprocessing的function
            processed_image = reader.image(FLAGS.batch_size,FLAGS.image_size,FLAGS.image_size,'train2014/',image_preprocessing_fn,epochs=FLAGS.epoch) #这里要preprocessing的image是一个batch，为training_data
            generated = model.net(processed_images,training=True) #输入“图像生成网络”的image为经过preprocessing_image，“图像生成网络”为要训练的网络
            processed_generated = [image_preprocessing_fn(image,FLAGS.image_size,FLAGS.image_size) for image in tf.unstack(generated,axis=0,num=FLAGS.batch_size)]
            processed_generated = tf.stack(processed_generated)
            #计算generated_image和content_image进入loss_model后，更layer的output
            _,endpoints_dict= network_fn(tf.concat([processed_generated,processed_images],0),spatial_squeeze=False)#endpoints_dict中存储的是2类image各个layer的值
            #log the structure of loss network
            tf.logging.info('loss network layers(you can define them in "content layer" and "style layer"):')
            for key in endpoints_dict:
                tf.logging.info(key) #屏幕输出loss_model的各个layer name

            """build losses"""
            content_loss = losses.content_loss(endpoints_dict,FLAGS.content_layers)
            style_loss,style_loss_summary = losses.style_loss(endpoints_dict,style_features_t,FLAGS.style_layers)
            tv_loss = losses.total_variation_loss(generated)

            loss = FLAGS.style_weight * style_loss + FLAGS.content_weight * content_loss + FLAGS.tv_weight * tv_loss

            # Add Summary for visualization in tensorboard
            """Add Summary"""
            tf.summary.scalar('losses/content_loss',content_loss)
            tf.summary.scalar('losses/style_loss',style_loss)
            tf.summary.scalar('losses/regularizer_loss',tv_loss)

            tf.summary.scalar('weighted_losses/weighted content_loss',content_loss * FLAGS.content_weight)
            tf.summary.scalar('weighted_losses/weighted style_loss',style_loss * FLAGS.style_weight)
            tf.summary.scalar('weighted_losses/weighted_regularizer_loss',tv_loss * FLAGS.tv_weight)
            tf.summary.scalar('total_loss',loss)

            for layer in FLAGS.style_layers:
                tf.summary.scalar('style_losses/' + layer,style_loss_summary[layer])
            tf.summary.image('genearted',generated)
            tf.summary.image('origin',tf.stack([image_unprocessing_fn(image) for image in tf.unstack(processed_images,axis=0,num=FLAGS.batch_size)]))
            summary = tf.summary.merge_all()
            writer = tf.summary.FileWriter(training_path)

            """prepare to train"""
            global_step = tf.Variable(0,name='global_step',trainable=False)#iteration step

            variable_to_train = []#需要训练的变量
            for variable in tf.trainable_variables():#在图像风格迁移网络（图像生成网络+损失网络）各参数中，找需要训练的参数
                if not (variable.name.startswith(FLAGS.loss_model)):
                    variable_to_train.append(variable)
            train_op = tf.train.AdamOptimizer(1e-3).minimize(loss,global_step = global_step,var_list = variable_to_train) #需要放入sess.run()

            variable_to_restore = []#在所有的全局变量中，找需要恢复默认设置的变量； 注意：local_variable指的是一些临时变量和中间变量，用于线程中，线程结束则消失
            for v tf.global_variables():
                if not (v.name.startswith(FLAGS.loss_model)):
                    variables_to_restore.append(v)
            saver = tf.train.Saver(variables_to_restore,write_version=tf.train.SaverDef.V1)#利用saver.restore()恢复默认设置；这里的variable_to_restore，是需要save and restore的var_list

            sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])#对全局变量和局部变量进行初始化操作：即恢复默认设置

            #restore variables for loss model 恢复loss model中的参数
            init_func = utils._get_init_fn(FLAGS)
            init_func(sess)

            #restore variables for training model if the checkpoint file exists. 如果training_model已有训练好的参数，将其载入
            last_file = tf.train.latest_checkpoint(training_path)#将train_path中的model参数数据取出
            if last_file:
                tf.logging.info('restoringmodel from {}'.format(last_file))
                saver.restore(sess,last_file) #那如果last_file不存在，就不执行restore操作吗？需要restore的参数只是图像生成网络吗？

            """start training"""
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            start_time = time.time()
            try:
                while not coord.should_stop():#查看线程是否停止(即：是否所有数据均运行完毕）
                    _,loss_t,step = sess.run([train_op,loss,global_step])
                    elapsed_time = time.time()
                    """logging"""
                    #print(step)
                    if step % 10 == 0:
                        tf.logging.info('step:%d, total loss %f, secs/step: %f' % (step,loss_t,elapsed_time))
                    """summary"""
                    if step % 25 == 0:
                        tf.logging.info('adding summary...')
                        summary_str = sess.run(summary)
                        writer.add_summary(summary_str,step)
                        writer.flush()
                    """checkpoint"""
                    if step % 1000 == 0:
                        saver.save(sess,os.path.join(training_path,'fast-style-model.ckpt'),global_step=step)#保存variable_to_restore中的参数值
            except tf.errors.OutOfRangeError:
                saver.save(sess,os.path.join(training_path,'fast-style-model.ckpt-done'))
                tf.logging.info('Done training -- epoch limit reached')
            finally:
                coord.request_stop()#要求停止所有线程
            coord.join(threads)#将线程并入主线程，删除
                
                    
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO) #将INFO及其以上的警告信息显示在屏幕上
    args = parse_args() #设定命令行参数，主要是configure document 存储位置
    FLAGS = utils.read_conf_file(args.conf) #读取configure
    main(FLAGS)
    #注意一个区别，tf.app.run()一般会与tf.app.flags.DEFINE_string()（定义命令行，同argparse.add_argument() ; FLAGS = tf.app.FLASG(读取命令行参数，如FLAGS.conf) 联用
    #而argparse设定命令行参数，返回args=parse_args()，将命令行参数args作为main()的形参；


#总结一下train.py的思路
#step1：取出loss_model，not train
#step2:将origin image进行input前的预处理
#step2:将preprocessed origin输入“图像生成网络”，形成generated，“图像生成网络”需要train
#step3:将generated进行input loss model前的预处理
#step4:将preprocessed origin和preprocessed generated送入loss model，得到各层的输出endpoints_dict
#step5:计算内容损失，和，风格损失，得到total_loss
#step5:建立需要训练的参数的list，以及需要restore and save的参数list
#step6:利用Adam梯度下降来优化total_loss，需要训练的参数放在variable_to_train list中
#step6:将全局变量和局部变量初始化
#step6:将“图像生成网络”和其它参数，进行restore
#step7:完成初始化操作以后，即可利用sess.run([train_op,loss_t,step])开始训练
#step8:训练结束后，可以通过saver.save(sess)将variable_to_restore list中的参数保存入train_path
            
#有一个疑问：train的步骤是在sess.run(train_op)中完成的，那，train_op的output是什么？？？是variable_to_train list
#saver.save(sess,os.path.join(path,'name'))，之所以不用明确指出需要保存那些variable，是因为saver=tf.train.Saver(variable_to_restore)中，已经指明；











            
            

            
            

        
        

        


