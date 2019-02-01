import tensorflow as tf
from model import CycleGAN #自定义模块？
from reader import Reader
from datetime import datetime
import os
import logging
from utils import ImagePool

FLAGS = tf.flags.FLAGS #与tf.app.run()联用

tf.flags.DEFINE_integer('batch_size', 1, 'batch size, default: 1')
tf.flags.DEFINE_integer('image_size', 256, 'image size, default: 256')
tf.flags.DEFINE_bool('use_lsgan', True,
                     'use lsgan (mean squared error) or cross entropy loss, default: True')
tf.flags.DEFINE_string('norm', 'instance',
                       '[instance, batch] use instance norm or batch norm, default: instance')
tf.flags.DEFINE_integer('lambda1', 10.0,
                        'weight for forward cycle loss (X->Y->X), default: 10.0')
tf.flags.DEFINE_integer('lambda2', 10.0,
                        'weight for backward cycle loss (Y->X->Y), default: 10.0')
tf.flags.DEFINE_float('learning_rate', 2e-4,
                      'initial learning rate for Adam, default: 0.0002')
tf.flags.DEFINE_float('beta1', 0.5,
                      'momentum term of Adam, default: 0.5')
tf.flags.DEFINE_float('pool_size', 50,
                      'size of image buffer that stores previously generated images, default: 50')
tf.flags.DEFINE_integer('ngf', 64,
                        'number of gen filters in first conv layer, default: 64')
tf.flags.DEFINE_string('X', 'data/tfrecords/apple.tfrecords',
                       'X tfrecords file for training, default: data/tfrecords/apple.tfrecords')
tf.flags.DEFINE_string('Y', 'data/tfrecords/orange.tfrecords',
                       'Y tfrecords file for training, default: data/tfrecords/orange.tfrecords')
tf.flags.DEFINE_string('load_model', None,
                        'folder of saved model that you wish to continue training (e.g. 20170602-1936), default: None')

def train():
    if FLAGS.load_model is not None: #如果该命令行参数不为空，则据此给出checkpoint_dir
        checkpoints_dir = "checkpoints/" + FLAGS.load_model
    else: #否则，根据当前时间，创建一个checkpoint_dir
        current_time = datetime.now().strftime("%Y%m%d - %H%M")
        checkpoints_dir = "checkpoints/{}".format(current_time)
        try:
            os.makedirs(checkpoints_dir)
        except os.error:
            pass

    graph = tf.Graph() #创建计算图
    with graph.as_default():
        cycle_gan = CycleGAN(
            X_train_file = FLAGS.X,
            Y_train_file = FLAGS.Y,
            batch_size = FLAGS.batch_size,
            image_size = FLAGS.image_size,
            use_lsgan = FLAGS.use_lsgan,
            norm = FLAGS.norm,
            lambda1 = FLAGS.lambda1,
            lambda2 = FLAGS.lambda1,
            learning_rate = FLAGS.learning_rate,
            beta1 = FLAGS.beta1,
            ngf = FLAGS.ngf
            ) #引入CycleGAN网络
        G_loss,D_Y_loss,F_loss,D_X_loss,fake_y,fake_x = cycle_gan.model() #返回值分别是：反向生成网络损失，正向判别函数损失，生成网络损失，逆向判别函数损失，正向生成的y，反向生成的x
        optimizers = cycle_gan.optimize(G_loss,D_Y_loss,F_loss,D_X_loss) #四个损失的优化器

        summary_op = tf.summary.merge_all() #将一些信息显示在stdoutput中
        train_writer = tf.summary.FileWriter(checkpoints_dir,graph) #将图保存在checkpoints_dir中
        saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        if FLAGS.load_model is not None: #如果已存在训练模型，则加载继续训练
            checkpoint = tf.train.get_checkpoint_state(checkpoints_dir) #将最新的model加载进来
            meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
            restore = tf.train.import_meta_graph(meta_graph_path) #加载model结构
            restore.restore(sess,tf.train.latest_checkpoint(checkpoints_dir)) #加载最新的model模型参数
            step = int(meta_graph_path.split("-")[2].split(".")[0])
        else:
            sess.run(tf.global_variables_initializer()) #初始化全局变量
            step = 0

        coord = tf.train.Coordinator() #进行线程管理
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)

        try:
            fake_Y_pool = ImagePool(FLASG.pool_size) #设定image缓冲大小
            fake_X_pool = ImagePool(FLAGS.pool_size)

            while not coord.should_stop():
                fake_y_val,fake_x_val = sess.run([fake_y,fake_x]) #先得出generated image x,y？？？

                #train
                _,G_loss_val,D_Y_loss_val,F_loss_val,D_X_loss_val,summary = (
                    sess.run(
                        [optimizers,G_loss,D_Y_loss,F_loss,D_X_loss,summary_op],
                        feed_dict = {cycle_gan.fake_y:fake_Y_pool.query(fake_y_val), #将上述得到的fake_x,fake_y输入到optimizers，G_loss，...，中，优化； 假设，初始化F,D_y，然后根据x得到fake_y，然后根据G,D_x，y，得到fake_x，根据这些value：x,y,fake_x,fake_y，求上述的几个loss，利用优化器对其进行优化
                                     cycle_gan.fake_x:fake_X_pool.query(fake_x_val)} #还是没太弄明白 为什么一会儿fake_y，一会儿self.fake_y；是要缓冲若干个fake_y？？？
                        )
                    ) #进行训练
                if step % 100 == 0: #到100步时，将信息输出到stdout
                    train_writer.add_summary(summary,step)
                    train_writer.flush()

                if step % 100 == 0:
                    logging.info('----------step %d:--------------' % step)
                    logging.info(' G_loss : {}'.format(G_loss_val))
                    logging.info(' D_Y_loss : {}'.format(D_Y_loss_val))
                    logging.info(' F_loss : {}'.format(F_loss_val))
                    logging.info(' D_X_loss : {}'.format(D_X_loss_val))

                if step % 10000 == 0:
                    save_path = saver.save(sess,checkpoints_dir + "/model.ckpt",global_step=step)
                    logging.info("Model saved in file: %s" % save_path)

                step += 1

        except KeyboardInterrupt:
            logging.info('Interrupted')
            coord.request_stop()
        except Exception as e:
            coord.request_stop(e)
        finally:
            save_path = saver.save(sess,checkpoints_dir + "/model.ckpt",global_step=step) #训练完成后，将训练好的model保存起来.ckpt；
            logging.info("Model saved in file: %s" % save_path)
            coord.request_stop()
            coord.join(threads)

def main(unused_argv):
    train()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO) #将等级为info及其以上的信息输出
    tf.app.run() #运行main()



                
                





















        
        
            

