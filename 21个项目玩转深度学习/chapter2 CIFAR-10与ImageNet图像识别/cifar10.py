#随机裁剪图片
distorted_image = tf.random_crop(reshaped_image,[height,width,3])
#随机翻转
distorted_image = tf.image.random_flip_left_right(distorted_image)
#随机改变亮度和对比度
distorted_image = tf.image.random_brightness(distorted_image,max_delta=63)
distorted_image = tf.image.random_contrast(distorted_image,lower=0.2,upper=0.8)

def inference(images):
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weight',shape=[5,5,3,64],stddev=5e-2,wd=0.0)
        conv = tf.nn.conv2d(images,kernel,[1,1,1,1],padding='SAME')
        biases =_variable_on_cpu('biases',[64],tf.constant_initializer(0.0))
        pre_activation =tf.nn.bias_add(conv,biases)
        conv1 =tf.nn.relu(pre_activation,name=scope.name)
        _activation_summary(conv1)

        pool1 =tf.nn.max_pool(conv1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME',name='pool1')

        norm1 = tf.nn.lrn(pool1,4,bias=1.0,alpha=0.001/9.0,beta=0.75,name='norm1')

    with tf.variable_scope('conv2') as scope:
        kernel =_variable_with_weight_decay('weights',shape=[5,5,64,64],stddev=5e-2,wd=0.0)
        biases =_variable_on_cpu('biases',[64],tf.constant_initializer(0.1))
        conv =tf.nn.conv2d(norm1,kernel,[1,1,1,1],padding='SAME')
        pre_activation =tf.nn.bias_add(conv,biases)
        conv2 =tf.nn.relu(pre_activation,name=scope.name)
        _activation_summary(conv2)
    norm2 =tf.nn.lrn(conv2,4,bias=1.0,alpha=0.001/9.0,beta=0.75,name='norm2')
    pool2 =tf.nn.max_pool(norm2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME',name='pool2')

    with tf.variable_scope('local3')as scope:
        reshape =tf.reshape(pool2,[FLAGS.batch_size,-1])
        dim =reshape.get_shape()[1].value
        weights =_variable_with_weight_decay('weights',shape=[dim,384],stddev=0.04,wd=0.004)
        biases =_variable_on_cpu('biases',[384],tf.constant_initializer(0.1))
        local3 =tf.nn.relu(tf.add(tf.matmul(reshape,weights),biases),name=scope.name)
        _activation_summary(local3)


     with tf.variable_scope('local4') as scope:
         weights =_variable_with_weight_decay('weights',shape=[384,192],stddev=0.04,wd=0.004)
         biases =_variable_on_cpu('biases',[192],tf.constant_initializer(0.1))
         local4 =tf.nn.relu(tf.add(tf.matmul(local3,weights),biases),name=scope.name)
         _activation_summary(local4)


     with tf.variable_scope('softmax_linear') as scope:
         weights =_variable_with_weight_decay('weights',[192,NUM_CLASSES],stddev=1/192.0,wd=0.0)
         biases =_variable_on_cpu('biases',[NUM_CLASSES],tf.constant_initializer(0.0))
         softmax_linear =tf.add(tf.matmul(local4,weights),biases,name=scope.name)
         _activation_summary(softmax_linear)

return softmax_linear

#命令行输入
python cifar10.py --train_dir cifar10_train/ --data_dir cifar10_data/
         
#在tensorflow中利用tensorboard查看训练进度,在cmd中input下列语句
tensorboard --logdir cifar10_train

        

        
