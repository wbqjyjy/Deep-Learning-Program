#将CIFAR-10数据集中的图片读取出来
if __name__ =="__main__"
with tf.Session as sess:
    reshape_image = inputs_origin('cifar10_data/cifar-10-batches-bin')
    threads =tf.train.start_queue_runners(sess=sess)
    sess.run(tf.global_variables_initializer()) #对变量初始化???
    if not os.path.exists('cifar10_data/raw/'):
        os.makedirs('cifar10_data/raw/')

    for i in range(30):
        image_array =sess.run(reshape_image)
        scipy.misc.toimage(image_array).save('cifar10_data/raw/%d.jpg'% i)

def inputs_origin(data_dir):
    filenames =[os.path.join(data_dir,'data_batch_%d.bin'% i) for i in xrange(1,6)]
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file:'+f)
    filename_queue =tf.train.string_input_producer(filenames)
    read_input =cifar10_input.read_cifar10(filename_queue)
    reshaped_image =tf.cast(read_input.uint8image,tf.float32)
    return reshaped_image

## read_cifar10 main code
label_bytes=1
result.height =32
result.width =32
result.depth =3
image_bytes =result.height * result.width * result.depth
record_bytes =label_bytes + image_bytes
reader= tf.FixedLengthRecordReader(record_bytes = record_bytes)
result key,value = reader.read(filename_queue)
    
