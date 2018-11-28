#tensorflow数据读取
#导入tensorflow
import tensorflow as tf
#新建一个Session
with tf.Session as sess:
    filename=['A.jpg','B.jpg','C.jpg']
    #string_input_producer产生文件名队列
    filename_queue=tf.train.string_input_producer(filename,shuffle=False,num_epochs=5)
    #利用reader从文件名列队中读取数据
    reader =tf.WholeFileReader()
    key,value=reader.read(filename_queue)
    #对string_input_producer中变量epoch进行初始化？？？
    tf.local_variables_initializer().run()
    #启动填充线程，将读取出的数据，填入内存队列
    threads =tf.train.start_queue_runners(sess=sess)
    i=0
    while True:
        i +=1
        #从Session中读取数据
        image_data=sess.run(value) #image_data为二进制数据
        #将读取的数据存入文件夹read中
        with open('read/test_%d.jpg' % i,'wb') as f:
            f.write(image_data)
