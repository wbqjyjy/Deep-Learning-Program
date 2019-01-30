#将图像缩放到统一大小

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import tempfile
import subprocess
import tensorflow as tf
import numpy as np
import tfimage as im
import threading
import time
import multiprocessing

edge_pool = None

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", required=True, help="path to folder containing images")
parser.add_argument("--output_dir", required=True, help="output path")
parser.add_argument("--operation", required=True, choices=["grayscale", "resize", "blank", "combine", "edges"])
parser.add_argument("--workers", type=int, default=1, help="number of workers")
# resize
parser.add_argument("--pad", action="store_true", help="pad instead of crop for resize operation")
parser.add_argument("--size", type=int, default=256, help="size to use for resize operation")
# combine
parser.add_argument("--b_dir",type=str,help="path to folder containing B images for combine operation")
a = parser.parse_args()

def resize(src):
    height,width,_ =src.shape
    dst = src
    if height != width:
        if a.pad: #如果需要padding，以较长边padding
            size = max(height,width)
            oh = (size - height) // 2
            ow = (size - width) // 2
            dst = im.pad(image = dst,offset_height = oh, offset_width = ow,target_height = size,target_width = size)
         else: #否则，以较短边修剪image
             size = min(height,width)
             oh = (height - size) //2
             ow = (width - size) //2
             dst = im.crop(image=dst,offset_height=oh,offset_width=ow,target_height=size,target_width=size)
    assert(dst.shape[0] == dst.shape[1])

    size,_,_ = dst.shape
    if size > a.size:#如果修剪后的image尺寸与a.size不符合，则进行相应的修改
        dst = im.downscale(images=dst,size=[a.size,a.size])
    elif size < a.size:
        dst = im.upscale(images=dst,size=[a.size,a.size])
    return dst

def blank(src):
    height,width,_ = src.shape
    if height != width: #如果长宽不等，引发错误
        raise Exception("non - quare image")

    image_size = height
    size = int(image_size * 0.3)
    offset = int(image_size / 2 - size /2) #确定blank的初始坐标
    dist = src #生成src副本
    dist[offset:offset+size,offset:offset+size] = np.ones([size,size,3]) #定义blank区间像素为1
    return dist

def combine(src,src_path):
    if a.b_dir is None:
        raise Exception("missing b_dir")
    basename,_ = os.path.splitext(os.path.basename(src_path)) #取出b的name
    for ext in ['.jpg','.png']:
        sibling_path = os.path.join(a.b_dir,basename + ext) #b完整文件名
        if os.path.existing(sibling_path):
            sibling = im.load(sibing_path) #如果sibling_path存在，下载该文件
            break
        else:
            raise Exception("could not find sibling image for" + src_path)
    height,width,_ = src.shape
    if sibling.shape[0] != height or sibling.shape[1] != width: #如果A,B图像size不一样，保存
        raise Exception("differing sizes")

    if src.shape[2] == 1: #若果为grayimage，转为rgb
        src = im.grayscale_to_rgb(images = src)

    if sibling.shape[2] == 1:
        sibling = im.grayscale_to_rgb(images = sibling)

    if src.shape[2] == 4:
        src = src[:,:,3]

    if sibling.shape[2] == 4:
        sibling = sibling[:,:,3]

    return np.concatenate([src,sibling],axis=1) #返回二者合并项

def grayscale(src):
    return im.grayscale_to_rgb(images=im.rgb_to_grayscale(images=src)) #返回rgb图像？？？

def blur(src,scale=4): #通过缩小在放大的方式，对Image进行模糊处理；将image_blur和原image通过combine结合后，即可作为训练数据，输入pix2pix训练DCGAN，训练好的DCGAN中的G网络，当输入一个Image_blur后，可以输出一个image_clear；
    height,width,_ = src.shape
    height_down = height // scale
    width_down = width // scale
    dst = im.downscale(images=src,size=[height_down,width_down])
    dst = im.upscale(images=dst,size=[height,width])
    return dst

net = None
def run_caffe(src): #????
    # lazy load caffe and create net
    global net
    if net is None:
        # don't require caffe unless we are doing edge detection
        os.environ["GLOG_minloglevel"] = "2" # disable logging from caffe
        import caffe
        # using this requires using the docker image or assembling a bunch of dependencies
        # and then changing these hardcoded paths
        net = caffe.Net("/opt/caffe/examples/hed/deploy.prototxt", "/opt/caffe/hed_pretrained_bsds.caffemodel", caffe.TEST)
        
    net.blobs["data"].reshape(1, *src.shape)
    net.blobs["data"].data[...] = src #data[...]???
    net.forward() #执行前向传播
    return net.blobs["sigmoid-fuse"].data[0][0,:,:]

def edges(src): #？？？
    import scipy.io
    src = src * 255 #转为255 的image
    border = 128 #用于padding
    src = src[:,:,:3] #保证channel=3
    src = np.pad(src,((border,border),(border,border),(0,0)),"reflect")
    src = src[:,:,::-1] #为什么要倒序？？？
    src -= np.array((104.00698793, 116.66876762, 122.67891434)) #???
    src = src.transpose((2,0,1)) #为啥子要转置啊？？？

    fuse = edge_pool.apply(run_caffe,[src]) #edge_pool为一个进程池，对其执行run_caffle
    fuse = fuse[border:-border,border:-border] #？？？

    with tempfile.NamedTemporaryFile(suffix = ".png") as png_file,tempfile.NamedTemporaryFile(suffix=".mat") as mat_file: #创建2个临时文件
        scipy.io.savemat(mat_file.name,{"input":fuse}) #将fuse存入临时文件mat_file

        octave_code = r"""
E = 1-load(input_path).input;
E = imresize(E, [image_width,image_width]);
E = 1 - E;
E = single(E);
[Ox, Oy] = gradient(convTri(E, 4), 1);
[Oxx, ~] = gradient(Ox, 1);
[Oxy, Oyy] = gradient(Oy, 1);
O = mod(atan(Oyy .* sign(-Oxy) ./ (Oxx + 1e-5)), pi);
E = edgesNmsMex(E, O, 1, 5, 1.01, 1);
E = double(E >= max(eps, threshold));
E = bwmorph(E, 'thin', inf);
E = bwareaopen(E, small_edge);
E = 1 - E;
E = uint8(E * 255);
imwrite(E, output_path);
"""

        config = dict(
            input_path="'%s'" % mat_file.name,
            output_path="'%s'" % png_file.name,
            image_width=256,
            threshold=25.0 / 255.0,
            small_edge=5,
        )

        args = ["octave"]
        for k,v in config.items():
            args.extend(["--eval","%s=%s;" % (k,v)])

        args.extend(["--eval",octave_code])
        try:
            subprocess.check_output(args,stderr=subprocess.STDOUT) #父进程等待子进程输出
        except subprocess.CalledProcessError as e:
            print("octave failed")
            print("returncode:",e.returncode)
            print("output:",e.output)
            raise
        return im.load(png_file.name) #返回image数据

def process(src_path, dst_path): #通过Process()函数，根据命令行参数，对image进行相应处理
    src = im.load(src_path)

    if a.operation == "grayscale":
        dst = grayscale(src)
    elif a.operation == "resize":
        dst = resize(src)
    elif a.operation == "blank":
        dst = blank(src)
    elif a.operation == "combine":
        dst = combine(src, src_path)
    elif a.operation == "edges":
        dst = edges(src)
    elif a.operation == "blur":
        dst = blur(src)
    else:
        raise Exception("invalid operation")

    im.save(dst, dst_path)


complete_lock = threading.Lock() #锁定
start = None
num_complete = 0
total = 0

def complete(): #???
    global num_complete,rate,last_complete #全局变量

    with complete_lock: #???
        num_complete += 1
        now = time.time()
        elapsed = now - start
        rate = num_complete / elapsed
        if rate > 0:
            remaining = (total - num_complete) / rate
        else:
            remaining = 0

        print("%d/%d complete  %0.2f images/sec  %dm%ds elapsed  %dm%ds remaining" % (num_complete, total, rate, elapsed // 60, elapsed % 60, remaining // 60, remaining % 60))

        last_complete = now
        
def main():
    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    src_paths = [] #输入
    dst_paths = [] #输出
    #二者用于创建 train_pair
    skipped = 0
    for src_path in im.find(a.input_dir):
        name,_ = os.path.splitext(os.path.basename(src_path)) #文件名
        dst_path = os.path.join(a.output_dir,name + ".png") #创建输出文件全路径
        if os.path.exists(dst_path): #如果输出已经存在，计数
            skipped += 1
        else:
            src_paths.append(src_path)
            dst_paths.append(dst_path)

    print("skipping %d files that already exist" % skipped)

    global total
    total = len(src_paths) #input总数

    print("processing %d files" % total)

    global start
    start = time.time()

    if a.operation == "edges":
        global edge_pool
        edge_pool = multiprocessing.Pool(a.workers) #进程池

    if a.workers == 1: #如果所用进程为1
        with tf.Session() as sess:
            for src_path,dst_path in zip(src_paths,dst_paths):
                process(src_path,dst_path)
                complete() #每进行一次for循环，complete()就会创建thread_lock，保证该进程进行不中断？？？，在进程执行过程中，会记录处理image的数量，及处理时间，剩余时间等内容，并返回
    else:
        queue = tf.train.input_producer(zip(src_paths,dst_paths),shuffle=False,num_epochs=1) #采用多进程
        dequeue_op = queue.dequeue()

        def worker(coord):
            with sess.as_default():
                while not coord.should_stop(): #进程未终止时
                    try:
                        src_path,dst_path = sess.run(dequeue_op) #出队
                    except tf.errors.OutOfRangeError:
                        coord.request_stop()
                        break

                    process(src_path,dst_path) #执行process()
                    complete()

        local_init_op = tf.local_variables_initializer() #进行变量初始化，为什么是局部变量？？？  什么时候执行全局初始化，什么时候执行local初始化？？？
        with tf.Session() as sess:
            sess.run(local_init_op)

            coord = tf.train.Coordinator() #进行进程管理
            threads = tf.train.start_queue_runners(coord=coord)
            for i in range(a.workers):
                t = threading.Thread(target=worker,args=(coord,)) #创造多个进程a.workers，执行process(src_path,dst_path)
                t.start() #开始执行
                threads.append(t)
            try:
                coord.join(threads)
            except KeyboardInterrupt:
                coord.request_stop()
                coord.join(threads)
    
main()    
    #main()函数主要执行process()操作；
    #首先获得：src_path,dst_path；
    #然后根据a.workers数量，确定采用多线程，还是单线程执行process()操作
        
        

    

    

        
        
    
