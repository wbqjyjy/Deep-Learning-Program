"""Converts image data to TFRecords file format with Example protos.
The image data set is expected to reside in JPEG files located in the
following directory structure.
  data_dir/label_0/image0.jpeg
  data_dir/label_0/image1.jpg
  ...
  data_dir/label_1/weird-image.jpeg
  data_dir/label_1/my-image.jpeg
  ...
where the sub-directory is the unique label associated with these images.
This TensorFlow script converts the training and evaluation data into
a sharded data set consisting of TFRecord files
  train_directory/train-00000-of-01024
  train_directory/train-00001-of-01024
  ...
  train_directory/train-00127-of-01024
and
  validation_directory/validation-00000-of-00128
  validation_directory/validation-00001-of-00128
  ...
  validation_directory/validation-00127-of-00128
where we have selected 1024 and 128 shards for each data set. Each record
within the TFRecord file is a serialized Example proto. The Example proto
contains the following fields:
  image/encoded: string containing JPEG encoded image in RGB colorspace
  image/height: integer, image height in pixels
  image/width: integer, image width in pixels
  image/colorspace: string, specifying the colorspace, always 'RGB'
  image/channels: integer, specifying the number of channels, always 3
  image/format: string, specifying the format, always'JPEG'
  image/filename: string containing the basename of the image file
            e.g. 'n01440764_10026.JPEG' or 'ILSVRC2012_val_00000293.JPEG'
  image/class/label: integer specifying the index in a classification layer. start from "class_label_base"
  image/class/text: string specifying the human-readable version of the label
    e.g. 'dog'
If you data set involves bounding boxes, please look at build_imagenet_data.py.
"""
from __future__ import absolute_import #导入“绝对导入”
from __future__ import division #导入精确除法
from __future__ import print_function #导入python3的print()形式

from datetime import datetime
import os
import random
import sys
import threading

import numpy as np
import tensorflow as tf
import logging

def _int64_feature(value):
    if not isinstance(value,list): #判断value是否为list类型，如果不是，则将value变为list [value]；
        value = [value]
    return tf.train.Feature(int64_list = tf.train.Int64List(value=value)) #将value转换为int64list的格式(feature value格式转换）

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]) #将value转换为ByteList格式

def _convert_to_example(filename,image_buffer,label,text,height,width):
    """Build an Example proto for an example.
    Args:
      filename: string, path to an image file, e.g., '/path/to/example.JPG'
      image_buffer: string, JPEG encoding of RGB image
      label: integer, identifier for the ground truth for the network
      text: string, unique human-readable, e.g. 'dog'
      height: integer, image height in pixels
      width: integer, image width in pixels
    Returns:
      Example proto
    """
    colorspace = 'RGB'
    channels = 3
    image_format = 'JPEG'

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height':_int64_feature(height),
        'image/width':_int64_feature(width),
        'image/colorspace':_bytes_feature(colorspace),
        'image/channels':_int64_feature(channels),
        'image/class/labels':_int64_feature(label),
        'image/class/text':_bytes_feature(text),
        'image/format':_bytes_feature(image_format),
        'image/filename':_bytes_feature(os.path.basename(filename)),
        'image/encoded':_bytes_feature(image_buffer)}))
    return example

class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""
    def __init__(self):
        self._sess = tf.Session() #创造一个会话，来进行image coding
        self._png_data =tf.placeholder(dtype=tf.string) #Tensor变量
        image =tf.image.decode_png(self._png_data,channels=3)#解码png图像
        self._png_to_jpeg = tf.image.encode_jpeg(image,format='rgb',quality=100) #将image转为jpeg格式 : tf.image.encode_jpeg()
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string) #解码Jpeg
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data,channels=3) #将image.jpeg解码 ： tf.image.decode_jpeg()

    def png_to_jpeg(self,image_data):
        return self._sess.run(self._png_to_jpeg,feed_dict={self._png_data:image_data})

    def decode_jpeg(self,image_data):
        image = self._sess.run(self._decode_jpeg,feed_dict={self._decode_jpeg_data:image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3  #判断是否为真，为假时，raise
        return image

def _is_png(filename):
        """Determine if a file contains a PNG format image.
        Args:
        filename: string, path of the image file.
        Returns:
        boolean indicating if the image is a PNG.
        """
        return '.png' in filename

def _process_image(filename,coder):
        """Process a single image file.
        Args:
        filename: string, path to an image file e.g., '/path/to/example.JPG'.
        coder: instance of ImageCoder to provide TensorFlow image coding utils.
        Returns:
        image_buffer: string, JPEG encoding of RGB image.
        height: integer, image height in pixels.
        width: integer, image width in pixels.
        """
        with open(filename,'r') as f:
            image_data = f.read() #读取image file
        if _is_png(filename):
            logging.info('Converting PNG to JPEG for %s' % filename)
            image_data = coder.png_to_jpeg(image_data)
        image = coder.decode_jpeg(image_data) #将ipeg解码
        assert len(image.shape) == 3
        height = image.shape[0]
        width = image.shape[1]
        assert image.shape[2] == 3 #RGB channel
        return image_data, height, width #返回jpeg image, image height, image width

def _process_image_files_batch(coder, thread_index, ranges, name, filenames, texts, labels, num_shards, command_args): # thread_index：指定各个batch的index
        """Processes and saves list of images as TFRecord in 1 thread.
        Args:
        coder: instance of ImageCoder to provide TensorFlow image coding utils.
        thread_index: integer, unique batch to run index is within [0, len(ranges)).
        ranges: list of pairs of integers specifying ranges of each batches to
        analyze in parallel.
        name: string, unique identifier specifying the data set
        filenames: list of strings; each string is a path to an image file
        texts: list of strings; each string is human readable, e.g. 'dog'
        labels: list of integer; each integer identifies the ground truth
        num_shards: integer number of shards for this data set.
        """
        num_threads = len(ranges) #the number of batches
        assert not num_shards % num_threads #num_shards：将数据分为若干块，每块为一个tfrecord，要保证各个batch拥有同等数量的shards，即num_shards能够整除num_threads
        num_shards_per_batch = int(num_shards/num_threads)
        shard_ranges = np.linspace(ranges[thread_index][0],ranges[thread_index][1],num_shards_per_batch + 1).astype(int) #返回每个batch的各个shard的range，如：返回[1,4,7]，则shard1range为1-4 sample,shard2range为4-7 sample。
        num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0] #每个batch中sample数量
        counter = 0
        for s in xrange(num_shards_per_batch):
            shard = thread_index * num_shards_per_batch + s # 每个batch中，shard的标号
            output_filename = '%s_%s_%.5d-of-%.5d.tfrecord' % (command_args.dataset_name,name,shard,num_shards) # train-00002-of-00010 编号模式
            output_file = os.path.join(command_args.output_directory,output_filename)
            writer = tf.python_io.TFRecordWriter(output_file) #构建一个tfrecord 文件

            shard_counter = 0
            files_in_shard = np.arange(shard_ranges[s],shard_ranges[s+1],dtype=int) #在batch thread_index中第s个shard的sample的index list
            for i in files_shard:
                filename = filenames[i]         
                label = labels[i]
                text = texts[i]

                _image_buffer,height,width = _process_image(filename,coder)

                example = _convert_to_example(filename,image_buffer,label,text,height,width) #将file转化为example协议块

                writer.write(example.SerializeToString()) #example.SerializeToString()将example中的map压缩为二进制，写入tfrecord file
                            
                shard_counter += 1
                            
                counter += 1

                if not counter % 1000:
                    logging.info('%s [thread %d]:Processed %d of %d images in thread batch.' % (datetime.now(),thread_index,counter,num_files_in_thread))
                    sys.stdout.flush() #刷新缓冲区
            writer.close() #batch中的一个shard 文件写完
            logging.info('%s [thread %d]: Wrote %d images to %s' % (datetime.now(),thread_index,shard_counter,output_file))
            sys.stdout.flush()
            shard_counter = 0
        logging.info('%s[thread %d]:Wrote %d images to %d shards' % (datetime.now(),thread_index,counter,num_files_in_thread))
        sys.stdout.flush()

def _process_image_files(name,filenames,texts,labels,num_shards,command_args):
        """Process and save list of images as TFRecord of Example protos.
        Args:
        name: string, unique identifier specifying the data set
        filenames: list of strings; each string is a path to an image file
        texts: list of strings; each string is human readable, e.g. 'dog'
        labels: list of integer; each integer identifies the ground truth
        num_shards: integer number of shards for this data set.
        """
        assert len(filenames) == len(texts)
        assert len(filenames) == len(labels)
        spacing = np.linspace(0,len(filenames),command_args.num_threads + 1).astype(np.int) #spacing代表各个batch的间隔
        ranges = []
        for i in xrange(len(spacing) - 1):
            ranges.append([spacing[i],spacing[i+1]]) #range[i]代表各个batch的sample区间
        logging.info('Launching %d threads for spacings: %s' % (command_args.num_threads,ranges)) #launch a thread for each batch
        sys.stdout.flush()

        coord = tf.train.Coordinator() #创建线程管理器对象
        coder = ImageCoder()

        threads = []
        for thread_index in xrange(len(ranges)):
            args =(coder,thread_index,name,ranges,filenames,texts,labels,num_shards,command_args)
            t = threading.Thread(target=_process_image_files_batch,args=args) #创建线程对象，在单独的线程中运行该程序（多线程操作）
            t.start()
            threads.append(t) #各个batch的线程列表

        coord.join(threads)
        logging.info('%s:Finished writing all %d images in dataset' % (datetime.now(),len(filenames)))
        sys.stdout.flush()

def _find_image_files(data_dir,labels_file,command_args):
    """Build a list of all images files and labels in the data set.
    Args:
      data_dir: string, path to the root directory of images.
        Assumes that the image data set resides in JPEG files located in
        the following directory structure.
          data_dir/dog/another-image.JPEG
          data_dir/dog/my-image.jpg
        where 'dog' is the label associated with these images.
      labels_file: string, path to the labels file.
        The list of valid labels are held in this file. Assumes that the file
        contains entries as such:
          dog
          cat
          flower
        where each line corresponds to a label. We map each label contained in
        the file to an integer starting with the integer 0 corresponding to the
        label contained in the first line.
    Returns:
      filenames: list of strings; each string is a path to an image file.
      texts: list of strings; each string is the class, e.g. 'dog'
      labels: list of integer; each integer identifies the ground truth.
    """
    logging.info('Determining list of input files and labels from %s' % data_dir)
    unique_labels = [l.strip() for l in tf.gfile.FastGFile(labels_file,'r').readlines()] #tf.gfile.FastGFile()实现对图片的读取

    labels = []
    filenames= []
    texts = []

    #leave label index 0 empty as a background class
    label_index = command_args.class_label_base #???  初始化一个label_index的值
    for text in unique_labels:
        jpeg_file_path = '%s/%s/*' % (data_dir,text)
        matching_files = tf.gfile.Glob(jpeg_file_path) #获取匹配pattern的文件，并以列表形式返回
        labels.extend([label_index] * len(matching_files)) #各个类别的label index list
        texts.extend([text] * len(matching_files))
        filenames.extend(matching_files)

        if not label_index % 100:
            logging.info('finished finding files in %d of %d classes' % (label_index,len(labels)))
        label_index += 1

    shuffled_index = range(len(filenames))
    random.seed(12345)
    random.shuffle(shuffled_index)

    filenames = [filenames[i] for i in shuffled_index]
    texts = [texts[i] for i in shuffled_index]
    labels = [labels[i] for i in shuffled_index]

    logging.info('found %d jped files across %d labels inside %s' % (len(filenames),len(unique_labels),data_dir))
    return filenames,texts,labels

def _process_dataset(name,directory,num_shards,labels_file,command_args):
    """Process a complete data set and save it as a TFRecord.
    Args:
      name: string, unique identifier specifying the data set.
      directory: string, root path to the data set.
      num_shards: integer number of shards for this data set.
      labels_file: string, path to the labels file.
    """
    filenames,texts,labels = _find_image_files(directory,labels_file,command_args)
    _process_image_files(name,filenames,texts,labels,num_shards,command_args)

def check_and_set_default_args(command_args):
    if not(hasattr(command_args,'train_shards')) or command_args.train_shards is None:
        command_args.train_shards = 5
    if not(hasattr(command_args,'validation_shards')) or command_args.validation_shards is None:
        command_args.validation_shards = 5
    if not(hasattr(command_args,'num_threads')) or command_args.num_threads is None:
        command_args.num_threads = 5
    if not(hasattr(command_args,'class_label_base')) or command_args.class_label_base is None:
        command_args.class_label_base = 0
    if not(hasattr(command_args,'dataset_name')) or command_args.dataset_name is None:
        command_args.dataset_name = ''
    assert not command_args.train_shards % command_args.num_threads,('Please make the command_args.num_threads commensurate with command_args.train_shards')
    assert not command_args.validation_shards % command_args.num_threads,('Please make the command_args.num_threads commensurate with ''command_args.validation_shards')
    assert command_args.train_directory is not None
    assert command_args.validation_directory is not None
    assert command_args.labels_file is not None
    assert command_args.output_directory is not None

def main(command_args):
    """
    command_args:需要有以下属性：
    command_args.train_directory  训练集所在的文件夹。这个文件夹下面，每个文件夹的名字代表label名称，再下面就是图片。
    command_args.validation_directory 验证集所在的文件夹。这个文件夹下面，每个文件夹的名字代表label名称，再下面就是图片。
    command_args.labels_file 一个文件。每一行代表一个label名称。
    command_args.output_directory 一个文件夹，表示最后输出的位置。
    command_args.train_shards 将训练集分成多少份。
    command_args.validation_shards 将验证集分成多少份。
    command_args.num_threads 线程数。必须是上面两个参数的约数。
    command_args.class_label_base 很重要！真正的tfrecord中，每个class的label号从多少开始，默认为0（在models/slim中就是从0开始的）
    command_args.dataset_name 字符串，输出的时候的前缀。
    图片不可以有损坏。否则会导致线程提前退出。
    """
    check_and_set_default_args(command_args)
    logging.info('Saving results to %s' % command_args.output_directory)

    _process_dataset('validation',command_args.validation_directory,command_args.validation_shards,command_args.labels_file,command_args)
    _process_dataset('train',command_args.train_directory,command_args.train_shards,command_args.labels_file,command_args)

                            
        
                            
        
        
        
                            
                            
                            
                            
                            









    
