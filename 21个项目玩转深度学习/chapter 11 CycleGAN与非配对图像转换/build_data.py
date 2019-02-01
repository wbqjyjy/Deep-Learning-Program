import tensorflow as tf
import random
import os

try:
    from os import scandir
except ImportError:
    from scandir import scandir

FLAGS = tf.flags.FLAGS #常与tf.app.run()联用

tf.flags.DEFINE_string('X_input_dir', 'data/apple2orange/trainA',
                       'X input directory, default: data/apple2orange/trainA')
tf.flags.DEFINE_string('Y_input_dir', 'data/apple2orange/trainB',
                       'Y input directory, default: data/apple2orange/trainB')
tf.flags.DEFINE_string('X_output_file', 'data/tfrecords/apple.tfrecords',
                       'X output tfrecords file, default: data/tfrecords/apple.tfrecords')
tf.flags.DEFINE_string('Y_output_file', 'data/tfrecords/orange.tfrecords',
                       'Y output tfrecords file, default: data/tfrecords/orange.tfrecords')

def data_reader(input_dir,shuffle=True):
    file_paths = []
    for image_file in scandir(input_dir):
        if img_file.name.endswith('.jpg') and img_file.is_file():
            file_paths.append(img_file.path)

    if shuffle:
        shuffled_index = list(range(len(file_paths)))
        random.seed(12345)
        random.shuffle(shuffled_index)

        file_paths = [file_paths[i] for i in shuffled_index]

    return file_paths

def _int64_feature(value):
    if not isinstance(value,list): #是否属于list类
        value = [value] #tf.train.Int64List()中接收的是list
    return tf.train.Feature(int64_list = tf.train.Int64List(value=value)) #返回tfrecords格式

def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value=[value]))

def _convert_to_example(file_path,image_buffer): #tfrecords中example的生成
    """Build an Example proto for an example.
  Args:
    file_path: string, path to an image file, e.g., '/path/to/example.JPG'
    image_buffer: string, JPEG encoding of RGB image
  Returns:
    Example proto
  """
    file_name = file_path.split('/')[-1]
    example = tf.train.Example(features = tf.train.Features(feature={
        'image/file_name': _bytes_feature(tf.compat.as_bytes(os.path.basename(file_name))),
        'image/encoded_image':_bytes_feature((image_buffer))
        }))
    return example

def data_writer(input_dir,output_file):
    """Write data to tfrecords
  """
    file_paths = data_reader(input_dir) #将input_dir中的filename打乱顺序后读入file_paths
    #创造tfrecords dir
    output_dir = os.path.dirname(output_file)
    try:
        os.makedirs(output_dir) #创建目录
    except os.error, e:
        pass

    images_num = len(file_paths) #input 文件的数量

    writer = tf.python_io.TFRecordWriter(output_file) #构建tfrecord step1

    for i in range(len(file_paths)):
        file_path = file_paths[i]

        with tf.gfile.FastGFile(file_path,'rb') as f:
            image_data = f.read() #读取file

        example = _convert_to_example(file_path,image_data) #构建tfrecord step2

        writer.write(example.SerializeToString()) #构建tfrecord step3

        if i % 500 == 0:
            print("Processed {}/{}.".format(i,images_num))
    print("Done.")
    writer.close() #关闭output_file step4

def main(unused_argv):
    print("convert x data to tfrecords...")
    data_writer(FLAGS.X_input_dir, FLAGS.X_output_file)
    print("convert y data to tfrecords...")
    data_writer(FLAGS.Y_input_dir, FLAGS.Y_output_file)

if __name__ == "__main__":
    tf.app.run()
    
            
    

    


























    
                            
