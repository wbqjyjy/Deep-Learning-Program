import tensorflow as tf
from glob import glob
import os
import argparse
import logging
from PIL import Image
import traceback

def glob_all(dir_path):
    pic_list = glob(os.path.join(dir_path,"*.jpg")) #取出与 directory 相似的所有文件名
    inside = os.listdir(dir_path) #列出dir_path中的所有目录
    for dir_name in inside:
        if os.path.isdir(os.path.join(dir_path,dir_name)): #如果是directory
            pic_list.entend(glob_all(os.path.join(dir_path,dir_name))) #利用递归,将directory中所有文件取出
    return pic_list #返回的是所有file_name

def parse_args():
    parser = argparse.Argumentparser() #创建类实例
    parser.add_argument('-p','--dir-path',default = 'data/') #添加命令行参数
    return parser.parse_args() #返回Namespace

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO) #日志输出INFO及以上
    args = parse_args() #namespace
    pic_all_list = glob_all(args.dir_path) #所有file_name
    for i , img_path in enumerate(pic_all_list):
        try:
            sess = tf.Session()
            with open(img_path,'rb') as f:
                img_byte = f.read()#读出二进制码
                img = tf.image.decode_jpeg(img_byte)#将img_byte解码为array
                data = sess.run(img)
                if data.shape[2] != 3:
                    print(data.shape)
                    raise Exception
            tf.reset_default_graph() #重新设置计算图
            img = Image.open(img_path)  #将img转为rgb
        except Exception:
            logging.warning('%s has broken.Delete it' % img_path)
            os.remove(img_path)
        if (i + 1) % 1000 == 0:
            logging.info('Processing %d/%d' % (i+1, len(pic_all_list)))

    
