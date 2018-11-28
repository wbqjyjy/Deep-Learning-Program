# coding:utf-8
from __future__ import absolute_import
import argparse #用于解析命令行参数
import os
import logging
from src.tfrecord import main #作者编写的一个脚本语言

def parse_args():
    parser= argparse.ArgumentParser() #创建ArgumentParser()对象，这个ArgumentParser对象中会保存所有将命令行参数转为python数据类型的必要信息。
    parser.add_argument('-t','--tensorflow-data-dir',default='pic/') #定义参数
    parser.add_argument('--train-shards',default=2,type=int)
    parser.add_argument('--validation-shards',default=2,type=int)
    parser.add_argument('--num-threads',default=2,type=int)
    parser.add_argument('--dataset-name',default='satellite',type=str)
    return parser.parser_args() #返回定义的各个参数的NameSpace

if __name__ == '__main__': #如果现在执行的.py文件为主程序，就执行下列代码
    logging.basicConfig(level=logging.INFO) #配置日志的输出信息，格式等
    args = parse_args() #将函数parse_args()返回的namespace赋值给args
    args.tensorflow_dir = args.tensorflow_data_dir #将tensorflow_data_dir的参数值赋给变量args.tensorflow_dir(args.tensorflow_dir相当于又给args增添了一个参数，其参数值与tensorflow_data_dir参数的相同）
    args.train_directory = os.path.join(args.tensorflow_dir,'train') #参数train_dirctory的值为trainding data的存储路径
    args.validation_directory = os.path.join(args.tensorflow_dir,'validation') #参数validation_directory的值为validation data的存储路径
    args.output_directory = args.tensorflow_dir #将新生成的文件存储于pic/中
    args.labels_file = os.path.join(args.tensorflow_dir,'label.txt') #参数labels_file的值为label file的存储路径
    if os.path.exists(args.labels_file) is False: #如果label file路径不存在
        logging.warning('Can\'t find label.txt.Now creat it.')
        all_entries = os.listdir(args.train_directory) #将train文件夹中所有文件名称以列表的形式展开
        dirnames = []
        for entry in all_entries:
            if os.path.isdir(os.path.join(args.train_directory,entry)): #如果entry为文件夹
                dirnames.append(entry) #将文件夹名称添加到dirnames
        with open(args.labels_file,'w') as f:
            for dirname in dirnames:
                f.write(dirname + '\n') #将各个数据的类别名写入label file
    main(args) #作者自己编写的程序main()，在src.tfrecord脚本中
        
        
