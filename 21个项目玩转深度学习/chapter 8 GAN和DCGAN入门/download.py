"""
Modification of https://github.com/stanfordnlp/treelstm/blob/master/scripts/download.py
Downloads the following:
- Celeb-A dataset
- LSUN dataset
- MNIST dataset
"""

from __future__ import print_function
import os
import sys
import gzip
import json
import shutil
import zipfile
import argparse
import requests
import subprocess
from tqdm import tqdm
from six.moves import urllib

parser = argparse.ArgumentParser(description = 'Download dataset for DCGAN') #创建类实例
parser.add_argument('datasets',metavar='N',type=str,nargs='+',choices=['celebA','lsun','mnist'],help='name of dataset to download [celebA,lsun,mnist]') #添加命令行参数

def download(url,dirpath):
    filename = url.split('/')[-1] #去除split list中最后一个string
    filepath = os.path.join(dirpath,filename) #设定文件的完整路径
    u = urllib.request.urlopen(url) #设定指向url文件的指针
    f = open(filepath,'wb') #以写的方式打开filepath
    filesize = int(u.headers["Content - Length"]) #取出u.headers中content - length中的内容
    print('Downloading: %s Bytes: %s' % (filename,filesize))

    downloaded = 0
    block_sz = 8192
    status_width = 70
    while True:
        buf = u.read(block_sz) #读取block_sz长度的数据
        if not buf:
            print('')
            break #如果buf为空，说明数据已经读取完毕，退出
        else:
            print('',end = '\r')
        downloaded += len(buf) #已经下载的数据量
        f.write(buf)
        status = (("[%-" + str(status_width + 1) + "s] %3.2f%%") % ('=' * int(float(downloaded) / filesize * status_width) + '>',downloaded * 100. / filesize))
        print(status,end='')
        sys.stdout.flush() #每一秒在stdout中输出一次
    f.close()
    return filepath

def download_file_from_google_drive(id,destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session() #创建session类实例
    response = session.get(URL,params = {'id':id},stream=True) #得到URL内容
    token = get_confirm_token(response) #自定义函数：将response中的value返回

    if token:
        params = {'id':id,'confirm':token}
        response = session.get(URL,params=params,stream=True)

    save_response_content(response,destination)#自定义函数：保存response数据到destination

def get_confirm_token(response):
    for key,value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response,destination,chunk_size=32*1024):
    total_size = int(response.headers.get('content-length',0))
    with open(destination,'wb') as f:
        for chunk in tqdm(response.iter_content(chunk_size),total=total_size,unit='B',unit_scale=True,desc=destination):
            if chunk:
                f.write(chunk)

def unzip(filepath):
    print('Extracting:' + filepath)
    dirpath = os.path.dirname(filepath)
    with zipfile.ZipFile(filepath) as zf:
        zf.extractall(dirpath)
    os.remove(filepath)

def download_celeb_a(dirpath):
    data_dir = 'celebA'
    if os.path.exists(os.path.join(dirpath,data_dir)):
        print('found celeba skip')
        return

    filename, drive_id = "img_align_celeba.zip","0B7EVK8r0v71pZjFTYXZWM3F1RnM"
    save_path = os.path.join(dirpath,filename)

    if os.path.exists(save_path):
        print('[*] {} already exists'.format(save_path))
    else:
        download_file_from_google_drive(drive_id,save_path)

    zip_dir =''
    with zipfile.ZipFile(save_path) as zf:
        zip_dir = zf.namelist()[0]
        zf.extractall(dirpath)
    os.remove(save_path)
    os.rename(os.path.join(dirpath,zip_dir),os.path.join(dirpath,data_dir)) #os.rename(要修改的目录名，修改后的目录名)

def _list_categories(tag):
    url = "http://lsun.cs.princeton.edu/htbin/list.cgi?tag=' + tag
    f = urllib.request.urlopen(url)
    return json.load(f.read()) #返回网页内容

def _download_lsun(out_dir,category,set_name,tag):
    url = http://lsun.cs.princeton.edu/htbin/download.cgi?tag={tag}' \
      '&category={category}&set={set_name}'.format(**locals()) #注意format(**locals())的用法，它能够将局部变量值映射给{}
    print(url)
    if set_name == 'test':
        out_name = 'test_1mdb.zip'
    else:
        out_name = '{category}_{set_name}_1mdb.zip'.format(**locals())
    out_path = os.path.join(out_dir,out_name)
    cmd= ['curl',url,'-o',out_path]
    print('downloading',category,set_name,'set')
    subprocess.call(cmd) #接受字符串类型的变量作为命令，并调用shell去执行这个字符串

def download_lsun(dirpath):
    data_dir = os.path.join(dirpath,'lsun')
    if os.path.exists(data_dir):
        print('found lsun -skip')
        return
    else:
        os.mkdir(data_dir)

    tag = 'latest'
    categories = ['bedroom']

    for category in categories:
        _download_lsun(data_dir,category,'train',tag)
        _download_lsun(data_dir,category,'val',tag)
    _download_lsun(data_dir,'','test',tag)

def download_mnist(dirpath):
    data_dir = os.path.join(dirpath,'mnist')
    if os.path.exists(data_dir):
        print('found mnist -skip')
        return
    else:
        os.mkdir(data_dir)
    url_base = 'http://yann.lecun.com/exdb/mnist/'
    file_names = ['train-images-idx3-ubyte.gz',
                  'train-labels-idx1-ubyte.gz',
                  't10k-images-idx3-ubyte.gz',
                  't10k-labels-idx1-ubyte.gz']
    for file_name in file_names:
        url = (url_base + file_name).format(**locals())
        print(url)
        out_path = os.path.join(data_dir,file_name)
        cmd = ['curl',url,'-o-,out_path]
        print('downloading',file_name)
        subprocess.call(cmd)
        cmd = ['gzip','-d',out_path]
        print('decompressing',file_name)
        subprocess.call(cmd)

def prepare_data_dir(path = './data'):
    if not os.path.exists(path):
               os.mkdir(path)

if __name__ == '__main__':
    args = parser.parse_args()#获取命令行参数
    prepare_data_dir() #建立数据存储dir

    if any(name in args.datasets for name in ['CelebA','celebA','celebA']): #any(iterator) iterator中只有有一个True，返回True
        download_celeb_a('./data')
    if 'lsun' in args.datasets:
        download_lsun('./data')
    if 'mnist' in args.datasets:
        download_mnist('./data')
            
        

    



                    









