from __future__ import absolute_import #进行绝对导入
from __future__ import division #进行精确除法
from __future__ import print_function #兼容print函数

try:
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen

import sys
import tarfile
import tempfile
import shutil

dataset = sys.argv[1]
url = "https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/%s.tar.gz" % dataset #要下载的数据集
with tempfile.TemporaryFile() as tmp: #创建类文件对象，用于临时数据保存，当该文件关闭时，数据消失
    print("downloading",url)
    shutil.copyfileobj(urlopen(url),tmp) #打开Url文件，并将其写入tmp
    print("extracting")
    tmp.seek(0)
    tar = tarfile.open(fileobj=tmp) #解压tmp中的文件
    tar.extractall() #解压文件，将其存在dataset/
    tar.close() #关闭压缩文件
    print('done')#有一个疑问？？？ 解压文件，存放在哪里，当前目录下？？？
#with语句后，temporaryfile关闭。   
    
