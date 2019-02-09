#SkipGram : woman->fell,woman->man；左边为出现的单词，右边为上下文

#导入一些需要的库
from __future__ import absolute_import #绝对导入（导入Python库）
from __future__ import division #精确除法
from __future__ import print_function #python2,3兼容

import collections
import path
import os
import random
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange
import tensorflow as tf

#下载语料库
url = 'http://mattmahoney.net/dc/'

def maybe_download(filename,expected_bytes):
    """如果filename存在，跳过下载；不存在，下载"""
    if not os.path.exists(filename):
        filename,_ = urllib.request.urlretrieve(url+filename,filename)
    statinfo = os.stat(filename) #os.stat()将文件的相关属性读出来
    if statinfo.st_size == expected_bytes:
        print('found and verified',filename)
    else:
        raise Exception('failed to verify' + filename + '.can you get to it with a browser?')
    return filename

#下载语料库并验证
filename = maybe_download('text3.zip',31344016)

#将语料库中的数据读为List
def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0]).split() #f.namelist()把压缩文件的压缩文件名以List的形式返回；f.read()读取某一个压缩文件；tf.compat.as_str()将压缩文件以string形式返回；string.split()将string按照空格分割，返回为List
    return data

vocabulary = read_data(filename) #将filename内容以list的形式返回

#制作一个单词表，将单词映射为id
#单词表大小=50000，其它未知单词定位UNK
vocabulary_size = 50000
def build_dataset(words,n_words): #words为单词表；n_words为单词数量
    count = [['UNK',-1]]
    count.extend(collections.Counter(words).most_common(n_words - 1)) #collections.Counter(words)记单词数；collections.Counter().most_common(n)将出现最频繁的前n个词以[word,count]形式返回
    dictionary = dict()
    for word,_ in count:
        dictionary[word] = len(dictionary) #key为word，value为id
    data = list()
    unk_count = 0                           
    for word in words:#words代表语料库中的单词
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count #语料库中UNK的数量
    reversed_dictionary = dict(zip(dictionary.values(),dictionary.keys()))
    return data,count,dictionary,reversed_dictionary #返回 单词id data，[word,count],{'word':index},{index:'word'}
    
data,count,dictionary,reversed_dictionary = build_dataset(vocabulary,vocabulary_size)

del vocabulary #删除原来的语料库list，节省空间
print('most common words (+UNK)',count[:5])
print('sample data',data[:10],[reversed_dictionary[i] for i in data[:10]]) #id对应的word

#使用data来制作训练集
data_index = 0

#定义函数generated_batch()，用于生成skip-gram模型用的batch
def generated_batch(batch_size,num_skips,skip_window): #batch_size：一个batch的单词对数量；num_skips：一个sample的单词对数量；skip_window：skip*2+1为一个sentence长度
    global data_index #全局变量
    assert batch_size // num_skips == 0 #确保sample数量为整数
    assert num_skips <= 2 * skip_window #确保有足有的上下文单词作为label
    batch = np.ndarray(shape=(batch_size),dtype=np.int32) #定义batch中“出现单词”的List
    labels = np.ndarray(shape=(batch_size,1),dtype=np.int32) #定义label的list
    span = 2 * skip_window + 1 #定义一个sentence的长度
    buffer = collections.deque(maxlen=span) #定义一个sentence的队列
    #向buffer中传入sentence
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    #将sample填入 batch -> labels
    for i in range(batch_size // num_skips):
        target = skip_window
        target_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in target_to_avoid:
                target = random.randint(0,span-1)
            target_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window] #batch列表
            labels[i * num_skips + j,0] = buffer[target] #label列表
        #buffer.popleft() 将buffer中第一个单词pop出去：个人理解，应该加入这个语句才对
        buffer.append(data_index) #向buffer中加入一个单词
        data_index = (data_index + 1) % len(data)
    data_index = (data_index + len(data) - span) % len(data) #???
    return batch,labels

batch,labels = generated_batch(batch_size=8,num_skips=2,skip_window=1)
for i in range(8):
    print(batch[i],reverse_dictionary[batch[i]],'->',labels[i,0],reverse_dictionary[labels[i,0]])

        
#建立模型
batch_size = 128 #batch中的单词对数
embedding_size = 128 #词嵌入向量维度
skip_window = 1
num_skips = 2

#定义validation
valid_size = 16 #拿出16个word进行验证
valid_window = 100 #从前100个高频词中选出valid_size
valid_examples = np.random.choice(valid_window,valid_size,replace=False) #从100中选16

num_sampled = 64 #构造nce损失时，噪声词的数量

graph = tf.Graph()
with graph.as_default():                                
    train_inputs = tf.placeholder(tf.int32,shape=[batch_size])
    train_labels = tf.placehoder(tf.int32,shape=[batch_size,1])
    valid_dataset = tf.constant(valid_examples,dtype=tf.int32)
    #构造词嵌入向量
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size,embedding_size],-1.0,1.0)
    #train_inputs相应的词嵌入向量
    embed = tf.nn.embedding_lookup(embeddings,train_inputs)

    #构造nce损失
    nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size,embedding_size],stddev = 1.0/math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
    loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights,
                       biases=nce_biases,
                       labels=train_labels,
                       inputs=embed,
                       num_sampled=num_sampled,
                       num_classes=vocabulary_size))
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    #validation：计算词之间相似度
    #对词嵌入向量进行归一化
    norm = tf.sqrt(tf.reduce_sum(tf.quare(embeddings),1,keep_dims=True))
    normalized_embeddings = embeddings / norm
    #查找validation的词嵌入向量
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,valid_dataset)
    #计算validation与vocabulary的相似度
    similarity = tf.matmul(valid_embeddings,normalized_embeddings,transpose_b=True)

    #变量初始化
    init = tf.global_variables_initializer()


#开始训练
num_steps = 100001

with tf.Session(graph=graph) as session:
    init.run() #全局变量初始化
    print('initialized')

    average_loss = 0
    for step in xrange(num_steps):
        batch_inputs,batch_labels = generated_batch(batch_size,num_skips,skip_window)
        feed_dict = {train_inputs:train_inpus,train_labels:batch_labels}
        _,loss_val = session.run([optimizer,loss],feed_dict = feed_dict)
        average_loss += loss_val
        #每2000步，求平均损失
        if step % 2000 == 0 :
            if step != 0:
                average_loss /= 2000
            print('average loss at step',step,':',average_loss)
            average_loss = 0
        #每10000步，进行一次验证
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in xrange(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8 #输出最相似的8个单词
                nearest = (-sim[i,:]).argsort()[1:top_k + 1]
                log_str = 'nearest to %s:' % valid_word
                for k in xrange(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = '%s %s' % (log_str,close_word)
                    print(log_str)

    final_embeddings = normalized_embeddings.eval() #归一化后的vacabulary词嵌入向量


#可视化

def plot_with_labels(low_dim_embs,labels,filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels) #要保证词嵌入向量个数>labels数
    plt.figure(figsize=(18,18))
    for i,label in enumerate(labels):
        x,y = low_dim_embs[i,:] #x,y???不是x吗？？
        plt.scatter(x,y)
        plt.annotate(label,
                     xy=(x,y),
                     xytext=(5,2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.savefig(filename)

try:
    from sklearn.manifold import TSNE
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    tsne = TSNE(perplexity=30,n_components=2,init='pca',n_iter=5000) #对词嵌入向量做降维处理
    plot_only=500 #只画500个单词的位置
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only,:]) #进行降维操作
    labels = [reverse_dictionary[i] for i in xrange(plot_only)] #获取label
    plot_with_labels(low_dim_embs,labels)
except ImportError:
    print('please install sklearn,matplotlib,and scipy to show embeddings')
        

                             
                             
            
                             
                             




                                
