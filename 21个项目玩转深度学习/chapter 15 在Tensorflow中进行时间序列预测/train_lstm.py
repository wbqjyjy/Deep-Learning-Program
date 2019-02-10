#利用LSTM模型预测“单变量时间序列”

# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import #进行绝对导入：导入Python库
from __future__ import print_function #兼容python2,3的print
from __future__ import division #进行精确除法

from os import path

import numpy as np
import tensorflow as tf

from tensorflow.contrib.timeseries.python.timeseries import estimators as ts_estimators
from tensorflow.contrib.timeseries.python.timeseries import model as ts_model
from tensorflow.contrib.timeseries.python.timeseries import NumpyReader

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

class _LSTMModel(ts_model.SequentialTimeSeriesModel):
    def __init__(self,num_units,num_features,dtype=tf.float32):
        super(_LSTMModel,self).__init__(train_output_names=['mean'],predict_output_names=['mean'],num_features=num_features,dtype=dtype) #super()，调用LSTMModel父类的init
        self._num_units = num_units
        self._lstm_cell = None
        self._lstm_cell_run = None
        self._predict_from_lstm_output = None

    def initialize_graph(self,input_statistics):
        super(_LSTMModel,self).initialize_graph(input_statistics=input_statistics)
        self._lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=self._num_units)
        self._lstm_cell_run = tf.make_template(
            name_ = 'lstm_cell',
            func_ = self._lstm_cell,
            create_scope_now_ = True)
        self._predict_from_lstm_output = tf.make_template(
            name_ = 'predict_from_lstm_output',
            func_ = lambda inputs: tf.layers.dense(inputs=inputs,units=self.num_features),
            create_scope_now_ = True)
        #tf.make_template()：给定一个任意函数,将其包装,以便它进行变量共享.
        #tf.layers.dense()：outputs = activation(inputs * kernel + bias)
    def get_start_state(self):
        return(tf.zeros([],dtype=tf.int64),tf.zeros([self.num_features],dtype=self.dtype),
               [tf.squeeze(state_element,axis=0) for state_element in self._lstm_cell_zero_state(batch_size=1,dtype=self.dtype)])

    def _transform(self,data):
        return (data - mean) / variance

    def _de_transform(self,data):
        return data * variance + mean

    def _filtering_step(self,current_times,current_values,state,predictions):
        state_from_time,prediction,lstm_state = state
        with tf.control_dependencies([tf.assert_equal(current_times,state_from_time)]):
            transformed_values = self._transform(current_values)
            predictions['loss'] = tf.reduce_mean((prediction - transformed_values) **2,axis=-1)
            new_state_tuple = (current_times,transformed_values,lstm_state)
            return (new_state_tuple,predictions)
        #tf.control_dependencies():指明某些操作执行的依赖关系

    def _prediction_step(self,current_times,state):
        _,previous_observation_or_prediction,lstm_state = state
        lstm_output, new_lstm_state = self._lstm_cell_run(
            inputs = previous_observation_or_prediction,state=lstm_state)
        next_prediction = self._predict_from_lstm_output(lstm_output)
        new_state_tuple = (current_times,next_prediction,new_lstm_state)
        return new_state_tuple,{'mean':self._de_transform(next_prediction)}

    def _imputation_step(self,current_times,state):
        """Advance model state across a gap."""
        return state

    def _exogenous_input_step(self,current_times,current_exogenous_regressors,state):
        """update model state based on exogenous regressors"""
        raise NotImplementedError(
            "Exogenous inputs are not implemented for this example")

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    #构建数据
    x=np.array(range(1000))
    noise = np.random.uniform(-0.2,0.2,1000)
    y = np.sin(np.pi * x/50) + np.cos(np.pi * x/50) + np.sin(np.pi * x/25) + noise
    data = {
        tf.contrib.timeseries.TrainEvalFeatures.TIMES:x,
        tf.contrib.timeseries.TrainEvalFeatures.VALUES:y,
        }
    #读取数据
    reader = NumpyReader(data)
    #构建train_batch
    train_input_fn = tf.contrib.timeseries.RandomWindowInputFn(
        reader,batch_size=4,window_size=100)
    #构建模型
    estimator = ts_estimators.TimeSeriesRegressor(
        model=LSTMModel(num_features=1,num_units=128),
        optimizer=tf.train.AdamOptimizer(0.001))
    #训练模型
    estimator.train(input_fn=train_input_fn,steps=2000)
    #校验模型
    evaluation_input_fn=tf.contrib.timeseries.WholeDatasetInputFn(reader)
    evaluation = estimator.evaluate(input_fn=evaluation_input_fn,steps=1)
    #预测
    (predictions,)=tuple(estimator.predict(
        input_fn=tf.contrib.timeseries.predict_continuation_input_fn(
            evaluation,steps=200)))

    #作图
    observed_times = evaluation['times'][0]
    observed = evaluation['observed'][0,:,:]
    evaluated_times = evaluation['times'][0]
    evaluated = evaluation['mean'][0]
    predicted_times=predictions['times']
    predicted=predictions['mean']

    plt.figure(figsize=(15,5))
    plt.axvline(999,linestyle='dotted',linewidth=4,color='r')
    observed_lines=plt.plot(observed_times,observed,label='observation',color='k')
    evaluated_lines=plt.plot(evaluated_times,evaluated,label='evaluation',color='g')
    predicted_lines=plt.plot(predicted_times,predicted,label='prediction',color='r')
    plt.legend(handles=[observed_lines[0],evaluated_lines[0],predicted_lines[0]],loc='upper left')
    plt.savefig('predict_result.jpg')
    
    
    
        
