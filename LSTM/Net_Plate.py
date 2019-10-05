import tensorflow as tf
from tensorflow.contrib import rnn

class NetWork:        
    def convolutional_network_model_Plate(self,x,is_training=False,reuse=False):
        with tf.variable_scope('convolutional_network_model_Plate'): 
#            x = tf.contrib.layers.flatten(x)
            x = tf.layers.dense(x,2048,activation=tf.nn.relu)#tf.contrib.layers.flatten(x) 
            x = tf.layers.dense(x,1024,activation=tf.nn.relu)
#            x = tf.contrib.layers.flatten(x)
            x = tf.layers.dense(x,512,activation=tf.nn.relu)
            x = tf.layers.dense(x,256,activation=tf.nn.relu)

#            in_shp = x.get_shape().as_list()
#            x = tf.reshape(x, [-1,in_shp[1]*in_shp[2]])
#            x = x.set_shape((None,None,256))
#            x = tf.reshape(x, [self.batch_size,size * 256])
#            x = tf.reshape(x,[-1,1])
             
            x = tf.contrib.layers.flatten(x) 
            output = tf.layers.dense(x,self.labels_size,activation=tf.nn.relu)

            

        return output
    
    def _create_one_cell(self):
            lstm_cell = tf.contrib.rnn.LSTMCell(self.lstm_size, state_is_tuple=True)
            if self.keep_prob < 1.0:
                lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)
            return lstm_cell
    
    def RNN(self,x,is_training=False,reuse=True):
        num_hidden_1  = 128;
#        num_hidden_2  = 512;
#        num_hidden_3  = 256
#        x = tf.unstack(x,8,self.train_see_days)
#        input_shape = x.get_shape().as_list()
#        ndim = len(input_shape)
#        axis = [1, 0] + list(range(2,ndim))
#        data = tf.transpose(x,(axis))
        
#        pars = tf.reshape(x,[-1,hps.z_size,2])    
        x = tf.unstack(x,self.train_see_days,1)
#        x = tf.unstack(data)
        
#        lstm_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(num_hidden_1, activation=tf.nn.relu),rnn.BasicLSTMCell(num_hidden_2, activation=tf.nn.relu),rnn.BasicLSTMCell(num_hidden_3, activation=tf.nn.relu)])
    
        # Define a lstm cell with tensorflow
        lstm_cell = rnn.BasicLSTMCell(num_hidden_1)
    
        # Get lstm cell output
        outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

#        x = tf.layers.dense(outputs[-1],512,activation=tf.nn.relu)

#        x = tf.layers.dense(outputs[-1],1024,activation=tf.nn.relu)
#        x = tf.layers.dense(x,512,activation=tf.nn.relu)
#        x = tf.layers.dense(x,256,activation=tf.nn.relu)
        output = tf.layers.dense(outputs[-1],self.labels_size,activation=None)
   
        return  output
    
    def new_Lsm(self,x):
        
        cell = tf.contrib.rnn.MultiRNNCell([NetWork._create_one_cell(self) for _ in range(self.num_layers)],state_is_tuple=True) if self.num_layers > 1 else NetWork._create_one_cell(self)
        val, state_ = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32, scope="dynamic_rnn")
        val = tf.transpose(val, [1, 0, 2])
        last = tf.gather(val, int(val.get_shape()[0]) - 1, name="lstm_state")
#        ws = tf.Variable(tf.truncated_normal([self.lstm_size,self.labels_size]), name="w")
#        bias = tf.Variable(tf.constant(0.1, shape=[self.labels_size]), name="b")
#        output = tf.matmul(last, ws) + bias
#        
        output = tf.layers.dense(last,self.labels_size,activation=None)
        return output