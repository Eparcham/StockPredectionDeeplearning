import tensorflow as tf
from tensorflow.contrib import rnn

class NetWork_Genrator:        
    def Genrator_Network(self,x,reuse=None):
        with tf.variable_scope('Genrator_Network', reuse = reuse): 
            cell = tf.contrib.rnn.MultiRNNCell([NetWork_Genrator._create_one_cell(self) for _ in range(self.num_layers)],state_is_tuple=True) if self.num_layers > 1 else NetWork_Genrator._create_one_cell(self)
            val, state_ = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32, scope="dynamic_rnn")
            val = tf.transpose(val, [1, 0, 2])
            last = tf.gather(val, int(val.get_shape()[0]) - 1, name="lstm_state") 
            output = tf.layers.dense(last,self.labels_size,activation=None)

            return output
    
    def _create_one_cell(self):
        lstm_cell = tf.contrib.rnn.LSTMCell(self.lstm_size, state_is_tuple=True)
        if self.keep_prob < 1.0:
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)
        return lstm_cell
    
    def Genrator_Network_(self,x,reuse=False):
        with tf.variable_scope('Genrator_Network_'): 
#            x = tf.contrib.layers.flatten(x)
            x = tf.layers.dense(x,2048,activation=tf.nn.leaky_relu)#tf.contrib.layers.flatten(x) 
            x = tf.layers.dense(x,1024,activation=tf.nn.leaky_relu)
#            x = tf.contrib.layers.flatten(x)
            x = tf.layers.dense(x,512,activation=tf.nn.leaky_relu)
            x = tf.layers.dense(x,256,activation=tf.nn.leaky_relu)
            x = tf.layers.dense(x,128,activation=tf.nn.leaky_relu)
            x = tf.layers.dense(x,256,activation=tf.nn.leaky_relu)
#            x1 = tf.zeros([1,256]);
#            for i in range(self.train_see_days):
#                x1 = (x[:,i,:] + x1);
#            x1 = x1 / self.train_see_days 
#            x1 = tf.concat(x,x)
#            x = tf.stack(x,1)
#            in_shp = x.get_shape().as_list()
#            x = tf.reshape(x, [-1,in_shp[1]*in_shp[2]])
#            x = x.set_shape((None,None,256))
#            x = tf.reshape(x, [self.batch_size,size * 256])
#            x = tf.reshape(x,[-1,1])
             
            x = tf.contrib.layers.flatten(x) 
            output = tf.layers.dense(x,self.labels_size,activation=None)

            

        return output