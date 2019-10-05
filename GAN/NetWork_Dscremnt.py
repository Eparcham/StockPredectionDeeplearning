import tensorflow as tf

class NetWork_Dscremnt: 
    def Dscremnt_Network(self,x,reuse=None):
        with tf.variable_scope('Dscremnt_Network', reuse = reuse): 
            x = tf.layers.dense(x,1024,activation=tf.nn.leaky_relu)
            x = tf.layers.dense(x,512,activation=tf.nn.leaky_relu)  
            x = tf.layers.dense(x,256,activation=tf.nn.leaky_relu) 
            output = tf.layers.dense(x,1,activation=None)
            output = tf.sigmoid(output) 

        return output