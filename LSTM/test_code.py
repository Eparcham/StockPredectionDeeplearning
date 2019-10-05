import tensorflow as tf
import numpy as np
import Net_Plate 
import Init_Plate 
import matplotlib.pyplot as plt

def load_network(sess_3,isRestore,self_1):
        if isRestore:
            tf.train.Saver().restore(sess_3,self_1.ckpt_dir)
        else:
            tf.train.Saver()
        global_vars = tf.global_variables()
    
        is_var_init = [tf.is_variable_initialized(var) for var in global_vars]
        is_initialized = sess_3.run(is_var_init)
        not_initialized_vars = [var for (var, init) in
                                zip(global_vars, is_initialized) if not init]
    
        if len(not_initialized_vars):
            sess_3.run(tf.variables_initializer(not_initialized_vars))
            
def loss_f(out,Y):
    delta = tf.constant(0.25);
    mse =  tf.reduce_mean(tf.multiply(tf.square(delta),tf.sqrt(1. + (tf.square((Y - out))/delta))-1.))#loss_func(self_1.g_1, (self_1.ground_truth32));
#    mse = tf.reduce_mean(tf.squared_difference(out, Y))    
    return mse

def train(self,sess_1,isRestore):
    

    self.out = Net_Plate.NetWork.new_Lsm(self,self.X)
    
    mse = loss_f(self.out,self.Y);
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
         opt = tf.train.AdamOptimizer(self.learning_rate).minimize(mse)
    load_network(sess_1,isRestore,self_1)
    return opt,mse
    
isRestore=1
graph_1 = tf.Graph()
sess_1 = tf.Session(graph = graph_1)
with graph_1.as_default():
    with sess_1.as_default():  
        self_1 = Init_Plate.initailztion();
        opt,mse = train(self_1,sess_1,isRestore)
        x_ = np.zeros([1,self_1.train_see_days,self_1.num_inputs])
        x_out = np.zeros([1,self_1.train_see_days])
        y_ = np.zeros([1,self_1.test_see_days])
        c_ = np.zeros([1,self_1.train_see_days,self_1.num_inputs])
        add = 0;
        for i in range(self_1.train_see_days):
            x_[0,add,:] = self_1.test[i,0:self_1.num_inputs]
            x_out[0,add] = self_1.test[i,self_1.selectFeature]
            add+=1;
        add = 0;
        for i in range(self_1.train_see_days,self_1.see_days):
            y_[0,add] = self_1.test[i,self_1.selectFeature]
            add+=1;
        add = self_1.train_see_days-1;
        for i in range(self_1.see_days,(self_1.see_days-self_1.train_see_days),-1):
            c_[0,add] = self_1.test[i-1,0:self_1.num_inputs]
            add-=1;
#        add = self_1.train_see_days-1;
#        for i in range(self_1.train_see_days):
#            c_[0,add] = self_1.test[-i,0:self_1.num_inputs]
#            add-=1;
        dx_ = x_;
        dc_ = c_;
        pred_1 = sess_1.run(self_1.out, feed_dict={self_1.X: x_})
        pred_2 = sess_1.run(self_1.out, feed_dict={self_1.X: c_})
        x = np.transpose(np.arange(0,self_1.see_days + self_1.test_see_days))
        fig = plt.figure(figsize=(13, 13))
        plt.plot(x[0:self_1.see_days],self_1.test[:,self_1.selectFeature],color='y')
        #plt.plot(x[0:self_1.train_see_days],x_out[0],color='b')
        plt.plot(x[0:self_1.train_see_days],x_out[0],color='b')
        plt.plot(x[self_1.see_days-self_1.test_see_days:(self_1.train_see_days +self_1.test_see_days)],y_[0],color='g')
        plt.plot(x[self_1.see_days-self_1.test_see_days:(self_1.train_see_days +self_1.test_see_days)],pred_1[0],color='r')
        plt.plot(x[self_1.see_days:],pred_2[0],color='k')
        plt.grid(True)
        plt.show()


                