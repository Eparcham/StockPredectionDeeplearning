import tensorflow as tf
import numpy as np
import Net_Plate 
import Init_Plate 
import os
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

def restrct_price_throshuld(net_out,low,high):    
    
    lowRangeMask  = tf.less(net_out,low)
    HighRangeMask = tf.greater(net_out,high)
    
    mask = tf.logical_or(lowRangeMask , HighRangeMask);
    y = tf.ones_like(low)
    v = tf.reduce_sum(tf.boolean_mask(y, mask))#tf.reduce_sum(tf.boolean_mask(y, mask)))

    
    return v

def signdetection(self,L_days,R_todays,Net_days):
    
    a1  = tf.less(L_days,R_todays)
    a2  = tf.less(L_days,Net_days)
    mask = tf.not_equal(a1 , a2);
    y = tf.ones_like(Net_days)
    loss = tf.reduce_sum(tf.boolean_mask(y, mask))
#    tf.print(loss)
    
#    loss = 0;
#    for i in range(self.batch_size):
#        if L_days[i]>R_todays[i]:
#            if L_days[i,0]<Net_days[i,0]:
#                loss+=self.signthr;
#        elif L_days[i,0]<R_todays[i,0]:
#            if L_days[i,0]>Net_days[i,0]:
#                loss+=self.signthr;
    return loss;
            
def loss_f(self,net_out,real_out,Thr):
#    dif_fisrt = net_out - real_out;
    ## throshold of -0.05 +0.05
    loss_sign = 0;
    loss_th = 0;
    low  = Thr[:,0];
    high = Thr[:,1];
    lastdayprice = (low + high) / 2;
    loss_sign = signdetection(self,lastdayprice,real_out[:,0],net_out[:,0]);
    loss_th  += restrct_price_throshuld(net_out[:,0],low,high)
    index_ = 0;
    for i in range(self.test_see_days,1):
       loss_sign += signdetection(self,real_out[:,i-1],real_out[:,i],net_out[:,i]);
       low  = net_out[:,i-1] + tf.multiply(net_out[:,i-1],self.ThroshuldPrice[0])
       high = net_out[:,i-1] + tf.multiply(net_out[:,i-1],self.ThroshuldPrice[1])
       loss_th += restrct_price_throshuld(net_out[:,i],low,high)   
#       loss_ava += tf.reduce_sum(tf.abs(real_out[:,index_] - net_out[:,index_]))
       index_ +=1;
    loss_ava = tf.reduce_mean(tf.abs(real_out - net_out))
    loss_th = self.W_noncorect * loss_th
    ## finsh thr
    
#    delta = tf.constant(0.25);
#    loss1 =  tf.reduce_sum(tf.multiply(tf.square(delta),tf.sqrt(1. + (tf.square((real_out - net_out))/delta))-1.))#loss_func(self_1.g_1, (self_1.ground_truth32));
    mse = loss_ava + self.signthr * loss_sign + loss_th 
    return mse,loss_sign,loss_th

def train(self,sess_1,isRestore):    
    self.out = Net_Plate.NetWork.new_Lsm(self,self.X)
    self.mse,self.loss_sign,self.loss_th = loss_f(self,self.out,self.Y,self.lastdayPrice);
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
         self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.mse)
    load_network(sess_1,isRestore,self_1)
  
def show_image(self_1,x_test,y_test,first_test,secend_test,needpred):
    
    plt.figure(figsize=(12, 6))
    days = np.arange(self_1.see_days + self_1.test_see_days)
    
    plt.plot(days[0:self_1.train_see_days],x_test[:,self_1.selectFeature],color='b')
    plt.plot(days[self_1.see_days-self_1.test_see_days:(self_1.train_see_days +self_1.test_see_days)],y_test,color='g')
    plt.plot(days[self_1.see_days-self_1.test_see_days:(self_1.train_see_days +self_1.test_see_days)],first_test,color='r')
    if needpred:
       plt.plot(days[self_1.see_days:],secend_test,color='k')
#    plt.legend(loc='upper left', frameon=False)
    plt.xlabel("day")
    plt.ylabel("normalized price")
#                plt.ylim((min(y_test[0]), max(y_test[0])))
#        plt.ylim(-0,+15000);
    plt.grid(ls='--')
    figname = os.path.join(self_1.model_plots_dir, "{}_epoch{:02d}_step{:04d}.png".format(e, needpred, self_1.showOut))
    plt.savefig(figname, format='png', bbox_inches='tight', transparent=True)
    plt.close()
  
isRestore=0
graph_1 = tf.Graph()
sess_1 = tf.Session(graph = graph_1)

with graph_1.as_default():
    with sess_1.as_default():  
        self_1 = Init_Plate.initailztion();
        train(self_1,sess_1,isRestore)
        
        for e in range(self_1.num_steps):
            learning_rate = self_1.learning_rate_ * (self_1.learning_rate_decay ** max(float(e + 1 - self_1.learning_rate_), 0.0))
            batch_x,batch_y,lastpriac,x_test,y_test,xx_test = Init_Plate.initailztion.data_extraction(self_1)   
            train_data_feed = {
                    self_1.learning_rate: learning_rate ,
                    self_1.X: batch_x,
                    self_1.Y: batch_y,
                    self_1.lastdayPrice:lastpriac
                    }
            _,error_G1,error_G2,error_D,out_network= sess_1.run([self_1.opt,self_1.mse,self_1.loss_sign,self_1.loss_th,self_1.out],train_data_feed)
            if e % self_1.showOut == 0:
                print('epach %06d E_Genetor1 is %011.05f  E_Genetor2 is %011.05f sign E_Discremnt is %011.05f  ' % (e,error_G1,error_G2,error_D))
                first_test = sess_1.run(self_1.out, feed_dict={self_1.X: x_test,self_1.learning_rate: 0.0})
                secend_test= sess_1.run(self_1.out, feed_dict={self_1.X: xx_test,self_1.learning_rate: 0.0})
                show_image(self_1,batch_x[0,:,:],batch_y[0,:],out_network[0,:],secend_test,0)
                show_image(self_1,x_test[0,:,:],y_test[0,:],first_test[0],secend_test[0],1)
               
            if np.mod(self_1.loop,100) == 1:
#            if np.mod(e, self_1.SAVE_INTERVAL) == 0:
                 tf.train.Saver().save(sess_1,self_1.ckpt_dir)
                