import tensorflow as tf
import numpy as np
import NetWork_Genrator 
import NetWork_Dscremnt
import Init_Plate 


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

#def loss_func(logits_in, labels_in): 
#	return tf.reduce_sum(tf.abs(logits_in - labels_in)) 
def loss_func(logits_in, labels_in): 
	return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits( 
						logits = logits_in, labels = labels_in))
    
def train(self,sess_1,isRestore):    
    self.G = NetWork_Genrator.NetWork_Genrator.Genrator_Network(self,self.X,reuse=False)
    self.D_logits_real = NetWork_Dscremnt.NetWork_Dscremnt.Dscremnt_Network(self,self.Y,reuse=False)
    self.D_logits_fake = NetWork_Dscremnt.NetWork_Dscremnt.Dscremnt_Network(self,self.G,reuse=True)
    
    D_real_loss = loss_func(self.D_logits_real, tf.ones_like(self.D_logits_real)*0.9) 
    D_fake_loss = loss_func(self.D_logits_fake, tf.zeros_like(self.D_logits_fake)) 
    
    self.loss_D = D_real_loss + D_fake_loss       
    self.loss_G =  loss_func(self.D_logits_fake, tf.ones_like(self.D_logits_fake))   #+ self_1.landa *tf.reduce_mean(lossL)#tf.reduce_sum(tf.abs(self_1.g_1-self_1.ground_truth32))  


#    self.mse,self.loss_sign,self.loss_th = loss_f(self,self.out,self.Y,self.lastdayPrice);
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
         self.D_ = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_D)
         self.G_ = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_G)
         
    load_network(sess_1,isRestore,self_1)
  
isRestore=0
graph_1 = tf.Graph()
sess_1 = tf.Session(graph = graph_1)
with graph_1.as_default():
    with sess_1.as_default():  
        self_1 = Init_Plate.initailztion();
        train(self_1,sess_1,isRestore)
        for e in range(self_1.num_steps):
            batch_x,batch_y,lastpriac = Init_Plate.initailztion.data_extraction(self_1)
            
            out_1,error_G= sess_1.run([self_1.G_, self_1.loss_G], feed_dict={self_1.X: batch_x})
            out_2,error_D= sess_1.run([self_1.D_, self_1.loss_D], feed_dict={self_1.X: batch_x, self_1.Y: batch_y})
            
            if e % 100 == 0:
               print('epach %06d E_Genetor is %011.05f sign E_Discremnt is %011.05f  ' % (e,error_G,error_D))
            if np.mod(e, self_1.SAVE_INTERVAL) == 0:
                 tf.train.Saver().save(sess_1,self_1.ckpt_dir)
                