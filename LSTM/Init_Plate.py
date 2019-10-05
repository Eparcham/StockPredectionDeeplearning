import numpy as np
import pandas as pd
import tensorflow as tf
import random
from sklearn.preprocessing import MinMaxScaler

class initailztion(object):
    def __init__(self): 
       data = pd.read_excel('Shiraz Petr.-a.xls')
       data = data[['Close','Open','High','Low','Vol','final']].values;
       self.data = data
       self.lenday = 9;
       self.selectFeature = 0;  # which is for class
       self.featureindex = []
       self.signthr = 1;
       self.keep_prob  = 0.8;
       self.num_layers = 1;
       self.lstm_size = 128;
       self.showOut = 100;
       self.model_plots_dir = 'img/'
       ## fuature extraction code
       self.data = initailztion.indicator_sma_ema(self,self.data,self.lenday)
       self.data = self.data[self.lenday:,:]
       self.loop = 0;
       self.global_step = 0;
       self.num_inputs=self.data.shape[1];
       self.size_data = self.data.shape[0];
       self.PerTrainDay = 0.9;
       self.see_days = 30;
       flag = 1;
       step = 0;
       while flag:
           div_ = np.mod(self.size_data,self.see_days);
           if div_:
               self.size_data-=1;
               step+=1;
           else:
               flag = 0;
       
       self.data = self.data[-self.size_data:,:]  
       self.data,self.scaler =initailztion.normalization(self.data)
       self.train_see_days = np.int(np.round(self.PerTrainDay * self.see_days));
       self.test_see_days = self.see_days - self.train_see_days;
       self.labels_size = self.test_see_days;
       self.test = self.data[-self.see_days:,:];
       self.data = self.data[:-self.see_days,:]
       batch_size = 64;
       num_batches = int(self.data.shape[0]) // batch_size
       if batch_size * num_batches < self.data.shape[0]:
            num_batches += 1
            
       self.batch_size = num_batches;
       self.learning_rate = tf.placeholder(tf.float32, None, name="learning_rate")
       self.learning_rate_=0.001
       self.learning_rate_decay = 0.99999995
       self.ThroshuldPrice = [-0.05, 0.05];
       self.num_steps = 50000000
       self.ckpt_dir="model/parcham98.ckpt"
       self.SAVE_INTERVAL = 500;
       self.W_noncorect = 1.0;
       self.start_point = 0;
       self.X = tf.placeholder(dtype=tf.float32, shape=[None,self.train_see_days,self.num_inputs])
       self.Y = tf.placeholder(dtype=tf.float32, shape=[None, self.test_see_days])
       self.lastdayPrice = tf.placeholder(dtype=tf.float32, shape=[None,2])
       
    def genratenew_randum_loop(self,n):
#        self.start_point=np.random.randint(n)  
        self.loop+=1;
        self.start_point = 0;
        return self
    
    def data_extraction(self):
#        self.train_see_days=np.random.randint(self.MinTrainSize,self.MaxTrainSize);
        x_batch = np.zeros([self.batch_size,self.train_see_days,self.num_inputs])
        y_batch = np.zeros([self.batch_size,self.test_see_days])
        x_test = np.zeros([1,self.train_see_days,self.num_inputs])
        y_test = np.zeros([1,self.test_see_days])
        xx_test = np.zeros([1,self.train_see_days,self.num_inputs]);
        last_batch = np.zeros([self.batch_size,2])
        n = self.data.shape[0];
        end_loop = n-(self.train_see_days + self.test_see_days);
        retrcet_loop = n-(self.train_see_days + self.test_see_days);   
        batch_indices = np.arange(0,self.batch_size)
        random.shuffle(batch_indices)
        for i in batch_indices:
            if self.start_point>end_loop:
               self = initailztion.genratenew_randum_loop(self,retrcet_loop)   
               
            x_batch[i,:,:] = self.data[self.start_point:(self.start_point + self.train_see_days),:] # all feature open close and ...
            
            self.start_point = self.start_point + self.train_see_days;
            
            y_batch[i,:]=self.data[self.start_point:(self.start_point + self.test_see_days),self.selectFeature] # just price 
            
            last_batch[i,:] = [self.data[self.start_point-1,self.selectFeature] + self.data[self.start_point-1,self.selectFeature] * self.ThroshuldPrice[0],self.data[self.start_point-1,self.selectFeature] + self.data[self.start_point-1,self.selectFeature] * self.ThroshuldPrice[1]]
            self.start_point = self.start_point + self.test_see_days;
        
        x_test[0,:,:] = self.test[0:self.train_see_days,:] 
        y_test[0,:]   = self.test[self.train_see_days:,self.selectFeature] 
        xx_test[0,:]  = self.test[-self.train_see_days:,:] 
        
        return x_batch,y_batch,last_batch,x_test,y_test,xx_test;
    
    def feature_extraction(data):
        return 
        
    def indicator_sma_ema(self,data,len_day):
        Multiplier =  2.0 / (len_day + 1.0);
        fea = np.zeros([data.shape[0],2]);
        matrix = np.zeros([data.shape[0] , 2 + data.shape[1]])
        for i in range(len_day,data.shape[0]):
            fea[i,0] = np.mean(data[(i - len_day):i,self.selectFeature]);
            matrix[i,0:data.shape[1]] = data[i,:];
            matrix[i,data.shape[1]] = fea[i,0]
            
            
        fea[len_day,1] = fea[len_day,0]
        matrix[len_day,data.shape[1] + 1] = fea[len_day,1];
        for i in range(len_day+1,data.shape[0]):
            fea[i,1] = (data[i,self.selectFeature] - fea[i-1,0]) * Multiplier+fea[i-1,0]
            matrix[i,data.shape[1] + 1] = fea[i,1];
        return matrix
    
    def normalization(data):
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        fea = scaler.fit_transform(data)
#      closing_price = scaler.inverse_transform(closing_price) use in test time
        return fea,scaler;
        
        
        