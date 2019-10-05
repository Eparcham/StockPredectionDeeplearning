import numpy as np
import pandas as pd

class initailztion(object):
    def __init__(self): 
       data = pd.read_excel('IRO3HRLZ0006.xls')
       data = data.values
       
       self.see_days = 160;
       self.train_see_days = 80;
       self.test_see_days = self.see_days - self.train_see_days;
       self.labels_size = self.test_see_days;
       self.data,max_ = initailztion.Normalization_extraction(self,data,self.see_days);
       train_start = 0
       n = self.data.shape[0]
       train_end = int(np.floor(0.8*n))
       self.num_inputs=1
       self.num_neurons=100
       self.num_outputs=1
       self.learning_rate=0.001
       self.num_iter=5000
       self.batch_size=1
       self.test = self.data[-1:, :]
       self.data = self.data[train_start:-1, :]
       self.x_train = self.data[:, 0:self.train_see_days]
       self.y_train = self.data[:,self.train_see_days: ]
       self.learning_rate = 0.0001
       self.num_steps = 3000000
       self.batch_size = 128
       self.ckpt_dir="model/parcham98.ckpt"
       self.SAVE_INTERVAL = 500;
    ## data normalization
    def Normalization_extraction(self,data,how_days):

        backtime = how_days-1;
        n = data.shape[0]
        row_number = n-backtime;#int(np.round(n / how_days));
        fea = np.zeros([row_number,how_days])
        add = 0;
        count_ = 0;
        i = 0;
        while i<n:
            if (count_<how_days):
                fea[add,count_] = data[i,1];
                count_ = count_ + 1;
                i+=1;
            else:
                add = add + 1;
                count_ = 0;
                i-=backtime;
        ind = np.argmax(data[:,1]);
        max_ = data[ind,1];
    #    fea = fea/max_;
        return fea,max_;