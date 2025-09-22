import random
import numpy as np
from typing import List
import torch

class DataLoader:
    def __init__(self, data:List[int], batch_size:int, context_length:int, shuffle=True):
        self.data = data
        self.data_len = len(data)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.context_length = context_length

    def get_train_batch_data(self):
        idxs = np.random.randint(0,self.data_len-self.context_length-1,size=(self.batch_size,)) 
        x = np.stack([self.data[i:i+self.context_length] for i in idxs])
        y = np.stack([self.data[i+1:i+self.context_length+1] for i in idxs])
        return torch.tensor(x),torch.tensor(y)
    
    def get_valid_batch_data_iter(self):
        start_num = (self.data_len - self.context_length - 1) // self.batch_size # 表示有多少个batch
        for i in range(start_num):
            bias = i * self.batch_size     # 表示每一个batch开始的位置
            x = np.stack([self.data[bias:bias+self.context_length] for i in range(self.batch_size)])
            y = np.stack([self.data[bias+1:bias+self.context_length+1] for i in range(self.batch_size)])
            yield torch.tensor(x),torch.tensor(y)

    def __len__(self):

        return self.data_len // self.batch_size