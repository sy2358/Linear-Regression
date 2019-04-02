
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd

data = pd.read_csv("./data_for_assignment_1.csv",encoding="CP949",header=0)

x = np.array(data["A"])
y = np.array(data["B"])

def GDM(x,y):
    w = b = 0
    epoch = 500
    n = len(x) 
    step = 0.00001
    
    for i in range(epoch):
        y_hat = w * x + b
        cost = (1/n) * sum([val**2 for val in (y-y_hat)])
        wd = -(2/n) * sum(x*(y - y_hat))
        bd = -(2/n) * sum(y - y_hat)
        w = w - step * wd
        b = b - step * bd
        print("w: {}    b: {},    cost:{}   epoch:{}".format(w,b,cost,i))

GDM(x,y)


# In[7]:


def Moment_opt(x,y):
    w = b = wm = bm = 0
    epoch = 500
    n = len(x)
    step = 0.00001
    
    for i in range(epoch):
        y_hat = w * x + b
        cost = (1/n) * sum([val**2 for val in (y-y_hat)])
        
        md = -(2/n) * sum(x*(y - y_hat)) 
        bd = -(2/n) * sum(y - y_hat)
        
        wm = 0.9 * wm + step * md
        bm = 0.9 * bm + step * bd
        w = w - wm
        b = b - bm
        print("w: {}    b: {},    cost:{}   epoch:{}".format(w,b,cost,i))

Moment_opt(x,y)

