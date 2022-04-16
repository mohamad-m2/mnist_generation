# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 10:16:08 2021

@author: mohamadm.2
"""

import csv 
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

    
def filter_f(filt,img):
    lc=[]
    n=filt.shape[0]
    for i in range(img.shape[0]-n+1):
        for j in range(img.shape[0]-n+1):
            x=img[i:i+n,j:j+n]*filt
            z=x.sum()
            lc.append(z)
    return np.array(lc).reshape((img.shape[0]-n+1,img.shape[0]-n+1))
      
    

a= csv.reader(open('../mnist.csv','r'),delimiter=',')
next(a)

ls=list(a)
l=[[float(i) for i in j] for j in ls]
l=np.array(l)
l=np.array([i[:-1].reshape((28,28)) for i in l])




flt=np.array([[1,0,-1]]*3)
flt3=np.array([[1,0,0,-1]]*4)
flt2=np.array([[1,-1]]*2)


flt4=np.array([[-1,0,1,0,-1]]*5)
flt5=np.array([[-1]]*1)

lets_See1=filter_f(flt,l[0])
lets_See2=filter_f(flt2,l[0])
lets_See3=filter_f(flt3,l[0])

lets_See4=filter_f(flt3,l[0])
lets_See5=filter_f(flt5,l[0])

plt.imshow(l[0],cmap='gray')
plt.figure()
plt.imshow(lets_See1,cmap='gray')
plt.figure()
plt.imshow(lets_See2,cmap='gray')
plt.figure()
plt.imshow(lets_See3,cmap='gray')


plt.figure()
plt.imshow(lets_See4,cmap='gray')

plt.figure()
plt.imshow(lets_See5,cmap='gray')
