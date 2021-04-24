import csv 
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import math
import functools as f
import scipy.linalg as sc 



def trans_dic(dic,n):
    dv={i:[np.dot(n.T,k.reshape((784,1))) for k in dic[i]] for i in dic}
    return dv

def reconstruct(z,n):
    x=[]
    for i in z:
        x.append(np.dot(n,i).reshape((28,28)))
    return x

def show(dic,m):
    x=round(np.sqrt(9*m/2))
    
    h=x//3
    w=2*x//3 +1

    figure,axes=plt.subplots(nrows=h, ncols=w,sharex=True, sharey=True)

    k,j=0,0
    for i in dic:
        axes[k,j].imshow(i,cmap='gray')
        j+=1
        if(j==w):
            j=0
            k+=1

def trans(n1,n2):
    figure, axes = plt.subplots(nrows=1, ncols=2)
    
    
    axes[0].imshow(n1,cmap='gray')
    axes[1].imshow(n2,cmap='gray')

def trans_pca(n,x):
    k=np.dot(n.T,x.reshape((784,1)))
    d=np.dot(n,k)
    d=d.reshape((28,28))  
    return d


def highst_m(co,x):
        m,n=np.linalg.eigh(co)

        return (n.T[-1:-(x+1):-1]).T
    
    
def comp(ls,x):
    min1=0
    for i in ls:
        z=x==i
        if(z.sum()==i.size):
            return False
        else:
            if(z.sum()>min1):
                min1=z.sum()
    return True,min1


a= csv.reader(open('../mnist.csv','r'),delimiter=',')
next(a)

dic={}
for b in a:
    
    c=np.array([float(x) for x in b[:-1]]).reshape((28,28))
    if(dic.get(float(b[-1]),0)==0):
        dic[float(b[-1])]=[]
    else:
        dic[float(b[-1])].append(c)
   
mean=f.reduce(lambda x,y: x+y,f.reduce(lambda x,y:x+y,dic.values())) 
mean=mean/sum([len(i) for i in dic.values()])

# centeralize data
dic_center={i:[k-mean for k in dic[i]] for i in dic}
#end

#calc cov

temp=f.reduce(lambda x,y:x+y,dic_center.values())
cov=0
for i in temp:
    cov+=np.dot(i.reshape((784,1)),i.reshape((1,784)))
    
cov=cov/(sum([len(dic_center[i]) for i in dic_center]))

n=highst_m(cov,100)

dic_pca=trans_dic(dic,n)



#trying to generate


used=np.array(dic_pca[0][:33])[:,:,0]

set_s=np.array(dic_pca[0])[:,:,0]
mean=set_s.mean(axis=0)    
set_c=set_s-mean.reshape((1,100))
cov_tied=np.dot(set_c.T,set_c)/set_c.shape[0]


mean20=used.mean(axis=0)



gen1=np.random.multivariate_normal(mean20,cov_tied)

gen11=(reconstruct([gen1],n)[0]).reshape((784))

for idx in range(gen11.size):
   if(gen11[idx]<100):
       gen11[idx]=0
   else:
       gen11[idx]=255

gen12=gen11.reshape((28,28))
       
plt.imshow(gen12,cmap='gray')


new=[]
for i in dic[0]:
    new.append(i.reshape(784,1))


arr=np.array(new)[:,:,0]
#
#arr_mean=arr.mean(axis=0)
#
#arr_c=arr-arr_mean.reshape((1,784))
#cov_arr=np.dot(arr_c.T,arr_c)/arr_c.shape[0]
#
#
#gen2=np.random.multivariate_normal(arr_mean,cov_arr)
#
#for idx in range(gen2.size):
#   if(gen2[idx]<100):
#       gen2[idx]=0
#   else:
#       gen2[idx]=255
#
#gen3=gen2.reshape(28,28)
#
#
#
#
##plt.imshow(arr[0].reshape(28,28),cmap='gray')
#plt.figure()
#plt.imshow(gen3,cmap='gray')