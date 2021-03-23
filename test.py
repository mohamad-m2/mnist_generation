import csv 
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import math
import functools as f
a= csv.reader(open('mnist.csv','r'),delimiter=',')
next(a)

dic={}
for b in a:
    
    c=np.array([float(x) for x in b[:-1]]).reshape((28,28))
    if(dic.get(float(b[-1]),0)==0):
        dic[float(b[-1])]=[]
    else:
        dic[float(b[-1])].append(c)
   

#plt.imshow(dic[2][0],cmap='gray')


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

#end
def show_pca_rank(n,dev,z1):
    k=np.zeros((50));
    
    for i in dev:
        for j in dev[i]:
            k=k+abs(j.T);
    k=(k*100)/k.sum()
    print(k.shape)
    m=np.array([i.reshape((28,28)) for i in n.T])
    figure, axes = plt.subplots(nrows=5, ncols=10,sharex=True, sharey=True)
    j=0
    for i in m:
        axes[j//10,j%10].imshow(i,cmap='gray')
        axes[j//10,j%10].set_title(str(round(k[0,j],1))+'%')
        j=j+1
    figure.tight_layout(pad=z1)
        
def show_pca(n):
    m=np.array([i.reshape((28,28)) for i in n.T])
    figure, axes = plt.subplots(nrows=5, ncols=10)
    j=0
    for i in m:
        axes[j//10,j%10].imshow(i,cmap='gray')
       
        j=j+1

def trans_dic(d,n):
    dv={i:[np.dot(n.T,k.reshape((784,1))) for k in dic[i]] for i in dic}
    return dv
    
def trans(n,f):
    figure, axes = plt.subplots(nrows=1, ncols=2)
    axes[0].imshow(f,cmap='gray')
    k=np.dot(n.T,f.reshape((784,1)))
    
    
    d=np.dot(n,k)

    d=d.reshape((28,28))    
    axes[1].imshow(d,cmap='gray')
    
  
def highst_m(co,x):
        m,n=np.linalg.eig(co)
        for i in range(len(n)):
             swap = i + np.argmax(m[i:])
             (m[i], m[swap]) = (m[swap], m[i])
             p=np.array(n[:,swap])
             n[:,swap]=np.array(n[:,i])
             n[:,i]=p
        return m,n[:,:x]
             

    
    
def extract(co):
    meanc=sum([sum([abs(k) for k in f]) for f in co])/co.size
    x=[]
    y=[]
    val=[]
    for idx,i in enumerate(co):
        for idy,j in enumerate(i):
            if(abs(j)>10*meanc):
                x.append(idx)
                y.append(idy)
                val.append(math.log(abs(j)))
    plt.figure()
    plt.scatter(x,y)
    
    
def calc_z(m):
    z={}
    for i in m:
        z[i]= f.reduce(lambda x,y: x+y,m[i])
        z[i]=z[i]/len(m[i])
    return z
    
def give (k):

        e={i:np.sum((z[i]-k)**2) for i in z}
        return min(e,key=e.get)
 
def calc_acc(m):
    count =0
    sum2=0
    for i in m:
        sum2+=len(m[i])
        for j in m[i]:
            if(give(j)!=i):
                count+=1
    return (count*100)/sum2
        