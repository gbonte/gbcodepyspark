#from joblib import Parallel, delayed  
#from multiprocessing import Pool
import ast

from numpy import linalg as LA
import numpy as np
import gc
import pickle
from sklearn import gaussian_process
from sklearn.kernel_ridge import KernelRidge
from sklearn import preprocessing
#from sklearn.preprocessing import scale 
from scipy.stats.stats import pearsonr 
import sys
import csv
from pyspark.mllib.linalg import DenseVector
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.stat import Statistics
from sklearn.ensemble import forest

from pylab import *
from pyspark.mllib.regression import LinearRegressionWithSGD
from pyspark.mllib.tree import DecisionTree

from pyspark.mllib.tree import RandomForest, RandomForestModel
from sklearn import linear_model
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

import scipy as sp
import scipy.stats
def compcorr(x):
    x.shape=(len(x),1)
    return(abs(np.corrcoef(transpose(np.hstack((x,Xs.value))))[0,1:]))

def compmrmr(x):
    if (len(x)<=1):
        return(x[0])
    else:
       return(x[0]-np.mean(x[1:]))


def uniloo(x,y,B=10):
    s=np.argsort(x)
    y=y[s]
    y=(y-np.mean(y))/np.std(y)
    N=len(y)
    B=min(B,N/2)
    e=zeros(N-B)
    for i in arange(N-B,step=2):
        e[i]=np.std(y[i:(i+B)])
    return(mean(e))

def vcorrcoef(X,y):
    Xm = np.reshape(np.mean(X,axis=1),(X.shape[0],1))
    ym = np.mean(y)
    r_num = np.sum((X-Xm)*(y-ym),axis=1)
    r_den = np.sqrt(np.sum((X-Xm)**2,axis=1)*np.sum((y-ym)**2))
    r = r_num/r_den
    return r

def unique(a):
    """ return the list with duplicate elements removed """
    return list(set(a))

def intersect(a, b):
    """ return the intersection of two lists """
    return list(set(a) & set(b))

def union(a, b):
    """ return the union of two lists """
    return list(set(a) | set(b))

def _getAplus(A):
    eigval, eigvec = np.linalg.eig(A)
    Q = np.matrix(eigvec)
    xdiag = np.matrix(np.diag(np.maximum(eigval, 0)))
    return Q*xdiag*Q.T

def _getPs(A, W=None):
    W05 = np.matrix(W**.5)
    return  W05.I * _getAplus(W05 * A * W05) * W05.I

def _getPu(A, W=None):
    Aret = np.array(A.copy())
    Aret[W > 0] = np.array(W)[W > 0]
    return np.matrix(Aret)

def nearPD(A, nit=10):
    n = A.shape[0]
    W = np.identity(n) 
# W is the matrix used for the norm (assumed to be Identity matrix here)
# the algorithm should work for any diagonal W
    deltaS = 0
    Yk = A.copy()
    for k in range(nit):
        Rk = Yk - deltaS
        Xk = _getPs(Rk, W=W)
        deltaS = Xk - Rk
        Yk = _getPu(Xk, W=W)
    return Yk


def generateSPDmatrix(n):
    A = 4*np.random.rand(n, n) # % generate a random n x n matrix
    #% construct a symmetric matrix using either
    A = A+A.T 
    A = A + n*np.identity(n)
    return(A)

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h


def upper_confidence_interval(data, confidence=0.95):
    return(mean_confidence_interval(data,confidence)[2])

def which(x):
  return(x.nonzero()[0])


def show(p):
  img = StringIO.StringIO()
  p.savefig(img, format='svg')
  img.seek(0)
  #print "%html <div style='width:600px'>" + img.buf + "</div>"



def dump(dumpfile,v):
  file = open(dumpfile, 'w')
  dict={'res':v}
  pickle.dump(dict, file)
  file.close()




def rddSum(D1,D2):
    nump=D1.getNumPartitions()    
    D1=D1.coalesce(1)
    D2=D2.coalesce(1)
    Ddiff=D1.zip(D2).map(lambda x:x[0]+x[1])
    return(Ddiff)


def rddDiff(D1,D2):
    nump=D1.getNumPartitions()    
    D1=D1.coalesce(1)
    D2=D2.coalesce(1)
    Ddiff=D1.zip(D2).map(lambda x:x[0]-x[1])
    return(Ddiff)

def mserdd(D1,D2):
    D1.coalesce(1)
    D2.coalesce(1)
    return(np.mean(D1.zip(D2).map(lambda x:pow(x[0]-x[1],2)).collect()))

def rddArr(D):
  D=D.map(lambda a,b: (b,a)).sortByKey()
  return (np.array(D.map(lambda a,b: b).collect()))


def rddApplyMean(D,axis=0):
    if (axis==0): # column
        N=D.count()
        return(D.reduce(lambda x,y:x+y)/N)

    if (axis==1): #row
        return(rddArr(D.map(lambda x:mean(x))))

def match(a,b):
    r=list()
    for x in a:
        if x in b:
            r=r+[b.index(x)]
    
    return np.mean(np.array(r))

def match2(a,b):
    r=list()
    for x in a:
        if x in b:
            r=r+[b.index(x)]
    return [min(np.array(r)),median(np.array(r)),max(np.array(r))]






def rddApplyVar(D,axis):
    if (axis==0): # column
         return(Statistics.colStats(D).variance())
      

    if (axis==1): #row
        return(rddArr(D.map(lambda x:var(x))))

def rddApplySum(D,axis=0):
    if (axis==0): # column
        return(D.sum())

    if (axis==1): #row
        return(rddArr(D.map(lambda x:np.sum(x))))

def rddNRows(rdd):
    return(rdd.count())

def rddSize(rdd):
    N=rdd.count()
    f=rdd.first()
    if (type(f).__name__=='float64'):
        n=1
    else:
        n=len(f[0])
    nump=rdd.getNumPartitions()
    return([N,n,nump])




def rddCreate2(N,n,npart=10,sd=0.1,lin=False):
## dataset generation
    yx=np.random.randn(N, n)
    
    Y=np.zeros((N,1))
    Y.shape=(N,1)
    fsall=np.random.choice(n,n/2,replace=False)
    for it in arange (0,n/2):
        fs=fsall[it]
        ya=yx[:,fs]
        ya.shape=(N,1)
        if lin:
          Y=Y+2*np.random.rand()*ya
        else:
          if (np.random.rand()<0.3):
            if (np.random.rand()<0.3):
              Y=Y+sin(ya)
            else :
              Y=Y+np.random.rand()*(ya)+np.random.rand()*pow(ya,2)
          else:
            if (np.random.rand()<0.3):
              Y=Y*log(abs(sin(ya)))
            else:
              Y=Y+abs(ya)
  
    Y.shape=(N,1)
    Y=(Y-mean(Y))/np.std(Y)
    Y=Y+sd*np.random.randn(N, 1)
    YXall=np.hstack((Y,yx))
    return (sc.parallelize(YXall,npart))

def arrCreate2(N,n,sd=0.1,seed=0,nofeat=5):
    mean=np.zeros(n)
    cov=generateSPDmatrix(n) #nearPD(np.identity(n)+np.random.rand(n,n))
    yx=np.random.multivariate_normal(mean, cov, N)
    if (seed==0):
        
        fsall=np.random.choice(n,nofeat,replace=False)
        F=datasets.make_friedman1(n_samples=N, n_features=len(fsall))
        yx[:,fsall]=F[0]
            
    if (seed==1):
       
        fsall=np.random.choice(n,nofeat,replace=False)
        F=datasets.make_regression(n_samples=N, n_features=len(fsall), n_informative=n/2,bias=0.0, effective_rank=None, tail_strength=0.5)
        yx[:,fsall]=F[0]

    if (seed==2):
       
        fsall=np.random.choice(n,min(nofeat,4),replace=False)
        F=datasets.make_friedman2(n_samples=N)
        yx[:,fsall]=F[0]

    if (seed==3):
        yx=np.random.randn(N, n)
        fsall=np.random.choice(n,min(nofeat,4),replace=False)
        F=datasets.make_friedman3(n_samples=N)
        yx[:,fsall]=F[0]
    
      
    yx=preprocessing.scale(yx)+5*np.random.randn(N,n)
    Y=F[1]
    Y.shape=(N,1)
    Y=(Y-np.mean(Y))/np.std(Y)+sd*np.random.randn(N,1)
    YXall=np.hstack((Y,yx))
    return (YXall,fsall)


def nonlinf(X,f):
    n=X.shape[1]
    fx=np.random.choice(n,np.random.choice(np.arange(2,n-2),1),replace=False)
    fy=np.setdiff1d(np.arange(n),fx)
    x=np.apply_along_axis(np.mean,1,abs(X[:,fx]))
    y=np.apply_along_axis(np.mean,1,abs(X[:,fy]))
  
    if f==0:
        Yhat=np.log(pow(x,2)+pow(y,2))
    if (f==1):
        Yhat=np.sqrt(abs(np.sin(pow(x,2)+pow(y,2))))
    if (f==2):
        Yhat=np.log(x*pow(y,2)+pow(x,2)*y)
    if (f==3):
        Yhat=np.sqrt(abs(pow(x,2)/(y+1)))
    if (f==4):
        Yhat=1/(pow(x,2)+pow(y,2)+1)
    if (f==5):
        Yhat=(x*np.sin(x*y))/(pow(x,2)+pow(y,2)+1)
    if (f==6):
        Yhat=y+np.exp(-abs(x)/2)
    if (f==7):
        Yhat=y*np.sin(x)+x*np.sin(y)
    if (f==8):
        Yhat=(pow(x,3)-2*x*y+pow(y,2))/(pow(x,2)+pow(y,2)+1)
    if (f>8):
        Yhat=np.sin(x)+np.log(y)
       
    return(Yhat)


def arrCreate(N,n,sd=0.1,seed=0,nofeat=5):
    mean=np.zeros(n)
    cov=generateSPDmatrix(n) #nearPD(np.identity(n)+np.random.rand(n,n))
    yx=np.random.multivariate_normal(mean, cov, N)
    fsall=np.random.choice(n,nofeat,replace=False)
    if False:
        regr = RandomForestRegressor(n_estimators=2,max_depth=5,n_jobs=-1)
        regr.fit(yx[:100,fsall],2*np.random.randn(100,1))
        Y=regr.predict(yx[:,fsall])
    else:
        Y=nonlinf(yx[:,fsall],int(np.random.choice(9,1)))

    Y.shape=(N,1)
    Y=(Y-np.mean(Y))/np.std(Y)+sd*np.random.randn(N,1)
    YXall=np.hstack((Y,yx))
    #fsall=np.setdiff1d(fsall,removed)
    return (YXall,fsall)

def rddCreatefromArray(A,np=10):
  N=A.shape[0]
  I=arange(N)
  I.shape=(N,1)
  B=hstack((A,I))
  return(sc.parallelize(B,np).map(lambda x: (x[:-1],x[-1])))

def isfloat(value):
  try:
    float(value)
    return True
  except:
    return False

def ffloat(x):
 if isfloat(x):
     return(float(x))
 else:
     return(float('nan'))

def parsePoint(line):
   
    values=[ffloat(x) for x in line.split(',')]
    return np.array(values)

def rddRead(f,np=100):  ## text file with ',' separator
    textFile = sc.textFile(f,np)
    D=textFile.map(parsePoint)
    D=D.repartition(np)
    return(D)


def outString(namelog,strwrite):
    f = open(namelog, "a")
    f.write(strwrite)
    f.close()




#def fit(iterator):
#    data=np.array(list(iterator))
#    X = data[:, 1:]
#    Y=data[:, 0]
#    localregr = regr.value
# Train the model using the training sets
#    localregr.fit(X, Y)
#    return([localregr.predict(X3)])


    
def rddBiasTs(iterator):
    ## Bias estimator with bag of little bootstraps
    
    data=np.array(list(iterator))
    if (data.size==0):
        return([])

    X = data[:, 1:]
    Y=data[:, 0]
    NY=len(Y)
    NYts=len(Yts.value)
    localregr = regr.value
    # Train the model using the training sets
    for it in arange (0,10):
        w=np.random.multinomial(B.value, [1/(NY+0.0)]*NY, size=1)/(B.value+0.0)
        w.shape=(NY)
        localregr.fit(X, Y,sample_weight=w)
        if (it==0):
            pr=localregr.predict(Xts.value)
            pr.shape=(NYts,1)
            P=pr
        else: 
            pr=localregr.predict(Xts.value)
            pr.shape=(NYts,1)
            P=hstack((P,pr))

    b=array(Yts.value-np.mean(P,axis=1))
    b.shape=(1,NYts)
    return(b)
  

def localPart(iterator):
    data=np.array(list(iterator))
    return([data.shape])

def rddDistrPart(D):
    return D.mapPartitions(localPart).collect()

def localPart2(iterator):
    data=np.array(list(iterator))
    return([data])

def rddGetPart(D,np):
    return D.mapPartitions(localPart2).collect()[np]

def conc(it):
  b=list(it)
  return(concatenate((b[0],b[1])))

def conclist(it):
  b=list(it)
  return([b[1]]+b[0])





def rddScale(D):
    summary = Statistics.colStats(D)
    sd=np.sqrt(summary.variance())
    mn=summary.mean()
    return(D.map(lambda x: (x-mn)/sd))



def f(splitIndex ,v): 
    return [(splitIndex,list(v))]



def corr2_coeff(A,B):
    A_mA = A - A.mean(1)[:,None]
    B_mB = B - B.mean(1)[:,None]
    ssA = (A_mA**2).sum(1);
    ssB = (B_mB**2).sum(1);
    return np.dot(A_mA,B_mB.T)/np.sqrt(np.dot(ssA[:,None],ssB[None]))

def getCorrelation(k,v):
    pairBlock=list(v)
    pairBlock1=pairBlock[0][0]
    
    blockMatrix1=pairBlock[0][1]
    if (len(pairBlock)==1):
        blockMatrix2=pairBlock[0][1]
        k=(pairBlock[0][0],pairBlock[0][0])
    else:
        blockMatrix2=pairBlock[1][1]
        k=(pairBlock[0][0],pairBlock[1][0])
    
    corrB1B2=corr2_coeff(np.array(blockMatrix1),np.array(blockMatrix2))
    
    return (k,corrB1B2)




def makePairParts2(k,v,p,maxk=5):
    Rfit=[]
    for localregr in regr.value:
        if (k< maxk):
            X=np.array(v)
            Y=X[:,0]
            X=X[:,1:]
            rfit=localregr.fit(X, Y)
        else:
            rfit=[]
        if Rfit==[]:
            Rfit=[rfit]
        else:
            Rfit=Rfit+[rfit]
    if k<maxk:
        return [(str(sorted([k,l])),(k,v,Rfit)) for l in np.setdiff1d(range(0,p),k)]
    else:
        return [(str(sorted([k,l])),(k,v,Rfit)) for l in range(0,maxk)]
    ##  return [(str(sorted([k,l])),(k,v,Rfit)) for l in range(max(0,k-10),min(p,k+10))]
    ##return [(str(sorted([k,l])),(k,v,Rfit)) for l in range(0,3)]

def getPred(k,v):    
    thr=1
    pairBlock=list(v)
    pairBlock1=pairBlock[0][0]
    ##print pairBlock
    #print pairBlock[1][0]
    blockMatrix1=pairBlock[0][1]
    if (len(pairBlock)==1):
        blockMatrix2=pairBlock[0][1]
        k0=pairBlock[0][0]
        k1=k0  #,pairBlock[0][0])    
        rfit0=pairBlock[0][2]
        rfit1=rfit0
    else:
        blockMatrix2=pairBlock[1][1]
        k0=pairBlock[0][0]
        rfit0=pairBlock[0][2]
        k1=pairBlock[1][0]
        rfit1=pairBlock[1][2]
    
   
    if (k0==k1):
        NX=np.array(blockMatrix1).shape[0]
        for i in arange(len(rfit0)):
            if i==0:
                O=np.ones((1,NX))[0]*float('nan')
            else:
                O=vstack((O,np.ones((1,NX))[0]*float('nan')))
        return ([ ((k0,k0), O)] )
    else:
      #X=np.array(blockMatrix1)
      ##Y=X[:,0]
      #X=X[:,1:]
      #rfit0=regr.fit(X, Y)
        corr12=[]
        for rf0 in rfit0:
            Xts=np.array(blockMatrix2)
            Yts=Xts[:,0]
            Xts=Xts[:,1:]
            pp=np.ones((1,Xts.shape[0]))[0]*float('nan')
            if (rf0!=[]):
                pp=rf0.predict(Xts)
            pp.shape=(1,len(Yts))
            if (corr12==[]):
                corr12=pp
            else:
                corr12=vstack((corr12,pp))
      #X=np.array(blockMatrix2)
      ##regr = RandomForestRegressor() ##linear_model.LinearRegression()
      #Y=X[:,0]
      #X=X[:,1:]
      #rfit1=regr.fit(X, Y)
        corr21=[]
        for rf1 in rfit1:
            Xts=np.array(blockMatrix1)
            Yts=Xts[:,0]
            Xts=Xts[:,1:]
            pp=np.ones((1,Xts.shape[0]))[0]*float('nan')
            if (rf1!=[]): #np.random.uniform()<thr):
                pp=rf1.predict(Xts)
            pp.shape=(1,len(Yts))
            if (corr21==[]):
                corr21=pp
            else:
                corr21=vstack((corr21,pp))
      
        return (((k0,k1),corr12),((k1,k0),corr21))


def getPred2(v0,v1):    
    thr=1
   
    
    blockMatrix1=v0[1]
   
    blockMatrix2=v1[1]
    k0=v0[0]
    rfit0=v0[2]
    k1=v1[0]
    rfit1=v1[2]
    
   
    if (k0==k1):
        NX=np.array(blockMatrix1).shape[0]
        for i in arange(len(rfit0)):
            if i==0:
                O=np.ones((1,NX))[0]*float('nan')
            else:
                O=vstack((O,np.ones((1,NX))[0]*float('nan')))
        return ([ ((k0,k0), O)] )
    else:
      #X=np.array(blockMatrix1)
      ##Y=X[:,0]
      #X=X[:,1:]
      #rfit0=regr.fit(X, Y)
        corr12=[]
        for rf0 in rfit0:
            Xts=np.array(blockMatrix2)
            Yts=Xts[:,0]
            Xts=Xts[:,1:]
            pp=np.ones((1,Xts.shape[0]))[0]*float('nan')
            if (rf0!=[]):
                pp=rf0.predict(Xts)
            pp.shape=(1,len(Yts))
            if (corr12==[]):
                corr12=pp
            else:
                corr12=vstack((corr12,pp))
      #X=np.array(blockMatrix2)
      ##regr = RandomForestRegressor() ##linear_model.LinearRegression()
      #Y=X[:,0]
      #X=X[:,1:]
      #rfit1=regr.fit(X, Y)
        corr21=[]
        for rf1 in rfit1:
            Xts=np.array(blockMatrix1)
            Yts=Xts[:,0]
            Xts=Xts[:,1:]
            pp=np.ones((1,Xts.shape[0]))[0]*float('nan')
            if (rf1!=[]): #np.random.uniform()<thr):
                pp=rf1.predict(Xts)
            pp.shape=(1,len(Yts))
            if (corr21==[]):
                corr21=pp
            else:
                corr21=vstack((corr21,pp))
      
        return (((k0,k1),corr12),((k1,k0),corr21))



def rddTest(iterator,whichmod=0):
    ## train the same prediction model to each partition of a Rdd and test it on a broadcasted dataset Xts
    data=np.array(list(iterator))
    if (data.shape[0]<10):
        return([])
    X = data[:, 1:]
    Y=data[:, 0]
    if (X.shape[0]<20):
        return([])
    NYval=bXts.value.shape[0]
    Pr=[]
    for localregr in [regr.value[i] for i in whichmod]:
        localregr=regr.value[1]
        rfit=localregr.fit(X, Y)
        pr=rfit.predict(bXts.value)
        pr.shape=(1,NYval)
        if Pr==[]:
            Pr=pr
        else:
            Pr=vstack((Pr,pr))
    return(Pr)


def rddFitTs(iterator,whichmod):
    ## train the same prediction model to each partition of a Rdd and test it on a broadcasted dataset Xts
    data=np.array(list(iterator))
    if (data.shape[0]<10):
        return([])
    X = data[:, 1:]
    Y=data[:, 0]
    if (X.shape[0]<20):
        return([])
   

    NYval=bXts.value.shape[0]
    Pr=[]
    for localregr in [regr.value[i] for i in whichmod]:
        rfit=localregr.fit(X, Y)
        pr=rfit.predict(bXts.value)
        pr.shape=(1,NYval)
        if Pr==[]:
            Pr=pr
        else:
            Pr=vstack((Pr,pr))
  
    return(Pr)

def rddSelect(iterator,m):
  data=np.array(list(iterator))
  if (data.shape[0]==0):
        return([])
  return([data[m,:]])

def rddPred(Dtr,Xts,it=10,whichmod=0):
    global bXts
    bXts=sc.broadcast(Xts)
    p=Dtr.getNumPartitions()
    m=len(whichmod)
    mp=np.zeros((m,Xts.shape[0]))
    for iter in arange(it):
        if it>1:
            Dtr=Dtr.repartition(p)
        ## Dpred=Dtr.map(lambda (a,b):a).mapPartitions(lambda x: rddFitTs(x,whichmod))
        Dpred=Dtr.keys().mapPartitions(lambda x: rddFitTs(x,whichmod))
        for i in arange(m):
            Dm=Dpred.mapPartitions(lambda x: rddSelect(x,i)).reduce(lambda x,y:x+y)/p
            mp[i,:]=mp[i,:]+Dm
    return(mp/it)


def WeightedSum(X,W):
    N=X.shape[1]
    wa=np.zeros((1,N))
    for i in np.arange(N):
        w=np.maximum(0.01,W[:,i])
        w=(1/w)/np.sum(1/w)
        wa[0,i]=np.sum(w*X[:,i])
    return wa



def rddAddModel(iterator,whichmod):
    Rfit=[]
    data=np.array(list(iterator))
    if (data.shape[0]<10):
        return([])
    X = data[:, 1:]
    Y=data[:, 0]
    if (X.shape[0]<20):
        return([])   
    Pr=[]
    for localregr in [regr.value[i] for i in whichmod]:
        rfit=localregr.fit(X, Y)
        Rfit=Rfit+[rfit]  
    return(Rfit)

def rddUseModel(iterator,Xts):
    rfit=list(iterator)[0]
    #return [pow(Yts-rfit.predict(Xts),2)]
    return [rfit.predict(Xts)]
def rddCreateModels(iterator,mod):

    data=np.array(list(iterator))
    X = data[:, 1:]
    Y=data[:, 0]
    m=mod.fit(X,Y)  
    return [m]

def rddCreateModels2(iterator):
    data=np.array(list(iterator))
    X = data[:, 1:]
    Y=data[:, 0]
    return [mregr.value]



def rddComputeCorr(iterator):
    data=np.array(list(iterator))
    X = data[:, 1:]
    Y=data[:, 0]
    #vcorrcoef(arrYX0[:,1:].T,np.reshape(arrYX0[:,0],(1,arrYX0.shape[0])))
    cf=vcorrcoef(X[:,1:].T,np.reshape(Y,(1,X.shape[0])))
    return [cf]



def meanbut(x):
    return np.mean(x[np.argsort(-x)[:-1]])

def rddApplyModels(iterator):
    data=np.array(list(iterator))
    X = data[:, 1:]
    NN=X.shape[0]
    Y=data[:, 0]
    Y.shape=(1,NN)
    Yhat=[]
    Ehat=[]
    mE=np.zeros(len(listmod.value))
    cnt=0
    for mod in listmod.value:
        if (Yhat==[]):
            Yhat=mod.predict(X)
            Yhat.shape=(1,NN)
            Ehat=pow(Y-Yhat,2)
            Ehat.shape=(1,NN)
            Yhat.shape=(1,NN)
            mE[cnt]=mean(Ehat)
        else:
            yhat=mod.predict(X)
            yhat.shape=(1,NN)
            ehat=pow(Y-yhat,2)
            ehat.shape=(1,NN)
            yhat.shape=(1,NN)
            Yhat=vstack((Yhat,yhat))
            Ehat=vstack((Ehat,ehat))
            mE[cnt]=mean(ehat)
        cnt=cnt+1

    wm=np.argmin(mE)
    Ym=np.mean(Yhat,0)
    Em=np.mean(np.delete(Ehat,wm,0),0)
    #Ym=np.apply_along_axis(meanbut,0,Yhat)
    #Em=np.apply_along_axis(meanbut,0,Ehat)
    Ym.shape=(NN,1)
    Em.shape=(NN,1)
    Y.shape=(NN,1)
    return (hstack((Em,X)))
   
