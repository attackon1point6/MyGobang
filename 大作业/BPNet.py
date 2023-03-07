import numpy as np
import math
from matplotlib import pyplot as plt
import random
try:
    import cPickle as pickle
except:
    import pickle

random.seed()

#产生(a,b)区间内的一个随机数
def rand(a,b):
    return (b-a)*random.random()+a 

def sigmoid(x):
    return math.tanh(x)

def dsigmoid(x):
    return 1.0-x**2

class Unit:
    def __init__(self,length):#length决定这一层有几个输入
        self.weight=[rand(-0.2,0.2) for i in range(length)]#初始化权值
        self.change=[0.0]*length#各权值更改量
        self.threshold=rand(-0.2,0.2)#阈值

    def calc(self,sample):
        self.sample=sample
        partsum=sum([i*j for i,j in zip(self.sample,self.weight)])#加权和
        #print(partsum)
        
        self.output=sigmoid(partsum)
        return self.output
    
    def update(self,diff,rate=0.1,factor=0.1):
        change=[rate*x*diff+factor*c for x,c in zip(self.sample,self.change)]
        self.weight=[w+c for w,c in zip(self.weight,change)]
        self.change=[x*diff for x in self.sample]

    def get_weight(self):
        return self.weight

    def set_weight(self,weight):
        self.weight=weight
    def set_threshold(self,threshold):
        self.threshold=threshold
    
class Layer:#out_length表示这一层的神经元数，input_length为每个神经元接收的数据项数
    def __init__(self,input_length,output_length):
        self.units=[Unit(input_length) for i in range(output_length)]
        self.output=[0.0]*output_length
        self.ilen=input_length

    def calc(self,sample):
        #self.output=[unit.calc(sample) for unit in self.units]
        self.output=[]
        for unit in self.units:
            self.output.append(unit.calc(sample))
        
        return self.output
    
    def update(self,diffs,rate=0.1,factor=0.1):
        for diff,unit in zip(diffs,self.units):
            unit.update(diff,rate,factor)
        
    def get_error(self,deltas):
        def error0(deltas,j):
            #return sum([delta*unit.weight[j] for delta,unit in zip(deltas,self.units)])
            neru0=self.units[j]
            sum=0
            for delta,neru0_weight in zip(deltas,neru0.weight):
                sum+=delta*neru0_weight
            return sum
        return [error0(deltas,j) for j in range(len(self.output))]

    def get_weights(self):
        weights={}
        for key,unit in enumerate(self.units):
            weights[key]=unit.get_weight()
        return weights
    
    def set_weights(self,weights):
        for key,unit in enumerate(self.units):
            unit.set_weight(weights[key])
    def set_thresholds(self,thresholds):
        for key,unit in enumerate(self.units):
            unit.set_threshold(thresholds[key])
    
class BP_Net:
    def __init__(self,ni,nh,no):
        self.ni=ni+1
        self.nh=nh
        self.no=no
        self.hlayer=Layer(self.ni,self.nh)
        self.olayer=Layer(self.nh,self.no)
    
    def calc(self,inputs):
        if len(inputs)!=self.ni-1:
            print("Wrong number of inputs!")
            exit(0)
        
        self.ai=inputs+[1.0]

        self.ah=self.hlayer.calc(self.ai)
        
        self.ao=self.olayer.calc(self.ah)
        
        return self.ao

    def update(self,targets,rate,factor):
        if len(targets)!=self.no:
            print("Wrong number of inputs!")
            exit(0)
        
        #计算误差
        output_deltas=[dsigmoid(ao)*(target-ao) for target,ao in zip(targets,self.ao)]
        
        hidden_deltas=[dsigmoid(ah)*error for ah,error in zip(self.ah,self.olayer.get_error(output_deltas))]

        #更新权重
        self.olayer.update(output_deltas,rate,factor)

        self.hlayer.update(hidden_deltas,rate,factor)

        return sum([0.5*(t-o)**2 for t,o in zip(targets,self.ao)])#最小平方误差准则公式

    def test(self,patterns):
        for p in patterns:
            print(p[0], '->')
            count=0
            for q in self.calc(p[0]): 
                if(q>0.4):
                    print("1",end=',')
                elif(q<-0.4):
                    print("-1",end=',')
                else:
                    print("0",end=',')
                #print(q,end=',')
                count+=1
                if(count==len(p[1])**0.5):
                    print('')
                    count=0
    
    def train(self,patterns,iterations=30,N=0.0001,M=0.1):
        counter=0
        label=1
        for i in range(iterations):
            error=0.0
            for p in patterns:
                inputs=p[0]
                targets=p[1]
                self.calc(inputs)
                error+=self.update(targets,N,M)
                print('数据'+str(label)+'-训练次数'+str(counter)+':error %-.10f' % error)
                label+=1
            counter+=1
            label=1

    def save_weights(self,fn):
        weights={
            "olayer":self.olayer.get_weights(),
            "hlayer":self.hlayer.get_weights()
        }
        with open(fn,"wb+") as f:
            pickle.dump(weights,f)
    
    def load_weights(self,fn):
        with open(fn,"rb") as f:
            weights=pickle.load(f)
            self.olayer.set_weights(weights["olayer"])
            self.hlayer.set_weights(weights["hlayer"])