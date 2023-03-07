import random
import math
import matplotlib.pyplot as plt
import BPNet

random.seed()

def rand(a,b):
    return (b-a)*random.random()+a 

def calcfit(net,pat):
    error=0.0
    for p in pat:
        inputs=p[0]
        targets=p[1]
        net.calc(inputs)
        error+=sum([0.5*(t-o)**2 for t,o in zip(targets,net.ao)])
        #print('数据'+str(label)+'-训练次数'+str(counter)+':error %-.10f' % error)

    return 1.0/error

class Person:
    def __init__(self,gene,pat):
        self.gene=gene
        self.net=BPNet.BP_Net(225,3,1)
        hweights=[gene[0:225],gene[225:450],gene[450:675]]
        self.net.hlayer.set_weights(hweights)
        self.net.hlayer.set_thresholds(gene[675:678])
        self.net.olayer.set_weights([gene[678:681]])
        self.net.olayer.set_thresholds(gene[681:682])
        self.fit=calcfit(self.net,pat)

class Group:
    def __init__(self,pat):
        self.GroupSize=40  #种群规模
        self.GeneSize=682  #基因数量
        self.pat=pat#训练数据
        self.initGroup()
        self.upDate()
    #初始化种群，随机生成若干个体
    def initGroup(self):
        self.group=[]
        i=0
        while(i<self.GroupSize):
            i+=1
            gene=[rand(-0.2,0.2) for j in range(683)]
            self.group.append(Person(gene,self.pat))

    #获取种群中适应度最高的个体
    def getBest(self):
        bestFit=self.group[0].fit
        best=self.group[0]
        for person in self.group:
            temp_bestfit=person.fit
            if(temp_bestfit>bestFit):
                bestFit=temp_bestfit
                best=person
        return best

    #计算种群中所有个体的平均误差
    def getAvg(self):
        sum=0
        for p in self.group:
            sum+=1/p.fit
        return sum/len(self.group)

    #根据适应度，使用轮盘赌返回一个个体，用于遗传交叉
    def getOne(self):
        #section的简称，区间
        sec=[0]
        sumsec=0
        for person in self.group:
            sumsec+=person.fit
            sec.append(sumsec)
        p=random.random()*sumsec
        for i in range(len(sec)):
            if(p>sec[i] and p<sec[i+1]):
                #这里注意区间是比个体多一个0的
                return self.group[i]

    #更新种群相关信息
    def upDate(self):
        self.best=self.getBest()

#遗传算法的类，定义了遗传、交叉、变异等操作
class GA:
    def __init__(self,pat):
        self.group=Group(pat)
        self.pCross=0.7    #交叉率
        self.pChange=0.01    #变异率
        self.Gen=1  #代数
        self.pat=pat

    #变异操作
    def change(self,gene):
        #把列表随机的一段取出然后再随机插入某个位置
        #length是取出基因的长度，postake是取出的位置，posins是插入的位置
        geneLenght=len(gene)
        index1 = random.randint(0, geneLenght - 1)
        index2 = random.randint(0, geneLenght - 1)
        newGene = gene[:]       # 产生一个新的基因序列，以免变异的时候影响父种群
        newGene[index1], newGene[index2] = newGene[index2], newGene[index1]
        return newGene

    #交叉操作
    def cross(self,p1,p2):
        geneLenght=len(p1.gene)
        index1 = random.randint(0, geneLenght - 1)
        index2 = random.randint(index1, geneLenght - 1)
        tempGene = p2.gene[index1:index2]   # 交叉的基因片段
        newGene = []
        p1len = 0
        for g in p1.gene:
              if p1len == index1:
                    newGene.extend(tempGene)     # 插入基因片段
                    p1len += 1
              if g not in tempGene:
                    newGene.append(g)
                    p1len += 1
        return newGene

    #获取下一代
    def nextGen(self):
        self.Gen+=1
        #nextGen代表下一代的所有基因
        nextGen=[]
        #将最优秀的基因直接传递给下一代
        nextGen.append(self.group.getBest().gene[:])
        while(len(nextGen)<self.group.GroupSize):
            pChange=random.random()
            pCross=random.random()
            p1=self.group.getOne()
            if(pCross<self.pCross):
                p2=self.group.getOne()
                newGene=self.cross(p1,p2)
            else:
                newGene=p1.gene[:]
            if(pChange<self.pChange):
                newGene=self.change(newGene)
            nextGen.append(newGene)
        self.group.group=[]
        for gene in nextGen:
            self.group.group.append(Person(gene,self.pat))
            self.group.upDate()

    #打印当前种群的最优个体信息
    def showBest(self):
        print("第{}代\t当前最优{}\t当前平均{}\t".format(self.Gen,1/self.group.getBest().fit,self.group.getAvg()))

    #n代表代数，遗传算法的入口
    def run(self,n):
        Gen=[]  #代数
        dist=[] #每一代的最优距离
        avgDist=[]  #每一代的平均距离
        
        i=1
        while(i<n):
            self.nextGen()
            self.showBest()
            i+=1
            Gen.append(i)
            dist.append(1/self.group.getBest().fit)
            avgDist.append(self.group.getAvg())
        #绘制进化曲线
        plt.plot(Gen,dist,'-r')
        plt.plot(Gen,avgDist,'-b')
        plt.show()