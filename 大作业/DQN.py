import numpy as np
import random
import BPNet
from collections import deque

GAMMA = 0.9 # discount factor for target Q
INITIAL_EPSILON = 0.5 # starting value of epsilon
FINAL_EPSILON = 0.01 # final value of epsilon
REPLAY_SIZE = 10000 # experience replay buffer size
BATCH_SIZE = 50 # size of minibatch

random.seed()

class DQN:
    def __init__(self,inputs):
        self.inputs=inputs
        self.replay_buffer=deque()
        self.epsilon=INITIAL_EPSILON
        self.create_Q_network()
    
    def create_Q_network(self):
        self.net=BPNet.BP_Net(225,3,1)
        self.Q_value=self.net.calc(self.inputs)

    def perceive(self,state,action,reward,next_state,done):
        self.replay_buffer.append([state,action,reward,next_state,done])
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()

        if len(self.replay_buffer) > BATCH_SIZE:
            self.train_Q_network()

    def train_Q_network(self):
        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replay_buffer,BATCH_SIZE)
        state_batch=[]
        action_batch=[]
        reward_batch=[]
        next_state_batch=[]

        for i in range(BATCH_SIZE):
            state_batch.append(minibatch[i][0])
            action_batch.append(minibatch[i][1])
            reward_batch.append(minibatch[i][2])
            next_state_batch.append(minibatch[i][3])
        
        # Step 2: calculate y
        y_batch=[0]*BATCH_SIZE
        Q_value_batch=[0]*BATCH_SIZE
        for i in range(BATCH_SIZE):
            value0=-10000.0
            done = minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else:
                for j in range(225):
                    if(next_state_batch[i][j]!=0):
                        continue
                    else:
                        tempinputs=next_state_batch[i]#做中间变量存遍历时的状态
                        tempinputs[j]=1
                        qrewardlist=self.net.calc(tempinputs)
                        if qrewardlist[0]>value0:
                            value0=qrewardlist[0]
                Q_value_batch[i]=value0
                y_batch.append(reward_batch[i] + GAMMA * Q_value_batch[i])

        pat=[]
        for i in range(BATCH_SIZE):
            pat=[[state,[y]] for state,y in zip(state_batch,y_batch)]

        self.net.train(pat,iterations=5,N=0.001,M=0.1)

    def egreedy_action(self):
        #随机选择一步
        if random.random() <= self.epsilon:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10
            posx=1
            posy=1
            value0=0.0
            for i in range(225):
                posx=random.randint(1,15)
                posy=random.randint(1,15)
                if(self.inputs[(posx-1)*15+posy-1]!=0):
                    continue
                else:
                    tempinputs=self.inputs
                    tempinputs[(posx-1)*15+posy-1]=1
                    qrewardlist=self.net.calc(tempinputs)
                    value0=qrewardlist[0]
                    break

            return posx,posy,value0
    
        #选择价值最高的一步
        else:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10
            value0=-1e9
            posx=1
            posy=1
            for i in range(225):
                if(self.inputs[i]!=0):
                    continue
                else:
                    tempinputs=self.inputs#做中间变量存遍历时的状态
                    tempinputs[i]=1
                    qrewardlist=self.net.calc(tempinputs)
                    if(qrewardlist[0]>value0):
                        value=qrewardlist[0]
                        posx=(int)(i/15)+1
                        posy=i%15+1

            return posx,posy,value0
            