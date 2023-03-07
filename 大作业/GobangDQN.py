import numpy as np
import math
from matplotlib import pyplot as plt
import random
import BPNet
import DQN
import GobangGamesearch
try:
    import cPickle as pickle
except:
    import pickle

COLUMN=15
ROW=15
# Hyper Parameters for DQN
GAMMA = DQN.GAMMA
INITIAL_EPSILON = DQN.INITIAL_EPSILON # starting value of epsilon
FINAL_EPSILON = DQN.FINAL_EPSILON # final value of epsilon
REPLAY_SIZE = DQN.REPLAY_SIZE # experience replay buffer size
BATCH_SIZE = DQN.BATCH_SIZE # size of minibatch
EPISODE = 1000 # Episode limitation
STEP = 225 # Step limitation in an episode
TEST = 10 # The number of experiment test every 100 episode

list1=[]#AI
list2=[]#human|AI2
list3=[]#all

list_all=[]#整个棋盘的点
next_point=[1,1]#AI下一步最应该走的位置

DEPTH=1#搜索深度

def changepos(posx,posy):#将二维坐标转换为一维下标
    return (posx-1)*15+posy-1

def game_win(list):
    for m in range(COLUMN):
        for n in range(ROW):
            if n<ROW-3 and (m,n) in list and (m,n+1) in list and (m,n+2) in list and (m,n+3) in list and (m,n+4) in list:
                return True
            elif m<COLUMN-3 and (m,n) in list and (m+1,n) in list and (m+2,n) in list and (m+3,n) in list and (m+4,n) in list:
                return True
            elif m<COLUMN-3 and n<ROW-3 and (m,n) in list and (m+1,n+1) in list and (m+2,n+2) in list and (m+3,n+3) in list and (m+4,n+4) in list:
                return True
            elif m<COLUMN-3 and n>4 and (m,n) in list and (m+1,n-1) in list and (m+2,n-2) in list and (m+3,n-3) in list and (m+4,n-4) in list:
                return True
    return False

def main():
    gametree1=GobangGamesearch.Gametree(1)#搜索深度
    
    for i in range(COLUMN):
        for j in range(ROW):
            gametree1.list_all.append((i+1,j+1))
            list_all.append((i+1,j+1))

    tempstate=[0]*225
    tempnext_state=[0]*225
    agent=DQN.DQN(tempstate)
    for episode in range(EPISODE):
        print("第%d次训练"%episode)
        tempstate=[0]*225
        tempnext_state=[0]*225
        agent.inputs=tempstate
        gametree1.list1=[]
        gametree1.list2=[]
        gametree1.list3=[]
        for step in range(STEP):
            posx,posy,value=agent.egreedy_action()
            gametree1.list2.append((posx,posy))
            gametree1.list3.append((posx,posy))
            tempnext_state[changepos(posx,posy)]=1
            agent.inputs[changepos(posx,posy)]=1
            #得到下一状态
            if gametree1.game_win(gametree1.list2):
                reward=1.0
                done=True
                agent.perceive(tempstate,(posx,posy),reward,tempnext_state,done)
                break

            pos=gametree1.ai(True)
            gametree1.list1.append(pos)
            gametree1.list3.append(pos)
            tempnext_state[changepos(pos[0],pos[1])]=-1
            agent.inputs[changepos(pos[0],pos[1])]=-1
            if gametree1.game_win(gametree1.list1):
                reward=-1.0
                done=True
                agent.perceive(tempstate,pos,reward,tempnext_state,done)
                break

            reward=0.1
            done=False
            agent.perceive(tempstate,pos,reward,tempnext_state,done)
            tempstate=tempnext_state

        if episode%100==0:
            change=0
            g=0
            list1=[]
            list2=[]
            list3=[]
            agent.inputs=[0]*225
            tempinputs=[0]*225
            while g==0:
                if change%2==1:
                    bestvalue=-10000
                    for i in range(225):
                        if tempinputs[i]==0:
                            tempinputs[i]=1
                            qrewardlist=agent.net.calc(tempinputs)
                            if(qrewardlist[0]>bestvalue):
                                bestvalue=qrewardlist[0]
                                posx=(int)(i/15)+1
                                posy=i%15+1
                            tempinputs[i]=0
                    
                    pos=(posx,posy)
                    tempinputs[changepos(posx,posy)]=1
                    agent.inputs[changepos(posx,posy)]=1
                    if pos in list3:
                        print("无落子位置。")
                        g=1

                    print("AI落子位置为："+str(pos))

                    list1.append(pos)
                    list3.append(pos)
                    
                    for i in range(ROW+1):
                        for j in range(COLUMN+1):
                            if(i==0):
                                print("%-3d"%(j),end='')
                            else:
                                if(j==0):
                                    print("%-3d"%(i),end='')
                                else:
                                    if (i,j) in list1:
                                        print("%-3s"%("-"),end='')
                                    elif (i,j) in list2:
                                        print("%-3s"%("+"),end='')
                                    else:
                                        print("%-3s"%("0"),end='')
                        print("")

                    if game_win(list1):
                        print("AI win!")
                        g=1
                    change=change+1
                else:
                    print("请输入你要走的位置坐标。格式为：'x,y'，范围为[1,15]")
                    a2,b2=eval(input())

                    if((a2,b2) in list3):
                        print("该位置已有棋子！")
                        continue
                    if(a2>15 or a2<1 or b2>15 or b2<1):
                        print("落子位置不合法！")
                        continue
                    agent.inputs[changepos(a2,b2)]=-1
                    tempinputs[changepos(a2,b2)]=-1
                    list2.append((a2,b2))
                    list3.append((a2,b2))

                    if game_win(list2):
                        print("You win!")
                        g=1
                    change=change+1

main()