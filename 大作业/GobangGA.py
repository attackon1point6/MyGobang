import numpy as np
import math
from matplotlib import pyplot as plt
import random
import BPNet
import GA
try:
    import cPickle as pickle
except:
    import pickle

COLUMN=15
ROW=15

list1=[]#AI
list2=[]#human
list3=[]#all

list_all=[]#整个棋盘的点
next_point=[0,0]#AI下一步最应该走的位置

DEPTH=3#搜索深度

def ai(n):
    global cut_count#剪枝次数
    cut_count=0
    global search_count#统计搜索次数
    search_count=0
    maxaiscore=negamax(True,DEPTH,-1e9,1e9,n)
    print("本次剪枝："+str(cut_count))
    print("本次搜索："+str(search_count))
    print('maxscore='+str(maxaiscore))
    return next_point[0],next_point[1]

#负值极大搜索 剪枝
def negamax(is_ai,depth,alpha,beta,n):
    if(game_win(list1) or game_win(list2) or depth==0):
        return evaluation(is_ai,n)
    
    blank_list=list(set(list_all).difference(set(list3)))
    #遍历每一个可能的位置
    for next_step in blank_list:
        global search_count
        search_count+=1
        
        if not has_neightnor(next_step):
            continue

        if is_ai:
            list1.append(next_step)
        else:
            list2.append(next_step)
        list3.append(next_step)

        value=-negamax(not is_ai,depth-1,-beta,-alpha,n)
        if is_ai:
            list1.remove(next_step)
        else:
            list2.remove(next_step)
        list3.remove(next_step)

        if value>alpha:
            if depth==DEPTH:
                next_point[0]=next_step[0]
                next_point[1]=next_step[1]
            if value>=beta:
                global cut_count
                cut_count+=1
                return beta
            alpha=value
            
    return alpha

def has_neightnor(pt):
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            if (pt[0] + i, pt[1]+j) in list3:
                return True
    return False

#评估函数
def evaluation(is_ai,n):
    total_score=0

    if is_ai:
        my_list=list1
        enemy_list=list2
    else:
        my_list=list2
        enemy_list=list1

    list3_to_board=[]

    for i in range(1,16):
        for j in range(1,16):
            if (i,j) in my_list:
                list3_to_board.append(1)
            elif (i,j) in enemy_list:
                list3_to_board.append(-1)
            else:
                list3_to_board.append(0)
    
    my_score=n.calc(list3_to_board)

    list3_to_board=[]

    for i in range(1,16):
        for j in range(1,16):
            if (i,j) in enemy_list:
                list3_to_board.append(1)
            elif (i,j) in my_list:
                list3_to_board.append(-1)
            else:
                list3_to_board.append(0)
    
    enemy_score=n.calc(list3_to_board)
    
    total_score=my_score[0]-enemy_score[0]*0.1

    return total_score    

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
    
    data01=[
        [0]*15,
        [0]*15,
        [0]*15,
        [0]*15,
        [0]*15,
        [0]*15,
        [0]*15,
        [0]*7+[-1]+[0]*7,
        [0]*15,
        [0]*15,
        [0]*15,
        [0]*15,
        [0]*15,
        [0]*15,
        [0]*15
    ]
    data02=[
        [0]*15,
        [0]*15,
        [0]*15,
        [0]*15,
        [0]*15,
        [0]*15,
        [0]*15,
        [0]*7+[-1,1]+[0]*6,
        [0]*8+[-1]+[0]*6,
        [0]*15,
        [0]*15,
        [0]*15,
        [0]*15,
        [0]*15,
        [0]*15
    ]
    data03=[
        [0]*15,
        [0]*15,
        [0]*15,
        [0]*15,
        [0]*15,
        [0]*15,
        [0]*6+[1]+[0]*8,
        [0]*7+[-1,1]+[0]*6,
        [0]*8+[-1]+[0]*6,
        [0]*9+[-1]+[0]*5,
        [0]*15,
        [0]*15,
        [0]*15,
        [0]*15,
        [0]*15
    ]
    data04=[
        [0]*15,
        [0]*15,
        [0]*15,
        [0]*15,
        [0]*15,
        [0]*15,
        [0]*6+[1]+[0]*8,
        [0]*7+[-1,1]+[0]*6,
        [0]*7+[1,-1]+[0]*6,
        [0]*9+[-1]+[0]*5,
        [0]*10+[-1]+[0]*4,
        [0]*15,
        [0]*15,
        [0]*15,
        [0]*15
    ]
    data05=[
        [0]*15,
        [0]*15,
        [0]*15,
        [0]*15,
        [0]*15,
        [0]*15,
        [0]*6+[1]+[0]*8,
        [0]*7+[-1,1]+[0]*6,
        [0]*7+[1,-1]+[0]*6,
        [0]*9+[-1]+[0]*5,
        [0]*8+[-1,0,-1]+[0]*4,
        [0]*11+[1,0,0,0],
        [0]*15,
        [0]*15,
        [0]*15
    ]
    data06=[
        [0]*15,
        [0]*15,
        [0]*15,
        [0]*15,
        [0]*15,
        [0]*15,
        [0]*6+[1,0,0,1]+[0]*5,
        [0]*7+[-1,1]+[0]*6,
        [0]*7+[1,-1]+[0]*6,
        [0]*6+[-1,0,0,-1]+[0]*5,
        [0]*8+[-1,0,-1]+[0]*4,
        [0]*11+[1,0,0,0],
        [0]*15,
        [0]*15,
        [0]*15
    ]
    data07=[
        [0]*15,
        [0]*15,
        [0]*15,
        [0]*15,
        [0]*11+[-1,0,0,0],
        [0]*10+[1,0,0,0,0],
        [0]*6+[1,0,0,1]+[0]*5,
        [0]*7+[-1,1]+[0]*6,
        [0]*7+[1,-1]+[0]*6,
        [0]*6+[-1,0,0,-1]+[0]*5,
        [0]*8+[-1,0,-1]+[0]*4,
        [0]*11+[1,0,0,0],
        [0]*15,
        [0]*15,
        [0]*15
    ]
    data08=[
        [0]*15,
        [0]*15,
        [0]*15,
        [0]*15,
        [0]*11+[-1,0,0,0],
        [0]*10+[1,0,0,0,0],
        [0]*6+[1,1,-1,1]+[0]*5,
        [0]*7+[-1,1]+[0]*6,
        [0]*7+[1,-1]+[0]*6,
        [0]*6+[-1,0,0,-1]+[0]*5,
        [0]*8+[-1,0,-1]+[0]*4,
        [0]*11+[1,0,0,0],
        [0]*15,
        [0]*15,
        [0]*15
    ]
    data09=[
        [0]*15,
        [0]*15,
        [0]*15,
        [0]*15,
        [0]*5+[-1]+[0]*5+[-1,0,0,0],
        [0]*6+[1,0,0,0,1,0,0,0,0],
        [0]*6+[1,1,-1,1]+[0]*5,
        [0]*7+[-1,1]+[0]*6,
        [0]*7+[1,-1]+[0]*6,
        [0]*6+[-1,0,0,-1]+[0]*5,
        [0]*8+[-1,0,-1]+[0]*4,
        [0]*11+[1,0,0,0],
        [0]*15,
        [0]*15,
        [0]*15
    ]
    data010=[
        [0]*15,
        [0]*15,
        [0]*15,
        [0]*15,
        [0]*5+[-1]+[0]*5+[-1,0,0,0],
        [0]*6+[1,0,0,0,1,0,0,0,0],
        [0]*6+[1,1,-1,1]+[0]*5,
        [0]*7+[-1,1]+[0]*6,
        [0]*7+[1,-1,-1]+[0]*5,
        [0]*6+[-1,0,0,-1,1]+[0]*4,
        [0]*8+[-1,0,-1]+[0]*4,
        [0]*11+[1,0,0,0],
        [0]*15,
        [0]*15,
        [0]*15
    ]
    data011=[
        [0]*15,
        [0]*15,
        [0]*15,
        [0]*15,
        [0]*5+[-1]+[0]*5+[-1,0,0,0],
        [0]*6+[1,0,0,0,1,0,0,0,0],
        [0]*6+[1,1,-1,1]+[0]*5,
        [0]*7+[-1,1]+[0]*6,
        [0]*7+[1,-1,-1]+[0]*5,
        [0]*6+[-1,0,0,-1,1]+[0]*4,
        [0]*6+[-1,0,-1,1,-1]+[0]*4,
        [0]*11+[1,0,0,0],
        [0]*15,
        [0]*15,
        [0]*15
    ]
    data012=[
        [0]*15,
        [0]*15,
        [0]*15,
        [0]*15,
        [0]*5+[-1]+[0]*5+[-1,0,0,0],
        [0]*6+[1,0,0,0,1,0,0,0,0],
        [0]*6+[1,1,-1,1]+[0]*5,
        [0]*6+[1,-1,1]+[0]*6,
        [0]*6+[-1,1,-1,-1]+[0]*5,
        [0]*6+[-1,0,0,-1,1]+[0]*4,
        [0]*6+[-1,0,-1,1,-1]+[0]*4,
        [0]*11+[1,0,0,0],
        [0]*15,
        [0]*15,
        [0]*15
    ]
    data013=[
        [0]*15,
        [0]*15,
        [0]*15,
        [0]*15,
        [0]*5+[-1]+[0]*5+[-1,0,0,0],
        [0]*6+[1,0,0,0,1,0,0,0,0],
        [0]*5+[-1,1,1,-1,1]+[0]*5,
        [0]*6+[1,-1,1]+[0]*6,
        [0]*6+[-1,1,-1,-1]+[0]*5,
        [0]*6+[-1,0,1,-1,1]+[0]*4,
        [0]*6+[-1,0,-1,1,-1]+[0]*4,
        [0]*11+[1,0,0,0],
        [0]*15,
        [0]*15,
        [0]*15
    ]
    data014=[
        [0]*15,
        [0]*15,
        [0]*15,
        [0]*15,
        [0]*5+[-1]+[0]*5+[-1,0,0,0],
        [0]*6+[1,0,0,0,1,0,0,0,0],
        [0]*5+[-1,1,1,-1,1]+[0]*5,
        [0]*6+[1,-1,1]+[0]*6,
        [0]*6+[-1,1,-1,-1]+[0]*5,
        [0]*6+[-1,0,1,-1,1]+[0]*4,
        [0]*6+[-1,0,-1,1,-1]+[0]*4,
        [0]*10+[1,1,0,0,0],
        [0]*15,
        [0]*15,
        [0]*15
    ]
    data1=np.array(data01).reshape(225,1)
    data2=np.array(data02).reshape(225,1)
    data3=np.array(data03).reshape(225,1)
    data4=np.array(data04).reshape(225,1)
    data5=np.array(data05).reshape(225,1)
    data6=np.array(data06).reshape(225,1)
    data7=np.array(data07).reshape(225,1)
    data8=np.array(data08).reshape(225,1)
    data9=np.array(data09).reshape(225,1)
    data10=np.array(data010).reshape(225,1)
    data11=np.array(data011).reshape(225,1)
    data12=np.array(data012).reshape(225,1)
    data13=np.array(data013).reshape(225,1)
    data14=np.array(data014).reshape(225,1)

    target01=[0.0]
    target02=[-4e-8]
    target03=[-4.5*1e-8]
    target04=[5*1e-8]
    target05=[-9.5*1e-8]
    target06=[4.55*1e-7]
    target07=[-8.5*1e-8]
    target08=[-9*1e-8]
    target09=[3.6*1e-7]
    target010=[-1.04*1e-6]
    target011=[-4.8*1e-7]
    target012=[0.099984565]
    target013=[0.099994515]
    target014=[1.0]
    """target1=np.array(target01).reshape(1,1)
    target2=np.array(target02).reshape(1,1)
    target3=np.array(target03).reshape(1,1)
    target4=np.array(target04).reshape(1,1)
    target5=np.array(target05).reshape(1,1)
    target6=np.array(target06).reshape(1,1)
    target7=np.array(target07).reshape(1,1)
    target8=np.array(target08).reshape(1,1)
    target9=np.array(target09).reshape(1,1)
    target10=np.array(target010).reshape(1,1)"""

    pat=[
        [data1,target01],
        [data2,target02],
        [data3,target03],
        [data4,target04],
        [data5,target05],
        [data6,target06],
        [data7,target07],
        [data8,target08],
        [data9,target09],
        [data10,target010],
        [data11,target011],
        [data12,target012],
        [data13,target013],
        [data14,target014]
    ]

    ga=GA.GA(pat)
    ga.run(200)
    n=ga.group.getBest()

    n.net.train(pat,1000,0.001,0.1)

    n.net.save_weights('d:/AI/evo_weights.txt')

    for i in range(COLUMN):
        for j in range(ROW):
            list_all.append((i+1,j+1))

    change=0
    g=0

    while g==0:
        if change%2==1:
            pos=ai(n.net)
            if pos in list3:
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

            list2.append((a2,b2))
            list3.append((a2,b2))

            if game_win(list2):
                print("You win!")
                g=1
            change=change+1   

main()