from math import *
import numpy as np

COLUMN=15
ROW=15

"""list1=[]#AI
list2=[]#human|AI2
list3=[]#all

list_all=[]#整个棋盘的点
next_point=[0,0]#AI下一步最应该走的位置

DEPTH=3#搜索深度

#棋型评估分
shape_score=[(50*1e-9,(0,1,1,0,0)),
             (50*1e-9,(0,0,1,1,0)),
             (200*1e-9,(1,1,0,1,0)),
             (500*1e-9,(0,0,1,1,1)),
             (500*1e-9,(1,1,1,0,0)),
             (5000*1e-9,(0,1,1,1,0)),
             (5000*1e-9,(0,1,0,1,1,0)),
             (5000*1e-9,(0,1,1,0,1,0)),
             (5000*1e-9,(1,1,1,0,1)),
             (5000*1e-9,(1,1,0,1,1)),
             (5000*1e-9,(1,0,1,1,1)),
             (5000*1e-9,(1,1,1,1,0)),
             (5000*1e-9,(0,1,1,1,1)),
             (5000*1e-9,(0,1,1,1,1,0)),
             (1e9*1e-9,(1,1,1,1,1))]

def ai(is_ai):
    global cut_count#剪枝次数
    cut_count=0
    global search_count#统计搜索次数
    search_count=0
    maxaiscore=negamax(is_ai,DEPTH,-1,1)
    print("本次剪枝："+str(cut_count))
    print("本次搜索："+str(search_count))
    print('maxscore='+str(maxaiscore))
    return next_point[0],next_point[1]

#负值极大搜索 剪枝
def negamax(is_ai,depth,alpha,beta):
    if(game_win(list1) or game_win(list2) or depth==0):
        return evaluation(is_ai)
    
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

        value=-negamax(not is_ai,depth-1,-beta,-alpha)
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
def evaluation(is_ai):
    total_score=0

    if is_ai:
        my_list=list1
        enemy_list=list2
    else:
        my_list=list2
        enemy_list=list1

    #算自己得分
    score_all_arr=[]
    my_score=0
    for pt in my_list:
        m=pt[0]
        n=pt[1]
        my_score+=cal_score(m,n,0,1,enemy_list,my_list,score_all_arr)
        my_score+=cal_score(m,n,1,0,enemy_list,my_list,score_all_arr)
        my_score+=cal_score(m,n,1,1,enemy_list,my_list,score_all_arr)
        my_score+=cal_score(m,n,-1,1,enemy_list,my_list,score_all_arr)

    #算敌人得分
    score_all_arr_enemy=[]
    enemy_score=0
    for pt in enemy_list:
        m=pt[0]
        n=pt[1]
        enemy_score+=cal_score(m,n,0,1,my_list,enemy_list,score_all_arr_enemy)
        enemy_score+=cal_score(m,n,1,0,my_list,enemy_list,score_all_arr_enemy)
        enemy_score+=cal_score(m,n,1,1,my_list,enemy_list,score_all_arr_enemy)
        enemy_score+=cal_score(m,n,-1,1,my_list,enemy_list,score_all_arr_enemy)

    total_score=my_score-enemy_score*0.1

    return total_score    

def cal_score(m,n,x_decrict,y_decrict,enemy_list,my_list,score_all_arr):
    add_score=0#加分项
    #在一个方向只取最大的分

    max_score_shape=(0,None)

    for item in score_all_arr:
        for pt in item[1]:
            if m==pt[0] and n==pt[1] and x_decrict==item[2][0] and y_decrict==item[2][1]:
                return 0
    
    #在落子点左右方向循环查找得分形状
    for offset in range(-5,1):
        pos=[]
        for i in range(0,6):
            if (m+(i+offset)*x_decrict,n+(i+offset)*y_decrict) in enemy_list:
                pos.append(2)
            elif (m+(i+offset)*x_decrict,n+(i+offset)*y_decrict) in my_list:
                pos.append(1)
            else:
                pos.append(0)
        tmp_shap5=(pos[0],pos[1],pos[2],pos[3],pos[4])
        tmp_shap6=(pos[0],pos[1],pos[2],pos[3],pos[4],pos[5])

        for (score,shape) in shape_score:
            if tmp_shap5==shape or tmp_shap6==shape:
                if score>max_score_shape[0]:
                    max_score_shape=(score,((m+(0+offset)*x_decrict,n+(0+offset)*y_decrict),
                                            (m+(1+offset)*x_decrict,n+(1+offset)*y_decrict),
                                            (m+(2+offset)*x_decrict,n+(2+offset)*y_decrict),
                                            (m+(3+offset)*x_decrict,n+(3+offset)*y_decrict),
                                            (m+(4+offset)*x_decrict,n+(4+offset)*y_decrict)),
                                            (x_decrict,y_decrict))
                
    #计算两个形状相交
    if max_score_shape[1] is not None:
        for item in score_all_arr:
            for pt1 in item[1]:
                for pt2 in max_score_shape[1]:
                    if pt1==pt2 and max_score_shape[0]>10 and item[0]>10:
                        add_score+=item[0]+max_score_shape[0]
                
        score_all_arr.append(max_score_shape)

    return add_score+max_score_shape[0]


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
"""

class Gametree:
    def __init__(self,depth):
        self.list1=[]#AI
        self.list2=[]#human|AI2
        self.list3=[]#all
        self.list_all=[]#整个棋盘的点
        self.next_point=[0,0]#AI下一步最应该走的位置

        self.DEPTH=depth#搜索深度

        #棋型评估分
        self.shape_score=[(50*1e-9,(0,1,1,0,0)),
                (50*1e-9,(0,0,1,1,0)),
                (200*1e-9,(1,1,0,1,0)),
                (500*1e-9,(0,0,1,1,1)),
                (500*1e-9,(1,1,1,0,0)),
                (5000*1e-9,(0,1,1,1,0)),
                (5000*1e-9,(0,1,0,1,1,0)),
                (5000*1e-9,(0,1,1,0,1,0)),
                (5000*1e-9,(1,1,1,0,1)),
                (5000*1e-9,(1,1,0,1,1)),
                (5000*1e-9,(1,0,1,1,1)),
                (5000*1e-9,(1,1,1,1,0)),
                (5000*1e-9,(0,1,1,1,1)),
                (5000*1e-9,(0,1,1,1,1,0)),
                (1e9*1e-9,(1,1,1,1,1))]

    def ai(self,is_ai):
        global cut_count#剪枝次数
        cut_count=0
        global search_count#统计搜索次数
        search_count=0
        maxaiscore=self.negamax(is_ai,self.DEPTH,-1,1)
        print("本次剪枝："+str(cut_count))
        print("本次搜索："+str(search_count))
        print('maxscore='+str(maxaiscore))
        return self.next_point[0],self.next_point[1]

    #负值极大搜索 剪枝
    def negamax(self,is_ai,depth,alpha,beta):
        if(self.game_win(self.list1) or self.game_win(self.list2) or depth==0):
            return self.evaluation(is_ai)

        blank_list=list(set(self.list_all).difference(set(self.list3)))
        #遍历每一个可能的位置
        for next_step in blank_list:
            global search_count
            search_count+=1
            
            if not self.has_neightnor(next_step):
                continue

            if is_ai:
                self.list1.append(next_step)
            else:
                self.list2.append(next_step)
            self.list3.append(next_step)

            value=-self.negamax(not is_ai,depth-1,-beta,-alpha)
            if is_ai:
                self.list1.remove(next_step)
            else:
                self.list2.remove(next_step)
            self.list3.remove(next_step)

            if value>alpha:
                if depth==self.DEPTH:
                    self.next_point[0]=next_step[0]
                    self.next_point[1]=next_step[1]
                if value>=beta:
                    global cut_count
                    cut_count+=1
                    return beta
                alpha=value
                
        return alpha

    def has_neightnor(self,pt):
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                if (pt[0] + i, pt[1]+j) in self.list3:
                    return True
        return False

    #评估函数
    def evaluation(self,is_ai):
        total_score=0

        if is_ai:
            my_list=self.list1
            enemy_list=self.list2
        else:
            my_list=self.list2
            enemy_list=self.list1

        #算自己得分
        score_all_arr=[]
        my_score=0
        for pt in my_list:
            m=pt[0]
            n=pt[1]
            my_score+=self.cal_score(m,n,0,1,enemy_list,my_list,score_all_arr)
            my_score+=self.cal_score(m,n,1,0,enemy_list,my_list,score_all_arr)
            my_score+=self.cal_score(m,n,1,1,enemy_list,my_list,score_all_arr)
            my_score+=self.cal_score(m,n,-1,1,enemy_list,my_list,score_all_arr)

        #算敌人得分
        score_all_arr_enemy=[]
        enemy_score=0
        for pt in enemy_list:
            m=pt[0]
            n=pt[1]
            enemy_score+=self.cal_score(m,n,0,1,my_list,enemy_list,score_all_arr_enemy)
            enemy_score+=self.cal_score(m,n,1,0,my_list,enemy_list,score_all_arr_enemy)
            enemy_score+=self.cal_score(m,n,1,1,my_list,enemy_list,score_all_arr_enemy)
            enemy_score+=self.cal_score(m,n,-1,1,my_list,enemy_list,score_all_arr_enemy)

        total_score=my_score-enemy_score*0.1

        return total_score    

    def cal_score(self,m,n,x_decrict,y_decrict,enemy_list,my_list,score_all_arr):
        add_score=0#加分项
        #在一个方向只取最大的分

        max_score_shape=(0,None)

        for item in score_all_arr:
            for pt in item[1]:
                if m==pt[0] and n==pt[1] and x_decrict==item[2][0] and y_decrict==item[2][1]:
                    return 0

        #在落子点左右方向循环查找得分形状
        for offset in range(-5,1):
            pos=[]
            for i in range(0,6):
                if (m+(i+offset)*x_decrict,n+(i+offset)*y_decrict) in enemy_list:
                    pos.append(2)
                elif (m+(i+offset)*x_decrict,n+(i+offset)*y_decrict) in my_list:
                    pos.append(1)
                else:
                    pos.append(0)
            tmp_shap5=(pos[0],pos[1],pos[2],pos[3],pos[4])
            tmp_shap6=(pos[0],pos[1],pos[2],pos[3],pos[4],pos[5])

            for (score,shape) in self.shape_score:
                if tmp_shap5==shape or tmp_shap6==shape:
                    if score>max_score_shape[0]:
                        max_score_shape=(score,((m+(0+offset)*x_decrict,n+(0+offset)*y_decrict),
                                                (m+(1+offset)*x_decrict,n+(1+offset)*y_decrict),
                                                (m+(2+offset)*x_decrict,n+(2+offset)*y_decrict),
                                                (m+(3+offset)*x_decrict,n+(3+offset)*y_decrict),
                                                (m+(4+offset)*x_decrict,n+(4+offset)*y_decrict)),
                                                (x_decrict,y_decrict))
                    
        #计算两个形状相交
        if max_score_shape[1] is not None:
            for item in score_all_arr:
                for pt1 in item[1]:
                    for pt2 in max_score_shape[1]:
                        if pt1==pt2 and max_score_shape[0]>10 and item[0]>10:
                            add_score+=item[0]+max_score_shape[0]
                    
            score_all_arr.append(max_score_shape)

        return add_score+max_score_shape[0]


    def game_win(self,list):
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
    gametree=Gametree(3)#搜索深度
    for i in range(COLUMN):
        for j in range(ROW):
            gametree.list_all.append((i+1,j+1))

    change=0
    g=0
    modle=eval(input("输入游戏模式，输入1为人机对抗，输入2为AI互相对抗\n"))
    while g==0:
        if change%2==1:
            pos=gametree.ai(True)
            if pos in gametree.list3:
                g=1

            print("AI落子(-)位置为："+str(pos))

            gametree.list1.append(pos)
            gametree.list3.append(pos)
            
            for i in range(ROW+1):
                for j in range(COLUMN+1):
                    if(i==0):
                        print("%-3d"%(j),end='')
                    else:
                        if(j==0):
                            print("%-3d"%(i),end='')
                        else:
                            if (i,j) in gametree.list1:
                                print("%-3s"%("-"),end='')
                            elif (i,j) in gametree.list2:
                                print("%-3s"%("+"),end='')
                            else:
                                print("%-3s"%("0"),end='')
                print("")

            if gametree.game_win(gametree.list1):
                print("AI win!")
                g=1
            change=change+1
        else:
            if(modle==1):
                print("请输入你要走的位置坐标。格式为：'x,y'，范围为[1,15]")
                a2,b2=eval(input())

                if((a2,b2) in gametree.list3):
                    print("该位置已有棋子！")
                    continue
                if(a2>15 or a2<1 or b2>15 or b2<1):
                    print("落子位置不合法！")
                    continue

                gametree.list2.append((a2,b2))
                gametree.list3.append((a2,b2))

                if gametree.game_win(gametree.list2):
                    print("You win!")
                    g=1
                change=change+1   
            elif(modle==2):
                pos=gametree.ai(False)
                if pos in gametree.list3:
                    g=1

                print("AI2落子(+)位置为："+str(pos))

                gametree.list2.append(pos)
                gametree.list3.append(pos)
            
                for i in range(ROW+1):
                    for j in range(COLUMN+1):
                        if(i==0):
                            print("%-3d"%(j),end='')
                        else:
                            if(j==0):
                                print("%-3d"%(i),end='')
                            else:
                                if (i,j) in gametree.list1:
                                    print("%-3s"%("-"),end='')
                                elif (i,j) in gametree.list2:
                                    print("%-3s"%("+"),end='')
                                else:
                                    print("%-3s"%("0"),end='')
                    print("")

                if gametree.game_win(gametree.list2):
                    print("AI2 win!")
                    g=1
                change=change+1

if __name__=="__main__":
    main()