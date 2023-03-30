import matplotlib.pyplot as plt
import numpy as np

class Car:
    def __init__(self) -> None:
        self.car_size=0.3
        self.x_list=[]
        self.y_list=[]
        self.theta_list=[]
    def draw(self,x,y,theta,points):
        #计算小车的三个顶点组坐标
        theta=theta+np.pi/2
        p1=np.array([x,y])+np.array([2*self.car_size*np.cos(theta),2*self.car_size*np.sin(theta)])
        p2=np.array([x,y])+np.array([np.sqrt(2)*self.car_size*np.cos(theta+np.pi*0.75),np.sqrt(2)*self.car_size*np.sin(theta+np.pi*0.75)])
        p3=np.array([x,y])+np.array([np.sqrt(2)*self.car_size*np.cos(theta-np.pi*0.75),np.sqrt(2)*self.car_size*np.sin(theta-np.pi*0.75)])
        #画出小车
        plt.scatter(x,y,c='r',s=0.3)
        plt.plot([p1[0],p2[0]],[p1[1],p2[1]],c='r')
        plt.plot([p2[0],p3[0]],[p2[1],p3[1]],c='r')
        plt.plot([p3[0],p1[0]],[p3[1],p1[1]],c='r')
        #画出激光点云
        plt.scatter(points[0],points[1])
        # plt.xlim([-2,5])
        # plt.ylim([-2,5])
        plt.xlim([-2,8])
        plt.ylim([-2,8])
        #保存坐标数据并画出历史轨迹
        self.x_list.append(x)
        self.y_list.append(y)
        self.theta_list.append(theta)
        plt.scatter(self.x_list,self.y_list,c='r',s=0.3)
        plt.show()