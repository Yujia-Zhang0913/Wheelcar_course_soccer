import numpy as np
import math
from plyfile import PlyData
import matplotlib.pyplot as plt
from icp import ICP
from car import Car


"""
ply2np:读取给定路径的.ply文件并返回对应的np.array变量,shape=(3,n)
"""
def ply2np(filename):
    plydata=PlyData.read(filename).elements[0].data
    point_num=plydata.shape[0]
    npdata=np.zeros([3,point_num],dtype=np.float64)
    # print(plydata)
    for idx,data in enumerate(plydata):
        npdata[0,idx]=data[0]
        npdata[1,idx]=data[1]
        npdata[2,idx]=data[2]
    return npdata

#读取各帧点云数据
npdata_dict={}
for i in range(10):
    npdata_dict[i]=ply2np(str(i)+'.ply')
    # print(npdata_dict[i].shape)

#利用ICP计算各帧间的变换R和t，并计算和第0帧间的变换矩阵T
icp=ICP()
transform_dict={'T_0_0':np.eye(4)}
for i in range(9):
    print("-------frame: "+str(i)+"->frame"+str(i+1)+"---------")
    A=npdata_dict[i]
    B=npdata_dict[i+1]
    R,t=icp.calcTransform(A,B)
    # transform_dict['R_'+str(i+1)+'_'+str(i)]=R
    # transform_dict['t_'+str(i+1)+'_'+str(i)]=t
    T=np.hstack([R,t])
    T=np.vstack([ T,np.array([0,0,0,1]) ])
    transform_dict['T_'+str(i+1)+'_'+str(i)]=T
    if i>0:
        transform_dict['T_'+str(i+1)+'_0']=np.dot(transform_dict['T_'+str(i)+'_0'],transform_dict['T_'+str(i+1)+'_'+str(i)])

# print(transform_dict)
# print(transform_dict['T_1_0'])
for i in range(9):
    print('T_'+str(i+1)+'_'+str(i),transform_dict['T_'+str(i+1)+'_'+str(i)])
    print('T_'+str(i+1)+'_0',transform_dict['T_'+str(i+1)+'_0'])

#计算机器人在世界坐标系下的坐标序列
robot_info={'x0':0,'y0':0,'theta0':0}
for i in range(1,10):
    T=transform_dict['T_'+str(i)+'_0']
    robot_info['x'+str(i)]=T[0,3]
    robot_info['y'+str(i)]=T[1,3]
    robot_info['theta'+str(i)]=math.atan2(T[1,0],T[0,0]) #x=atan2(sinx,cosx)

print(robot_info)


car=Car()
points_list=[]
for i in range(10):
    points_bodyframe=np.vstack([npdata_dict[i],np.ones([1,180])]) #第i帧体坐标系下的坐标
    points=np.dot(transform_dict['T_'+str(i)+'_0'],points_bodyframe)#将各帧点云转换到世界坐标系下
    # print(points.shape)
    # if i==1:
    #     print(points[0])
    points_list.append(points)
    car.draw(robot_info['x'+str(i)],robot_info['y'+str(i)],robot_info['theta'+str(i)],points)

for i in range(10):
    plt.scatter(points_list[i][0],points_list[i][1],c=[(0.1*i,1-0.1*i,1,0.5)],s=[1])
plt.xlim([-2,6])
plt.ylim([0,8])
# plt.scatter(2,2,c=(0,1,1,0.5),s=9)
plt.show()