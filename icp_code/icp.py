import numpy as np
import matplotlib.pyplot as plt

class ICP:
    def __init__(self) -> None:
        self.max_error=0.05
        self.max_iter=5
        self.outlier_th=0.75
        self.R0=np.eye(3)
        self.t0=np.zeros([3,1])

    def calcTransform(self,A,B,delete_outliers=0):
        #核心函数，由两帧点云A、B计算变换R、t
        #param: A0 当前帧，元素ai
        #param: B0 目标帧（下一帧），元素bi
        #目标: 最小化Σ(ai-Rbi-t)
        R=self.R0
        t=self.t0
        iter=0
        error=1
        while(error>self.max_error and iter<self.max_iter):
            A_new,B_new=self.findNearest(A,B) #重新排列A和B
            R,t=self.stepForward(A_new,B_new) #按线性代数方法计算R和t

            error,outliers=self.calcError(A_new,B_new,R,t) #R和t对应的误差
            if delete_outliers==False:
                outliers=np.zeros(180,dtype=np.int)
            print("point_num:",outliers.reshape(-1).shape[0])
            print("error=",error)
            A=np.delete(A_new,outliers,axis=1) #删除outlier
            B=np.delete(B_new,outliers,axis=1)
            iter+=1
        print(R)
        self.draw(A,B,R,t)
        return R,t
    
    def findNearest(self,A,B):
        #寻找两帧数据A、B之间的最近邻匹配，并分别按顺序排列

        # B = np.array(B[:2,:])
        # A = np.array(A[:2,:])

        #以下一段进行匹配，同时要注意清洗数据，有nan的要跳过。B_new和A_new保存匹配后的数据，二者列数相同
        B_new = np.zeros([3,1])
        A_new = np.zeros([3,1])
        if (B.shape[1] >= A.shape[1]):#B列数多于A列数,则遍历A
            for i in range(A.shape[1]):
                #print(A[:,i].reshape(3,1))
                if(np.isnan(A[0][i])==True): #是nan则跳过
                    continue
                dist=self.calcDist(A[:,i].reshape(3,1),B) #A中第i个点和B中各个点的距离
                #print(i,dist)
                index=np.nanargmin(dist) #找到离A中第i个点最近的
                A_point=A[:,i].reshape(3,1)
                B_point=B[:,index].reshape(3,1)
                A_new=np.hstack([A_new,A_point])
                B_new=np.hstack([B_new,B_point])
            A_new=np.delete(A_new,0,axis=1)#删除初始化的第一列0
            B_new=np.delete(B_new,0,axis=1)
            return A_new,B_new
        else:#B列数小于A列数,则遍历B
            for i in range(B.shape[1]):
                #print(A[:,i].reshape(3,1))
                if(np.isnan(B[0][i])==True): #是nan则跳过
                    continue
                dist=self.calcDist(B[:,i].reshape(3,1),A) #A中第i个点和B中各个点的距离
                #print(i,dist)
                index=np.nanargmin(dist)
                A_point=A[:,index].reshape(3,1)
                B_point=B[:,i].reshape(3,1)
                A_new=np.hstack([A_new,A_point])
                B_new=np.hstack([B_new,B_point])
            A_new=np.delete(A_new,0,axis=1)#删除初始化的第一列0
            B_new=np.delete(B_new,0,axis=1)
            return A_new,B_new

    def stepForward(self,A,B):
        #param: A 当前帧，已排序
        #param: B 下一帧，已排序
        A_avg=np.average(A,axis=1).reshape(3,1) #3*1
        B_avg=np.average(B,axis=1).reshape(3,1) #3*1
        # print("A_avg",A_avg)
        # print("B_avg",B_avg)
        A_ce=A-A_avg #去均值化，对应q矩阵 #3*n
        B_ce=B-B_avg #去均值化，对应q'矩阵 #3*n
        # print("A_ce",A_ce)
        # print("B_ce",B_ce)
        W=np.matmul(B_ce,A_ce.T)
        # print("W",W)
        U,S,VT=np.linalg.svd(W) #svd分解
        R=np.matmul(VT.T,U.T) #V*UT  3*3
        t=A_avg-np.matmul(R,B_avg) #3*1
        # T=np.hstack([R,t])
        return R,t

    def calcError(self,A,B,R,t):

        error_mat=A-np.dot(R,B)-t #(3,n)
        dist=np.linalg.norm(error_mat,axis=0) #(n,)
        outliers=dist>self.outlier_th
        # print(dist.shape)
        return np.average(dist),outliers
    
    def calcDist(self,p,p_list):
        #计算向量p和向量组p_list中各个向量的距离
        #param: p (3,1)
        #param: p_list (3,n)
        return np.linalg.norm(p_list-p,axis=0)
    
    def draw(self,A,B,R,t):
        #画出A帧（A系）、B帧（A系）、B帧（B系）
        B_w=np.dot(R,B)+t
        plt.scatter(A[0],A[1],c='b')
        plt.scatter(B_w[0],B_w[1],c='r')
        plt.scatter(B[0],B[1],c='y')
        plt.xlim([-10,10])
        plt.ylim([-10,10])
        plt.show()
    
R=np.eye(3)
t=np.zeros([3,1])
A=np.ones([3,10])
B=np.ones([3,10])
icp=ICP()
print(icp.calcError(A,B,R,t))

