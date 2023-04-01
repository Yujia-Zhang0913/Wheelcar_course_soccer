import numpy as np
# import pyomo 
# from cvxopt import solvers, matrix

# using conda install -c conda-forge cvxopt
# to install cvxopt and dependencies
# These pre-built packages are linked against OpenBLAS and include all the optional extensions (DSDP, FFTW, GLPK, and GSL).

import casadi 

# import numpy as np
# from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline

class MPCPredict():
    def __init__(self,pace=100,m=4,p=6,Q=1,H=1) -> None:
        # 同学们，导航规划作业中，机器人最大加速度4000mm/s^2 最大速度3500mm/s
        # 最大角加速度20rad/s^2 最大角速度15rad/s
        self.m=m 
        self.p=p
        self.Q=1
        self.H=1
        
        self.controlDt=0.02
        self.a_max=4000
        self.alpha_max=15
        self.uv_max=3500
        self.uw_max=15
        self.duv_max=self.a_max*self.controlDt
        self.duw_max=self.alpha_max*self.controlDt
        
        self.pace=pace
        # pace mm 

        self.CB=np.zeros((2,2))

        self.__initMX()
        pass
    def __initMX(self):
        self.su_=np.zeros((self.p,self.m))
        for i in range(self.p):
            for j in range(self.m):
                if (i>=j):
                    self.su_[i,j]=i+1-j 
        self.su_=self.su_.repeat(2,axis=1).repeat(2,axis=0)
        self.sx_=np.arange(1,self.p+1)
        self.sx_=self.sx_.reshape((self.p,1)).repeat(2,axis=1)
        print("su_",self.su_)

        self.ubx_=np.tile(np.array([self.duv_max,self.duw_max]),(1,self.p))
        self.lbx_=np.tile(np.array([-self.duv_max,-self.duw_max]),(1,self.p))

        self.lba_0=np.tile(np.array([-self.uv_max,-self.uw_max]),(1,self.p))
        self.uba_0=np.tile(np.array([self.uv_max,self.uw_max]),(1,self.p))

    def __reMX(self):
        self.CB=np.array([np.cos(self.theta),0],[np.sin(self.theta),0])
        self.su=np.multiply(self.su_,np.tile(self.CB,(self.m,self.p)))

        self.H_qp=2*self.Q*self.su.T*self.su.T+self.H 

        self.__reRefPath()
        self.sx=np.multiply(self.sx_+1,np.tile(np.array([self.delta_x,self.delta_y]).reshape(1,2),(self.p,1)))
        # 这里因为 y就是x的前两个维度  所以将sx delta x k 与 I yk合并
        self.E_qp_=-(self.sx+np.tile(np.array([self.x,self.y]).reshape(1,2),(self.p,1)))
        self.g_qp=-2*self.Q*self.su_*(self.R_qp+self.E_qp)

        self.uba_=self.uba_0+self.E_qp_
        self.lba_=self.lba_0+self.E_qp_
        self.A_qp=np.ones((self.p*2,1))

    def reCurrentState(self,x,y,v,w):
        self.delta_x=x-self.x
        self.delta_y=y-self.y
        self.x=x 
        self.y=y 
        self.v=v 
        self.w=w 
    def reObstacles(self,obstacles):
        self.obstacles=obstacles
        # the obstacles are described in the robot world 
    def rePredict(self ):
        # self.__reRefPath()
        self.__reMX()


        H = casadi.DM(self.H_qp)
        A = casadi.DM(self.A_qp)
        g = casadi.DM(self.g_qp)
        lbx=casadi.DM(self.lbx_)
        ubx=casadi.DM(self.ubx_)
        lba=casadi.DM(self.lba_)
        uba=casadi.DM(self.uba_)
        # lba = .

        qp = {}
        qp['h'] = H.sparsity()
        qp['a'] = A.sparsity()
        S = casadi.conic('S','qpoases',qp)
        print(S)


        r = S(h=H, g=g, a=A, lbx=lbx, ubx=ubx,a=A,lba=lba,uba=uba)
        x_opt = r['x']
        print('x_opt: ', x_opt)

        self.Control()
        pass
    def __reRefPath(self):
        self.R_qp=np.zeros((self.p,2))
        pass 
    def RefreshPath(self,path_x,path_y):
        print(self.path.shape)
        seg_x=np.abs(self.path[1:,0]-self.path[:-1,0])
        seg_y=np.abs(self.path[1:,1]-self.path[:-1,1])
        x=(seg_x+seg_y)/self.pace
        self.Length=np.sum(seg_x)+np.sum(seg_y)
        self.steps=self.Length//self.pace+1

        self.xpos=np.interp.int

        #平滑处理后
        x_smooth = np.linspace(0, self.steps, self.steps)  # np.linspace 等差数列,从x.min()到x.max()生成300个数，便于后续插值
        y_smooth = make_interp_spline(t, path_x)(x_smooth)


        pass 
    def Control(self):
        

        pass 


# def test_for_cvxopt():
    
#     # import numpy as np

#     P = matrix(np.diag([1.0,0]))  #  对于一些特殊矩阵，用numpy创建会方便很多（在本例中可能感受不大）
#     q = matrix(np.array([3.0,4]))
#     G = matrix(np.array([[-1.0,0],[0,-1],[-1,-3],[2,5],[3,4]]))
#     h = matrix(np.array([0.0,0,-15,100,80]))
#     sol = solvers.qp(P,q,G,h)
#     return sol



if __name__=="__main__":
    predictor=MPCPredict()
