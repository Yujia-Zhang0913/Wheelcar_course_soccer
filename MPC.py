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
    def __init__(self,x,y,theta,pace=100,m=4,p=10,Q=1,H=1) -> None:
        # 同学们，导航规划作业中，机器人最大加速度4000mm/s^2 最大速度3500mm/s
        # 最大角加速度20rad/s^2 最大角速度15rad/s
        self.x=x 
        self.y=y
        self.theta=theta
        # calculate in radian

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

        self.Debugger=None
        self.action=None
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

        self.ubx_=np.tile(np.array([self.duv_max,self.duw_max]),(self.m,1)).reshape(2*self.m,1)
        self.lbx_=np.tile(np.array([-self.duv_max,-self.duw_max]),(self.m,1)).reshape(2*self.m,1)

        self.lba_0=np.tile(np.array([-self.uv_max,-self.uw_max]),(self.p,1)).reshape(2*self.p,1)
        self.uba_0=np.tile(np.array([self.uv_max,self.uw_max]),(self.p,1)).reshape(2*self.p,1)

        self.R_qp=np.zeros((self.p*2,1))
        self.R_qp_index=np.zeros(self.p).astype(int)
        for i in range(self.p):
            self.R_qp_index[i]=i*2
        # self.R_qp_index.astype(int)

    def __reMX(self):
        # qp means the matrix used for QP splver
        self.CB=np.array([[np.cos(self.theta),0],[np.sin(self.theta),0]])
        # print(np.tile(self.CB,(self.m,self.p)))
        self.su=np.multiply(self.su_,np.tile(self.CB,(self.p,self.m)))

        self.H_qp=2*self.Q*np.matmul(self.su.T,self.su)+self.H 

        self.__reRefPath()
        self.sx=np.multiply(self.sx_+1,np.tile(np.array([self.delta_x,self.delta_y]).reshape(1,2),(self.p,1)))
        # 这里因为 y就是x的前两个维度  所以将sx delta x k 与 I yk合并
        self.E_qp_=-(self.sx+np.tile(np.array([self.x,self.y]).reshape(1,2),(self.p,1))).reshape(2*self.p,1)
        # print(self.E_qp_,self.R_qp.shape)
        self.g_qp=-2*self.Q*np.matmul(self.su_.T,(self.R_qp+self.E_qp_).reshape(2*self.p,1))

        # self.su_inv=self.su.i
        self.uba_=self.uba_0+self.E_qp_
        self.lba_=self.lba_0+self.E_qp_
        # print(self.lba_.shape,self.uba_.shape)
        self.A_qp=self.su

    def reCurrentState(self,x,y,theta,v,w):
        self.delta_x=x-self.x
        self.delta_y=y-self.y
        self.delta_theta=theta-self.theta
        self.x=x 
        self.y=y 
        self.theta=theta
        self.v=v 
        self.w=w 
    def reObstacles(self,obstacles):
        self.obstacles=obstacles
        # the obstacles are described in the robot world 
    def rePredict(self ):
        # self.__reRefPath()
        self.__reMX()

        print("\n----------------H_qp-------------\n",self.H_qp)
        print("\n----------------A_qp-------------\n",self.A_qp)
        print("\n----------------g_qp-------------\n",self.g_qp)
        print("\n----------------lbx_-------------\n",self.lbx_)
        print("\n----------------ubx_-------------\n",self.ubx_)
        print("\n----------------lba_-------------\n",self.lba_)
        print("\n----------------uba_-------------\n",self.uba_)

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


        r = S(h=H, g=g, a=A, lbx=lbx, ubx=ubx,lba=lba,uba=uba)
        x_opt = r['x']
        print('x_opt: ', x_opt)

        self.Control(x_opt[0],x_opt[1])
        pass
    def __reRefPath(self):
        
        self.R_qp[self.R_qp_index,0]=self.xpos[self.curSteps:self.curSteps+self.p]
        self.R_qp[self.R_qp_index+1,0]=self.ypos[self.curSteps:self.curSteps+self.p]
        
        self.curSteps+=1
        
        pass 
    def RefreshPath(self,path_x,path_y):
        # path_x=np.array(path_x)
        # path_y=np.array(path_y)
        print(path_x.shape)
        seg_x=np.abs(path_x[1:]-path_x[:-1])
        seg_y=np.abs(path_y[1:]-path_y[:-1])
        t=np.zeros(path_x.shape[0])
        # t[0]=0.
        t[1:]=(seg_x+seg_y)/self.pace
        for i in range(1,path_x.shape[0]):
            if t[i]<=0:
                t[i]=0.1
            t[i]+=t[i-1]
        print('seg x \n',seg_x,'seg y \n',seg_y,'t \n',t)
        self.Length=np.sum(seg_x)+np.sum(seg_y)
        self.steps=self.Length//self.pace

        # self.xpos=np.interp.int

        #平滑处理后
        steps = np.linspace(0.1, self.steps-0.01, int(self.steps))  # np.linspace 等差数列,从x.min()到x.max()生成300个数，便于后续插值
        print('steps\n',steps)
        # print(make_interp_spline(t, path_x))
        # self.pos = make_interp_spline(t, path)(steps)
        self.xpos = make_interp_spline(t, path_x)(steps)
        self.ypos = make_interp_spline(t, path_y)(steps)
        self.curSteps=0

        pass 
    def setAction(self,action):
        self.action=action
    def setDebugger(self,Debugger):
        self.Debugger=Debugger
    def Control(self,vx,vw):
        self.action.sendCommand(vx=vx, vy=0, vw=vw)

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
    predictor=MPCPredict(0,0,0.2)
    predictor.setAction()
    predictor.reCurrentState(0,0,0,0,0)
    predictor.RefreshPath([100,200,300,600],[100,300,400,200])
    predictor.rePredict()
