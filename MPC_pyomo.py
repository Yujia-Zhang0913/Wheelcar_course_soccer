import numpy as np
import scipy
# import pyomo 
# from cvxopt import solvers, matrix

# using conda install -c conda-forge cvxopt
# to install cvxopt and dependencies
# These pre-built packages are linked against OpenBLAS and include all the optional extensions (DSDP, FFTW, GLPK, and GSL).

# import pyomo 
from pyomo.environ import *
import pyomo as pyo
# which can be download here 
# https://www.coin-or.org/download/binary/Ipopt/

from scipy.interpolate import make_interp_spline
# pyo.core.base.block.add_component()
class MPCPredict():
    def __init__(self,x,y,theta,pace=80,m=10,p=20,Q=1,H=1) -> None:
        # 同学们，导航规划作业中，机器人最大加速度4000mm/s^2 最大速度3500mm/s
        # 最大角加速度20rad/s^2 最大角速度15rad/s

        self.x=x 
        self.y=y
        self.theta=theta
        # state of x 
        # self.model.x_state=Param(RangeSet(0,2),initialize={0:x,1:y,2:theta},mutable=True)
        # print("ini state",self.x,self.y,self.theta)

        # calculate in radian

        self.m=m 
        # self.m=Param()
        self.p=p
        # coefficient of Q and P 

        self.Q=10
        # 单次规划位置与期望位置的值
        self.H=0.1
        # 单次规划的速度与角度变化率
        self.R=0.1
        # obstacle
        self.S=0.1
        # 与上一次规划的差值

        self.predictDt=1
        self.controlDt=0.4
        self.a_max=500
        self.alpha_max=0.5

        self.duv_max=self.a_max*self.controlDt
        self.duw_max=self.alpha_max*self.controlDt
        
        self.pace=pace
        # pace mm 

        # self.CB=np.zeros((2,2))
        self.v_opt=np.zeros(self.m)
        for i in range(self.m):
            self.v_opt[i]=60
        self.w_opt=np.zeros(self.m)

        self.__initMX()

        self.debugger=None
        self.action=None
        self.notFisrtPlan=False
        self.optimizer=SolverFactory('ipopt', executable=r'E:\ProgramFiles\Ipopt-3.11.1\bin\ipopt.exe')
        # self.optimizer.options["threads"] = 8
        
        pass
    def __limitTheta(self,theta):
        if theta>np.pi:
            theta-=int(theta/np.pi)*np.pi
        elif theta<-np.pi:
            theta+=int(theta/np.pi)*np.pi
        return theta
    def __initMX(self):


        self.R_qp_x=np.zeros(self.p)
        self.R_qp_y=np.zeros(self.p)
        self.pre_v=np.zeros(self.p)
        self.pre_w=np.zeros(self.p)
        self.pre_theta=np.zeros(self.p)
        self.pre_x=np.zeros(self.p)
        self.pre_y=np.zeros(self.p)     
    def reCurrentState(self,x,y,theta,v,w):
        # self.model.delta_x=Param(initialize=x-self.model.x)
        # self.model.delta_y=Param(initialize=y-self.model.y)
        # self.model.delta_theta=Param(initialize=theta-self.model.theta)

        self.delta_x=x-self.x
        self.delta_y=y-self.y
        self.delta_theta=theta-self.theta

        # self.model.x=Param(initialize=x) 
        # self.model.y=Param(initialize=y) 
        # self.model.theta=Param(initialize=theta)
        self.x=x 
        self.y=y
        self.theta=theta
        v=np.sqrt(self.delta_x**2+self.delta_x**2)/self.controlDt
        w=self.__limitTheta(self.delta_theta) /self.controlDt


        self.v=v 
        self.w=w 
        print("--------v w ----------",self.v,self.w)
    def reObstacles(self,obstacles):
        self.obstacles=obstacles
        # the obstacles are described in the robot world 
    def __selectObstaclss(self):
        dis=scipy.spatial.distance.cdist(np.array((self.x,self.y)).reshape(1,2), self.obstacles, metric='euclidean')
        # print(dis)
        self.selectedObs=np.zeros(15)
        for i in range(15):
            self.selectedObs[i]=(dis[:,i]<1000)
        print("---------self.selectedObs---------",self.selectedObs)
        pass 
    def rePredict(self ):
        if self.curSteps+self.p<self.steps:
            # self.__reRefPath()


            # Parameters
            #需要调节的参数如下
            # Constraint
            self.model = ConcreteModel()
            self.model.x = Param(initialize=self.x)
            self.model.y = Param(initialize=self.y)
            self.model.theta= Param(initialize=self.theta)
            self.model.uv_max=Param(initialize=3500)
            self.model.uw_max=Param(initialize=0.2)


            self.model.Q=Param(initialize=self.Q,mutable=True)
            self.model.H=Param(initialize=self.H,mutable=True)
            self.model.R=Param(initialize=self.R,mutable=True)
            self.model.S=Param(initialize=self.S,mutable=True)

            self.model.v=Param(initialize=self.v) 
            self.model.w=Param(initialize=self.w)

            
            self.model.ind_m=RangeSet(0,self.m-1)
            self.model.ind_p=RangeSet(0,self.p-1)

            
            self.model.u_delta_v=Var(self.model.ind_m,bounds=(-self.duv_max,self.duv_max))
            self.model.u_delta_w=Var(self.model.ind_m,bounds=(-self.duw_max,self.duw_max))
            self.__reRefPath()
            self.__selectObstaclss()
            # self.model.u_v=Var(self.model.ind)
            # self.model.u_w=Var(self.model.ind)
            # self.model.u_x=Var(self.model.ind)
            # self.model.u_y=Var(self.model.ind)
            def vmaxConstraint(model,k):
                sum_delta_v=0
                for i in model.ind_m:
                    if i<=k:
                        sum_delta_v+=model.u_delta_v[i]
                return model.uv_max-model.v-sum_delta_v>=0
            def vminConstraint(model,k):
                sum_delta_v=0
                for i in model.ind_m:
                    if i<=k:
                        sum_delta_v+=model.u_delta_v[i]
                return  +model.uv_max+model.v+sum_delta_v>=0
            # def vminConstraint(model,k):
            #     sum_delta_v=0
            #     for i in model.ind_m:
            #         if i<=k:
            #             sum_delta_v+=model.u_delta_v[i]
            #     return  model.v+sum_delta_v>=0
            def wmaxConstraint(model,k):
                sum_delta_w=0
                for i in model.ind_m:
                    if i<=k:
                        sum_delta_w+=model.u_delta_w[i]
                return model.uw_max-model.w-sum_delta_w>=0
            def wminConstraint(model,k):
                sum_delta_w=0
                for i in model.ind_m:
                    if i<=k:
                        sum_delta_w+=model.u_delta_w[i]
                return +model.uw_max+model.w+sum_delta_w>=0


            # self.model.vmax = Constraint(self.model.ind,rule=lambda model,k:model.uv_max-model.v-sum(model.u_delta_v[0:k])
            #     if k<self.m else Constraint.Skip)
            # self.model.vmin = Constraint(self.model.ind,rule=lambda model,k:+model.uv_max+model.v+sum(model.u_delta_v[0:k])
            #     if k<self.m else Constraint.Skip)
            # self.model.wmax = Constraint(self.model.ind,rule=lambda model,k:model.uw_max-self.model.w-sum(model.u_delta_w[0:k])
            #     if k<self.m else Constraint.Skip)
            # self.model.wmin = Constraint(self.model.ind,rule=lambda model,k:+model.uw_max+self.model.w+sum(model.u_delta_w[0:k])
            #     if k<self.m else Constraint.Skip)

            # self.model.vx_update=Constraint(self.model.ind,rule=lambda model,k:model.u_v[k]==model.v+sum(model.u_delta_v[i] if i<k for i in model.ind) if k<self.m else Constraint.Skip)
            # self.model.vw_update=Constraint(self.model.ind,rule=lambda model,k:model.u_w[k]==model.w+sum(model.u_delta_w[0:k])
            #     if k<self.m else Constraint.Skip)

            # self.model.obstacles=Constraint(self.model.ind,rule=lambda model,k:+model.uw_max+model.w+sum(model.u_delta_w[0:k])
            #     if k<self.m else Constraint.Skip)

            self.model.vmaxConstraint=Constraint(self.model.ind_m,rule=vmaxConstraint)
            self.model.vminConstraint=Constraint(self.model.ind_m,rule=vminConstraint)
            self.model.wmaxConstraint=Constraint(self.model.ind_m,rule=wmaxConstraint)
            self.model.wminConstraint=Constraint(self.model.ind_m,rule=wminConstraint)
            # 这里只考虑到m的


            
            def ObjRule(model):
                uqu=0
                pre_v=model.v
                pre_w=model.w

                pre_theta=model.theta
                pre_x=model.x 
                pre_y=model.y
                for i in model.ind_p:
                    if i<self.m:
                        pre_v+=model.u_delta_v[i]
                        pre_w+=model.u_delta_w[i]
                        uqu+=model.H*(model.u_delta_v[i]**2+model.u_delta_w[i]**2)
                        
                        if i<self.m-1 and self.notFisrtPlan:
                            uqu+=model.S*((model.u_delta_v[i]-self.v_opt[i+1])**2+(model.u_delta_w[i]-self.w_opt[i+1])**2)
                    pre_theta+=pre_w*self.controlDt
                    pre_x+=cos(pre_theta)*pre_v*self.controlDt
                    pre_y+=sin(pre_theta)*pre_v*self.controlDt

                    uqu+=model.Q*((pre_x-model.R_qp_x[i])**2+(pre_y-model.R_qp_y[i])**2)
                    for j in range(15):
                        if self.selectedObs[j]:
                            uqu-=model.R*((pre_x-self.obstacles[j,0])**2+(pre_y-self.obstacles[j,1])**2)

                return uqu
            # Objective
            # self.model.xQx = self.model.Q[1]*sum((self.model.z[0,i]+self.model.x-self.model.R_qp_x[i])**2 for i in self.model.ind)
            # self.model.yQy = self.model.Q[2]*sum((self.model.z[1,i]-path[i-1][1])**2 for i in self.model.zk_number)


            # self.model.vRv = self.model.R[1]*sum((self.model.vmax-self.model.v[i])**2 for i in self.model.uk_obj)
            # self.model.wRw = self.model.R[2]*sum(self.model.w[i]**2 for i in self.model.uk_obj)

            # self.model.dvSdv = self.model.S[1]*sum((self.model.v[i+1]-self.model.v[i])**2 for i in self.model.uk_obj)
            # self.model.dwSdw = self.model.S[2]*sum((self.model.w[i+1]-self.model.w[i])**2 for i in self.model.uk_obj)

            self.model.obj = Objective(expr=ObjRule,sense=minimize)
            

            #Solve
            self.optimizer.solve(self.model)

            # x_opt = [self.model.z[0,k]() for k in self.model.zk_number]
            # y_opt = [self.model.z[1,k]() for k in self.model.zk_number]
            # θ_opt = [self.model.z[2,k]() for k in self.model.zk_number]
            self.v_opt=[self.model.u_delta_v[k]() for k in self.model.ind_m]
            self.w_opt=[self.model.u_delta_w[k]() for k in self.model.ind_m]
            print("--------v,w------------",self.v_opt,self.w_opt)
            print(self.v,self.w)
            self.notFisrtPlan=True
            # self.Control(self.model.u_delta_v[0],self.model.u_delta_v[0])

        else:
            pass
    def __reRefPath(self):
        # print(max(0,-4))
        print(np.abs(self.xpos[max(self.curSteps-self.p,0):self.curSteps+self.p]-self.x)+np.abs(self.ypos[max(self.curSteps-self.p,0):self.curSteps+self.p]-self.y))
        nowSteps=np.argmin(np.abs(self.xpos[max(self.curSteps-self.p,0):self.curSteps+self.p]-self.x)+np.abs(self.ypos[max(self.curSteps-self.p,0):self.curSteps+self.p]-self.y))+max(self.curSteps-self.p,0)
        print("-----now step-----",nowSteps)

        self.curSteps=nowSteps+1
        # print(self.xpos[self.curSteps:self.curSteps+self.p].shape)
        def R_qp_x_init(model, i):
            return self.xpos[self.curSteps+i]
        def R_qp_y_init(model, i):
            return self.ypos[self.curSteps+i]
        # self.model.R_qp_x=Param(self.model.ind_p,initialize =set(self.xpos[self.curSteps:self.curSteps+self.p].tolist()))
        # self.model.R_qp_y=Param(self.model.ind_p,initialize =set(self.ypos[self.curSteps:self.curSteps+self.p].tolist()))
        self.model.R_qp_x=Param(self.model.ind_p,initialize =R_qp_x_init)
        self.model.R_qp_y=Param(self.model.ind_p,initialize =R_qp_y_init)
        # print(self.model.R_qp_x)
        # print(set(self.ypos[self.curSteps:self.curSteps+self.p].tolist()))
        # print([self.model.R_qp_x[v] for v in self.model.ind_p])
        self.R_qp_x[:]=self.xpos[self.curSteps:self.curSteps+self.p]
        self.R_qp_y[:]=self.ypos[self.curSteps:self.curSteps+self.p]
        
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
        self.steps=self.Length//self.pace*10

        # self.xpos=np.interp.int

        #平滑处理后
        self.curSteps=0
        steps=np.zeros(int(self.steps)+self.p)
        # steps[:self.p]=np.linspace(0.1, 5, self.p)
        # self.curSteps=self.p+10
        steps[:-self.p] = np.linspace(0.1, self.steps-0.01, int(self.steps))  # np.linspace 等差数列,从x.min()到x.max()生成300个数，便于后续插值
        print('steps\n',steps)
        # print(make_interp_spline(t, path_x))
        # self.pos = make_interp_spline(t, path)(steps)
        self.xpos = make_interp_spline(t, path_x)(steps[:-self.p])
        self.ypos = make_interp_spline(t, path_y)(steps[:-self.p])
        self.xpos[-self.p:]=path_x[-1]
        self.ypos[-self.p:]=path_y[-1]
        

        pass 
    def setAction(self,action):
        self.action=action
    def setDebugger(self,debugger):
        self.debugger=debugger
    def Control(self,package):
        if self.curSteps+self.p<self.steps:
            self.action.sendCommand(vx=self.v_opt[0]*self.predictDt+self.v, vy=0, vw=self.w_opt[0]*self.predictDt+self.w)
            self.debugger.draw_points_numpy(package,self.R_qp_x[:],self.R_qp_y[:])#追踪的midpos（白色）

            self.pre_w[0]=self.w+self.w_opt[0]
            for i in range(1,self.m):
                self.pre_w[i]=self.w_opt[i]+self.pre_w[i-1]
            self.pre_theta[0]=self.theta+self.w_opt[0]*self.controlDt
            for i in range(1,self.p):
                self.pre_theta[i]=self.pre_theta[i-1]+self.pre_w[i]*self.controlDt
            self.pre_v[0]=self.v+self.v_opt[0]
            for i in range(1,self.m):
                self.pre_v[i]=self.v_opt[i]+self.pre_v[i-1]
            self.pre_x[0]=self.x+cos(self.pre_theta[0])*self.pre_v[0]*self.controlDt
            for i in range(1,self.p):
                self.pre_x[i]=cos(self.pre_theta[i])*self.pre_v[i]*self.controlDt+self.pre_x[i-1]
            self.pre_y[0]=self.y+sin(self.pre_theta[0])*self.pre_v[0]*self.controlDt
            for i in range(1,self.p):
                self.pre_y[i]=sin(self.pre_theta[i])*self.pre_v[i]*self.controlDt+self.pre_y[i-1]

            print("-------pre x-----------",self.pre_x)
            print("-------pre y-----------",self.pre_y)
            self.debugger.draw_points_numpy(package,self.pre_x[:],self.pre_y[:],color='r')#规划处的vw在predict_time后到达的位置（红色）
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
    # predictor.setAction()
    predictor.reCurrentState(0,0,0,0,0)
    predictor.RefreshPath(np.array([100,200,300,600,800,400]),np.array([100,300,400,200,400,900]))
    predictor.rePredict()
