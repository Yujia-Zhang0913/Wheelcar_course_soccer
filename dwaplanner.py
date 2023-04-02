from calendar import c
from re import S
from math import floor
import numpy as np
import math

class Config:
    robot_radius = 100
    def __init__(self,obs_radius,predict_time=1,predict_time_collision=2.5,gain_heading=0.6,gain_avoid_collision=0.25,gain_velocity=0.15,w_max=10,dt=0.1):
        self.obs_radius = obs_radius
        self.dt = dt  # [s] 仿真时间步长

        self.max_speed = 3000  # [mm/s]
        self.min_speed = -3000  # [mm/s]
        self.max_accel = 4000  # [mm/ss]
        self.v_reso = self.max_accel*self.dt/10  # [m/s]  #可行速度为+-10个reso
        # print(self.v_reso)

        self.max_yawrate = w_max  # [rad/s]
        self.max_dyawrate = 20 # [rad/ss]
        self.yawrate_reso = self.max_dyawrate*self.dt/10.0  # [rad/s]

        self.predict_time = predict_time  # [s]
        self.predict_time_collision = predict_time_collision  # [s] #simple

        self.gain_heading = gain_heading
        self.gain_avoid_collision = gain_avoid_collision
        self.gain_velocity = gain_velocity

        # self.tracking_dist = self.predict_time*self.max_speed*0.7
        # self.arrive_dist = 0.1

class DWA:
    def __init__(self):
        pass
    
    def plan(self,state,dwaconfig,midpos,obs,obs_vel):
        # state: 数组 0:x 1:y 2:theta 3:vx 4:vw
        # dwaconfig: DWA的参数配置
        # midpos: 跟踪点坐标
        # obs: 障碍物坐标集合，np.array (n,2)
        # obs_vel:障碍物速度集合，np.array(n,2)
        # 注意：plan函数在num_vw*num_vx的速度空间中选择最佳速度，因此变量大都为shape==(num_vw,num_vx)的np array，并大量使用广播、切片等操作，尽量避免for循环以加速计算
        num_vw=21
        num_vx=21
        self.heading=np.zeros([num_vw,num_vx])
        self.avoid_collision=np.zeros([num_vw,num_vx])
        self.velocity=np.zeros([num_vw,num_vx])

       # 1、计算heading：
        #21*21方阵
        self.vx=np.zeros([num_vw,num_vx])
        self.vw=np.zeros([num_vw,num_vx])
        self.r=np.zeros([num_vw,num_vx])
        self.x_pr=np.zeros([num_vw,num_vx])
        self.y_pr=np.zeros([num_vw,num_vx])
        self.dist_left=np.zeros([num_vw,num_vx])
        self.dist_total=np.hypot(midpos[0]-state[0],midpos[1]-state[1])  #当前位置距离目标点的总距离

        for vw_point in range(num_vw):
            for vx_point in range(num_vx):
                self.vx[vw_point][vx_point]=state[3]-10*dwaconfig.v_reso+vx_point*dwaconfig.v_reso
                self.vw[vw_point][vx_point]=state[4]-10*dwaconfig.yawrate_reso+vw_point*dwaconfig.yawrate_reso
                if(self.vw[vw_point][vx_point]==0):
                    self.vw[vw_point][vx_point]=1 #由于vw要做分母，不能为0
        #print(dwaconfig.v_reso)
        #print(self.vx)
        self.r=self.vx/self.vw
        self.x_pr=state[0]-self.r*np.sin(state[2])+self.r*np.sin(state[2]+self.vw*dwaconfig.predict_time)
        self.y_pr=state[1]+self.r*np.cos(state[2])-self.r*np.cos(state[2]+self.vw*dwaconfig.predict_time)
        self.dist_left=np.hypot(self.x_pr-midpos[0] ,self.y_pr-midpos[1])  #预期点距离目标点的剩余距离
        self.heading=1-self.dist_left/self.dist_total


        # 2、计算avoid_collision：predict_time_collision内路径上到障碍物的最近距离
        num_time=10
        x_pr_time=np.zeros([num_vw,num_vx,num_time])
        y_pr_time=np.zeros([num_vw,num_vx,num_time])

        #时间序列，作为x_pr_time和y_pr_time的第三维
        for time_point in range(num_time):
            x_pr_time[:,:,time_point]=state[0]-self.r*np.sin(state[2])+self.r*np.sin(state[2]+self.vw*dwaconfig.predict_time_collision*(time_point+1)/num_time)
            y_pr_time[:,:,time_point]=state[1]+self.r*np.cos(state[2])-self.r*np.cos(state[2]+self.vw*dwaconfig.predict_time_collision*(time_point+1)/num_time)
            # obsx_pr_time[:,:,time_point]=
        x_pr_time=x_pr_time.reshape(-1)
        y_pr_time=y_pr_time.reshape(-1)#一维化
            
        min_dist_list=np.zeros([num_vw*num_vx*num_time,])
        #print(min_dist_list.shape)
        #位置循环
        for pos in range(num_vw*num_vx*num_time):
            time_point=pos%num_time #由于在一维列表里循环，需要取余计算时间点time
            obs_pr=obs+obs_vel*dwaconfig.predict_time_collision*(time_point+1)/num_time #按直线运动，预测time_point时的障碍物位置
            dist_list=np.hypot(obs_pr[:,0]-x_pr_time[pos],obs_pr[:,1]-y_pr_time[pos]) #按障碍物列表
            min_dist_list[pos]=dist_list.min() #到达的点离所有障碍物最近的距离
        min_dist_list=min_dist_list.reshape(num_vw,num_vx,num_time)
        for i in range(num_vw):
            for j in range(num_vx):
                self.avoid_collision[i][j]=min_dist_list[i,j,:].min()#在时间序列中取到障碍物的最近距离，作为v，w点的避障得分
                if(self.avoid_collision[i][j]<dwaconfig.robot_radius+dwaconfig.obs_radius):
                    self.avoid_collision[i][j]=-10000
        self.avoid_collision=self.avoid_collision/np.max(self.avoid_collision)#归一化
        #print(self.avoid_collision)

        # 3、计算velocity
        self.velocity=abs(self.vx)/dwaconfig.max_speed #- 0.1*abs(self.vw)/dwaconfig.max_yawrate
        #print(self.velocity)

        # 4、加权求和，计算evaluation
        self.evaluation = dwaconfig.gain_heading*self.heading + dwaconfig.gain_avoid_collision*self.avoid_collision + dwaconfig.gain_velocity*self.velocity  #simple mode

        #超过速度和角速度范围的值赋低分
        for i in range(num_vw):
            for j in range(num_vx):
                if not(dwaconfig.min_speed<self.vx[i][j]<dwaconfig.max_speed and -dwaconfig.max_yawrate<self.vw[i][j]<dwaconfig.max_yawrate):
                    self.evaluation[i][j]=-10000
        #print(self.evaluation)
        

        #返回得分最高的(v,w)
        index=np.argmax(self.evaluation)
        index_vw=floor(index/num_vx)
        index_vx=index%num_vx
        vw_return=state[4]-10*dwaconfig.yawrate_reso+index_vw*dwaconfig.yawrate_reso
        vx_return=state[3]-10*dwaconfig.v_reso+index_vx*dwaconfig.v_reso
        position_predict=[self.x_pr[index_vw,index_vx],self.y_pr[index_vw,index_vx]]
        # print("index:",index)
        # print("index_vx:",index_vx,"index_vw",index_vw)
        print("vx_return:",vx_return,"vw_return",vw_return)
        # print("heading:",self.heading[index_vw,index_vx],"avoid_collision:",self.avoid_collision[index_vw,index_vx],"velocity:",self.velocity[index_vw,index_vx])

        return vx_return,vw_return,position_predict



# dwa=DWA()
# config=Config(1)
# state=[0.1,1,0,1,0]
# midpos=(6,1)
# obs=np.array([[3,1],])
# dwa.plan(state,config,midpos,obs)