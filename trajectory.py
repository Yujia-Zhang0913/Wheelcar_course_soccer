import numpy as np 
import vision
from scipy.interpolate import CubicSpline
from scipy.spatial.distance import euclidean

class Controller():
    def __init__(self,path,robot:vision.Robot,k_1=100,k_2=100,leadTime=0.1,leadLength=30) -> None:
        # path 为[n,2]的numpy assay
        self.path=path
        self.path2viapoints()
        self.nextViaPoints=30
        # self.goal=goal
        self.accepted_distance=10
        self.leadTime=leadTime
        self.leadLength=leadLength
        self.leadPace=15000

        self.v_max=3500 #m/s
        self.k_1=k_1
        self.k_2=k_2
        self.mu=100
        self.lam=2

        self.rou=0
        self.alpha=0
        self.beta=0

        self.ControlMethod=self.oneSample
        self.robot=robot
        pass
    def path2viapoints(self):
        self.Length=np.sum(np.abs(self.path[1:,0]-self.path[:-1,0]))+np.sum(np.abs(self.path[1:,1]-self.path[:-1,1]))
        # print(self.path.shape)
        self.cs_x=CubicSpline(np.linspace(0,self.Length,self.path.shape[0]),self.path[:,0])
        self.cs_y=CubicSpline(np.linspace(0,self.Length,self.path.shape[0]),self.path[:,1])
        pass 
    def refresh(self,path=None):
        if(path!=None):
            self.path=path
            self.path2viapoints()
            self.nextViaPoints=30
        
        # self.currenctPos=pos 
        self.selectViaPoint()
        self.nextViaPoints+=self.leadPace
        return self.ControlMethod()
        pass

    def selectViaPoint(self):
        self.x_goal=self.cs_x(self.nextViaPoints/self.Length*self.path.shape[0])
        self.y_goal=self.cs_y(self.nextViaPoints/self.Length*self.path.shape[0])

        self.theta_goal=np.arctan2(self.cs_x((self.nextViaPoints+self.leadLength)/self.Length)-self.x_goal,self.cs_x((self.nextViaPoints+self.leadLength)/self.Length)-self.x_goal)
        
        # 最近的点（self.leadLength决定
        self.rou=np.sqrt((self.x_goal-self.robot.x)**2+(self.y_goal-self.robot.y)**2)
        self.beta=np.arctan2((self.x_goal-self.robot.x),(self.y_goal-self.robot.y))
        self.alpha=self.beta-self.robot.orientation
        pass 
    def controlRule(self):
        k=lambda rou,alpha,beta:1/rou*(self.k_2*(alpha-np.arctan(-self.k_1*beta))+(1+self.k_1/(1+(self.k_1*beta)**2))*np.sin(alpha))
        k_=k(rou=self.rou,alpha=self.alpha,beta=self.beta)
        v=self.v_max/(1+self.mu*abs(k_)**self.lam)

        w=v*k_
        if w>15:
            w=15
        elif w<15:
            w=-15
        w=w/180*np.pi 

        print("k",k_)
        return (v,w)
    def oneSample(self):
        return self.controlRule()
        pass 
if __name__=="__main__":
    path=np.array([[-621,2227],[-1584,2807],[-4207,1438],[-4207,858],[-2107,-2250]])
    myRobot=vision.Robot(id=0)
    myRobot.x=0
    myRobot.y=0
    controller=Controller(path=path,robot=myRobot)
    # while True:
    print("v,w",controller.refresh())
    print("x,y",myRobot.x,myRobot.y,myRobot.orientation)
    print("desired x,y",controller.x_goal,controller.y_goal,controller.theta_goal)
    print("para",controller.rou,controller.alpha,controller.beta)
