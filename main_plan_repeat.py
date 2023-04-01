from vision import Vision
from action import Action
from debug import Debugger
from trajectory import Controller
from zss_debug_pb2 import Debug_Msgs
import time
from A_star import A_star
from RRT_plan import RRT
from RRT_rewire import RRT_
# from dwaplanner import DWA
# from dwaplanner import Config
from dwaplanner_txs import DWA
from dwaplanner_txs import Config
import numpy as np
from scipy.signal import savgol_filter


def B_smooth(x,y,k):
	 # n表示曲线的点数减1，因为下标从0开始；也表示B样条的基函数要计算到n
	n = len(x) - 1
	# u_base表示准均匀集合的分母
	u_base = n + 1 - k
	if u_base==0:
		return x,y
	# U表示准均匀集合，重复k阶个起始点和结束点
	U = []
	for i in range(k):
		U.append(0)
	u_basic = 1.0/u_base
	for i in range(u_base+1):
		U.append(i * u_basic)
	for i in  range(k):
		U.append(1)
	x_smooth=[]
	y_smooth=[]
	def B_spline(i, u, k, U):
		Nik_u = 0.0
		if k==0:
			if u >= U[i] and u < U[i+1]:
				Nik_u = 1.0
		else:
			base1 = U[i+k] - U[i]
			base2 = U[i+k+1] - U[i+1]
			if base1 == 0.0:
				base1 = 1.0
				para1=(u - U[i]) / base1
				if u - U[i]==0:
					para1=0
			else:
				para1=(u - U[i]) / base1
			if base2 == 0.0:
				base2 = 1.0
				para2=(U[i+k+1] - u) / base1
				if u - U[i]==0:
					para2=0
			else:
				para2=(U[i+k+1] - u) / base2
			Nik_u = para1 * B_spline(i, u, k-1, U) + para2 * B_spline(i+1, u, k-1, U)
		return Nik_u
	u=0.0
	while u <= 1.0:
		tmp_x=0
		tmp_y=0
		# 累加各个原始点和样条基函数的乘积
		for i in range(len(x)):
			Nik_u = B_spline(i, u, k, U)
			tmp_x += x[i] * Nik_u
			tmp_y += y[i] * Nik_u
		# 计算完一组，就放入拟合曲线的集合中
		x_smooth.append(tmp_x)
		y_smooth.append(tmp_y)
		u += 0.02
	return x_smooth,y_smooth

# 想要使用反馈控制，就把所有带control的取消注释
if __name__ == '__main__':
	# time.sleep(10)
	vision = Vision()
	time.sleep(0.1)
	action = Action()
	debugger = Debugger()
	# blue_robot_xy=[np.array([vision.blue_robot[i].x,vision.blue_robot[i].y]) for i in range(1,15) if vision.blue_robot[i].x!=-999999 and vision.blue_robot[i].y!=-999999]
	# yellow_robot_xy=[np.array([vision.yellow_robot[i].x,vision.yellow_robot[i].y]) for i in range(0,15) if vision.yellow_robot[i].x!=-999999 and vision.yellow_robot[i].y!=-999999]
	# obstacles=np.array(blue_robot_xy+yellow_robot_xy)
	# print(obstacles)
	myRobot=vision.my_robot
	# my_robot_xy=np.array([myRobot.x,myRobot.y])
	# a_star=A_star(obstacles, my_robot_xy, [-2400, -1500], -4950,-3694,4950,3696)
	# rrt=RRT(obstacles, my_robot_xy, [-2400, -1500], -4950,-3694,4950,3696)
	# rrt_=RRT_(obstacles, my_robot_xy, [-2400, -1500], -4950,-3694,4950,3696)
	# 想要使用反馈控制，注释下两行
	# best_path_X,best_path_Y=rrt_.Process()
	# path=np.transpose([best_path_X,best_path_Y])
	# 想要使用反馈控制，取消下一行注释
	# controller=Controller(path=path,robot=myRobot)
	i=1
	flag=1
	while True:
		blue_robot_xy=[np.array([vision.blue_robot[i].x,vision.blue_robot[i].y]) for i in range(1,15) if vision.blue_robot[i].x!=-999999 and vision.blue_robot[i].y!=-999999]
		yellow_robot_xy=[np.array([vision.yellow_robot[i].x,vision.yellow_robot[i].y]) for i in range(0,15) if vision.yellow_robot[i].x!=-999999 and vision.yellow_robot[i].y!=-999999]
		obstacles=np.array(blue_robot_xy+yellow_robot_xy)
		my_robot_xy=np.array([myRobot.x,myRobot.y])
		# print(obstacles)
		# 1. path planning & velocity planning
		# Do something
		# 想要使用反馈控制，下面三个选一个取消注释
		if flag==1:#若到达终点则进行重规划
			if i%2==1:
				start=[2400,1500]
				target=[-2400,-1500]
			else:
				start=[-2400,-1500]
				target=[2400,1500]
			flag=0
		rrt_=RRT_(obstacles, my_robot_xy, target, -4950,-3694,4950,3696)
		# best_path_X,best_path_Y=a_star.Process()
		# best_path_X,best_path_Y=rrt.Process()
		best_path_X,best_path_Y=rrt_.Process()
		##B样条
		best_path_X_filtered,best_path_Y_filtered=B_smooth(best_path_X,best_path_Y,4)
		best_path_X=np.append(best_path_X,target[0])#最后加上终点的精确坐标，以防差一点到不了终点
		best_path_Y=np.append(best_path_Y,target[1])
		# 想要使用反馈控制，取消下三行注释
		# path=np.transpose([best_path_X,best_path_Y])
		# controller=Controller(path=path,robot=myRobot)
		# print(best_path_X,best_path_Y)

		# 2. send command
		# 想要使用反馈控制，取消下三行注释
		# v,w=controller.refresh()
		# action.sendCommand(vx=v, vy=0, vw=w)
		# print(v,w,myRobot.x,myRobot.y,myRobot.orientation)

		# 想要使用反馈控制，注释下10行
		dwa=DWA()
		dwaconfig=Config(100)
		robot_info=[myRobot.x,myRobot.y,myRobot.orientation,myRobot.vx,myRobot.vw]
		dis_temp=np.hypot([best_path_X[i]-myRobot.x for i in range(len(best_path_X))],[best_path_Y[i]-myRobot.y for i in range(len(best_path_Y))]).tolist()
		# dis_index=dis_temp.index(min(dis_temp))
		dis_index=np.argmin(dis_temp)
		
		index_delta=25
		if dis_index+index_delta+1>len(best_path_Y):
			dis_index=len(best_path_Y)-index_delta-1
		midpos=[best_path_X[dis_index+index_delta],best_path_Y[dis_index+index_delta]]
		v,w,position_predict=dwa.plan(robot_info, dwaconfig, midpos, obstacles)
		action.sendCommand(vx=v, vy=0, vw=w)
		myRobot.vx=v #更新信息
		myRobot.vw=w
		#判断是否到达终点
		# dist_to_target=np.linalg.norm(my_robot_xy-np.array(target))
		if np.linalg.norm(my_robot_xy-np.array(target))<80:
			flag=1 #标记到达终点
			i=i+1 #记录转向
		# print(v,w,myRobot.x,myRobot.y,myRobot.orientation)
		# 3. draw debug msg
		package = Debug_Msgs()
		# debugger.draw_point(package, controller.x_goal,controller.y_goal)
		# debugger.draw_circle(package, myRobot.x, myRobot.y)
		debugger.draw_lines(package, x1=best_path_X[0:len(best_path_X)-2], y1=best_path_Y[0:len(best_path_X)-2], x2=best_path_X[1:len(best_path_X)-1], y2=best_path_Y[1:len(best_path_X)-1])
		debugger.draw_point(package,midpos[0],midpos[1])#追踪的midpos（白色）
		debugger.draw_point(package,position_predict[0],position_predict[1],color='r')#规划处的vw在predict_time后到达的位置（红色）
		debugger.draw_point(package,start[0],start[1],color='b') #起点(蓝色)
		debugger.draw_point(package,target[0],target[1],color='g') #终点（绿色）
		debugger.send(package)

		time.sleep(0.02)
