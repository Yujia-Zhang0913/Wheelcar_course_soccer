from vision import Vision
from action import Action
from debug import Debugger
from trajectory import Controller
from zss_debug_pb2 import Debug_Msgs
import time
from A_star import A_star
from RRT_plan import RRT
from RRT_rewire import RRT_
from dwaplanner import DWA
from dwaplanner import Config
import numpy as np

if __name__ == '__main__':
	# time.sleep(10)
	vision = Vision()
	time.sleep(0.1)
	action = Action()
	debugger = Debugger()
	blue_robot_xy=[np.array([vision.blue_robot[i].x,vision.blue_robot[i].y]) for i in range(1,15) if vision.blue_robot[i].x!=-999999 and vision.blue_robot[i].y!=-999999]
	yellow_robot_xy=[np.array([vision.yellow_robot[i].x,vision.yellow_robot[i].y]) for i in range(0,15) if vision.yellow_robot[i].x!=-999999 and vision.yellow_robot[i].y!=-999999]
	obstacles=np.array(blue_robot_xy+yellow_robot_xy)
	# print(obstacles)
	myRobot=vision.my_robot
	my_robot_xy=np.array([myRobot.x,myRobot.y])
	a_star=A_star(obstacles, my_robot_xy, [-2400, -1500], -4950,-3694,4950,3696)
	rrt=RRT(obstacles, my_robot_xy, [-2400, -1500], -4950,-3694,4950,3696)
	rrt_=RRT_(obstacles, my_robot_xy, [-2400, -1500], -4950,-3694,4950,3696)
	best_path_X,best_path_Y=rrt_.Process()
	path=np.transpose([best_path_X,best_path_Y])
	# controller=Controller(path=path,robot=myRobot)
	while True:
		# 1. path planning & velocity planning
		# Do something
		# best_path_X,best_path_Y=a_star.Process()
		# best_path_X,best_path_Y=rrt.Process()
		# best_path_X,best_path_Y=rrt_.Process()
		# path=np.transpose([best_path_X,best_path_Y])
		# controller=Controller(path=path,robot=myRobot)
		# print(best_path_X,best_path_Y)
		# 2. send command

		# v,w=controller.refresh()
		# action.sendCommand(vx=v, vy=0, vw=w)
		# print(v,w,myRobot.x,myRobot.y,myRobot.orientation)

		dwa=DWA()
		dwaconfig=Config(22)
		robot_info=[myRobot.x,myRobot.y,myRobot.orientation,0,0]
		dis_temp=np.hypot([best_path_X[i]-myRobot.x for i in range(len(best_path_X))],[best_path_Y[i]-myRobot.y for i in range(len(best_path_Y))]).tolist()
		dis_index=dis_temp.index(min(dis_temp))
		if dis_index+2>len(best_path_Y):
			dis_index=len(best_path_Y)-2
		[v,w]=dwa.plan(robot_info, dwaconfig, [best_path_X[dis_index+2],best_path_Y[dis_index+2]], obstacles)
		action.sendCommand(vx=v, vy=0, vw=w)
		print(v,w,myRobot.x,myRobot.y,myRobot.orientation)

		# 3. draw debug msg
		package = Debug_Msgs()
		# debugger.draw_point(package, controller.x_goal,controller.y_goal)
		debugger.draw_circle(package, myRobot.x, myRobot.y)
		debugger.draw_lines(package, x1=best_path_X[0:len(best_path_X)-2], y1=best_path_Y[0:len(best_path_X)-2], x2=best_path_X[1:len(best_path_X)-1], y2=best_path_Y[1:len(best_path_X)-1])
		debugger.send(package)

		time.sleep(0.02)