#!/usr/bin/env python
from __future__ import print_function

import rospy
import cv2 
#import imutils
import threading
from Tkinter import *
import PIL
from PIL import ImageTk
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg  import Point
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
import math
import sys
import time
import matplotlib.pyplot as plt

import imutils
from numpy import asarray
from numpy import savetxt



global da,label,root,imgtk,contador
da=0
global root
contador=0
global yaw_z, x_pr,y_pr,x_w,y_w,x_robot,y_robot, x_hist, y_hist,theta_hist,per_hist,x_w2,y_w2,per_hist2
per_hist2 = []
per_hist = []
x_hist = []
y_hist = []
theta_hist = []
yaw_z = 0
x_pr =0
y_pr = 0
x_w = np.array([0]) #coordenada x estimada de la persona en el marco global
y_w = np.array([0]) #coordenada y estimada de la persona en el marco global
x_w2 = 0
y_w2 = 0
x_robot = 0
y_robot = 0
sys.setrecursionlimit(3000)


#root = Tk()
def callback(data):
	#rospy.loginfo(rospy.get_caller_id()+"I_heard%s", data.data)
	global da#,imgtk
	da = data
	
def callbackDos(dat):
	#rospy.loginfo(rospy.get_caller_id()+"I_heard%s", data.x)
	print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
	print('Posicion de la persona',dat)
	global x_pr, y_pr
	x_pr = dat.x
	y_pr = dat.y


def euler_from_quaternion(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.degrees(math.atan2(t0, t1))
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.degrees(math.asin(t2))
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        global yaw_z
        yaw_z = math.degrees(math.atan2(t3, t4))
        return roll_x, pitch_y, yaw_z # in radians

def callback_odometry(msg):

	global x_robot,y_robot

	print('Posicion del Robot')
	print(msg.pose.pose.position)
	x_robot = msg.pose.pose.position.x
	y_robot = msg.pose.pose.position.y
	x = msg.pose.pose.orientation.x
	y = msg.pose.pose.orientation.y
	z = msg.pose.pose.orientation.z
	w = msg.pose.pose.orientation.w
	print(euler_from_quaternion(x,y,z,w))




def rotar(x,y,theta):
	theta = np.deg2rad(theta)
	r = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
	new_vector = np.matmul(r,np.array([[x],[y]]))
	x1 = new_vector[0]
	y1 = new_vector[1]
	return x1,y1

def pixel_to_world(p_x,p_y):
	with open('/home/robotica/PG_ws/src/proyecto_de_grado/scripts/matrix4.txt', 'r') as f:
	    input = np.array([[np.float(num) for num in line.split(',')] for line in f])
	def closest_node(node, nodes):
	    nodes = np.asarray(nodes)
	    dist_2 = np.sum((nodes - node)**2, axis=1)
	    return np.argmin(dist_2)
	puntos = input[:,0:2]
	punto = np.array([p_x,p_y])

	indice = closest_node(punto,puntos)
	vector = input[indice,2::]
	x = vector[0]
	y = vector[1]
	global yaw_z, x_robot, y_robot
	theta = yaw_z-45
	print('angulo real', theta)
	print('x_before ',x,'y_before ',y,)
	x_w, y_w = rotar(x,y,theta)
	print('x_before ',x_w,'y_before ',y_w,)
	x_world = x_w+x_robot
	y_world = y_w + y_robot


	return x_world, y_world

def init_ima():
	global root,label, contador
	
	print(contador)
	root = Tk()
	root.title('Human Detector 2022')
	root.geometry("700x550")
	root.resizable(width=False,height=False)

	# IMAGEN 
	LF_image = LabelFrame(root,text = "",labelanchor = "n",height=120, width=450)
	LF_image.pack(fill = "both")
	#PI_object = PhotoImage(file = "/german_flag.jpeg")
	imgg = ImageTk.PhotoImage(PIL.Image.open("/home/robotica/PG_ws/src/proyecto_de_grado/scripts/computer_vision.jpg").resize((695,120), PIL.Image.ANTIALIAS))
	L_image = Label(LF_image,image = imgg)
	L_image.pack()

	LF_display_info = Frame(root,height = 510,width = 360)
	LF_display_info.pack(fill = "both", expand = "yes")
	LF_image_left = LabelFrame(LF_display_info,text = "Robot Camera",labelanchor = "n",height=240, width=240)
	LF_image_left.pack(fill = "both", expand = "yes",side = LEFT)

	label=Label(LF_image_left,text='primeraVEz')
	label.pack() 

	LF_image_right = Frame(LF_display_info,height=250, width=200)
	LF_image_right.pack(fill = "both", expand = "yes",side = LEFT)
	LF_image_up = LabelFrame(LF_image_right,text = "Detection",labelanchor = "n",height=125, width=200)
	LF_image_up.pack(fill = "both", expand = "yes")

	LF_options = LabelFrame(root,text = "Options",labelanchor = "n",height = 40,width = 450)
	LF_options.pack(fill = "both", expand = "yes")
	B_options_start = Button(LF_options,text = "Start",width = 5)
	B_options_start.pack(fill = "x",expand = "yes",side = LEFT)
	B_options_stop = Button(LF_options, text = "Stop",width = 5)
	B_options_stop.pack(fill = "x", expand = "yes", side = LEFT)

	def detect(frame):
	    bounding_box_cordinates, weights =  HOGCV.detectMultiScale(frame, winStride = (4, 4), padding = (8, 8), scale = 1.03)
	    
	    person = 1
	    for x,y,w,h in bounding_box_cordinates:
	        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
	        #cv2.putText(frame, f'person{person}', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
	        #person += 1
	    
	    cv2.putText(frame, 'Status : Detecting ', (40,40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
	    #cv2.putText(frame, f'Total Persons : {person-1}', (40,70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
	    #cv2.imshow('output', frame)

	    return frame

	def update_image():
		global da,label,im, contador
		bridge = CvBridge()
		try:
			cv_image = bridge.imgmsg_to_cv2(da, "bgr8")#bgr8
			#cv_image = cv2.blur(cv_image,(3,3))
			#------------------------------------------------------------------------------------------------------
			#cam = cv2.VideoCapture(0)
			#ret, prev = cam.read()

			#------------------------------------------------------------------------------------------------------
			cv2.imwrite(r'/home/robotica/PG_ws/src/proyecto_de_grado/images/image'+str(contador)+'.jpg', cv_image)
			#cv2.imwrite('../images/'+'image'+str(contador)+'.jpg', cv_image)
			#tic = time.time()
			bounding_box_cordinates, weights =  HOGCV.detectMultiScale(cv_image, winStride = (4, 4), padding = (8, 8), scale = 1.09)
			puntos_p = []
			for x,y,w,h in bounding_box_cordinates:
				
				print('equis',x)
				new_x = np.int32(x+40)
				new_y = np.int32(y+40)
				new_w = np.int32(w-30)
				new_h = np.int32(h-30)
				print('new equis',new_x)
				x_circ = np.int32(new_x+np.int32((np.int32(x+w-40)-new_x)/2))
				y_circ = np.int32(y+h-40)
				#puntos.append(np.array([x_circ,y_circ]))
				cv2.rectangle(cv_image, (new_x,new_y), (np.int32(x+w-40),np.int32(y+h-40)), (0,255,0), 2)
				#toc = time.time()-tic
				#print('tiempo de deteccion',toc)
				cv2.circle(cv_image,(x_circ,y_circ), 10, (0,0,255), -1)
				print(x_circ,y_circ)
				font = cv2.FONT_HERSHEY_SIMPLEX
				
				#-----------------------------------------------------------------------------------------------
				'''
				print('equis',x)
				new_x = np.int32(x+40)
				new_y = np.int32(y+40)
				new_w = np.int32(w-30)
				new_h = np.int32(h-30)
				print('new equis',new_x)
				x_circ = np.int32(new_x+np.int32((np.int32(x+w-40)-new_x)/2))
				y_circ = np.int32(y+h-40)
				cv2.rectangle(cv_image, (new_x,new_y), (np.int32(x+w-40),np.int32(y+h-40)), (0,255,0), 2)
				cv2.circle(cv_image,(x_circ,y_circ), 10, (0,0,255), -1)
				print(x_circ,y_circ)
				font = cv2.FONT_HERSHEY_SIMPLEX
				'''
				#-----------------------------------------------------------------------------------------------
				#global x_w,y_w
				x_w,y_w = pixel_to_world(x_circ,y_circ)
				puntos_p.append(np.array([x_w[0],y_w[0]]))
				print('real world coordinates',x_w,y_w)
				text_box = 'x='+str(np.round(x_w[0],4))+', y='+str(np.round(y_w[0],4))
				global x_pr, y_pr
				text_02 = 'x='+str(np.round(x_pr,4))+', y='+str(np.round(y_pr,4))
				cv2.putText(cv_image,text_box,(x_circ-100,y_circ+30), font, 0.7,(255,0,0),2,cv2.LINE_AA)
				#cv2.putText(cv_image,text_02,(x_circ-100,y_circ+60), font, 0.7,(0,0,255),2,cv2.LINE_AA)
				global yaw_z
				print('angulo Z: ',yaw_z)
				#cv2.circle(cv_image,(np.int32(x+w-40),np.int32(y+h-40)), 10, (0,0,255), -1)
				#cv2.circle(cv_image,(np.int32(x+w-40-new_x),np.int32(y+h-40)), 10, (255,0,0), -1)
				#cv2.imwrite(r'/home/robotica/PG_ws/src/proyecto_de_grado/imagesdos/image'+str(contador)+'.jpg', cv_image)
			
				#cv2.rectangle(cv_image, (x+100,y+100), (x+w-100,y+h-100), (255,0,0), 2)
				a = float(x)
				b = float(y)
				c = float(w)
				d = float(h)
				e = int((a)+((c-a)/2))
				d = int((b)+((h-b)/2))
				#cv2.circle(img=cv_image,center=(x+100,y+100),radius=10,color=(0,0,255),thickness=2)
				print(e)
				#print(cv_image[(x)+((w-x)/2),(y)+((h-y)/2)])
			
			xw2 = x_w
			yw2 = y_w
			global per_hist,x_w,y_w,per_hist2
			delta_personas = []
			x_w = np.array([xw2[0],0])
			y_w = np.array([yw2[0],0])
			if len(puntos_p)==1:
				x_w = np.array([puntos_p[0][0],0])
				y_w = np.array([puntos_p[0][1],0])
			if len(per_hist2)>1 and len(puntos_p)>1:
				#x_w = np.array([0,0])
				#y_w = np.array([0,0])
				for i in range(len(puntos_p)):
					delta_personas.append(np.linalg.norm(puntos_p[i]-per_hist2[-1]))
				index_min = np.argmin(delta_personas)
				x_w = np.array([puntos_p[index_min][0],0])
				y_w = np.array([puntos_p[index_min][1],0])
			#global per_hist2
			per_hist2.append(np.array([x_w[0],y_w[0]]))
			
		except CvBridgeError as e:
			print(e)
		b,g,r = cv2.split(cv_image)
		img = cv2.merge((r,g,b))  

		# Convert the Image object into a TkPhoto object
		im = PIL.Image.fromarray(img).resize((400,300), PIL.Image.ANTIALIAS)
		#im = PIL.Image.fromarray(img).resize((350,350), PIL.Image.ANTIALIAS)
		imgtk = ImageTk.PhotoImage(image=im)
		label.configure(image=imgtk)
		label.image=imgtk
	
	def print_im():
		global contador #imgtk, 
		contador=contador+1
		'''
		try:
			print('dentro_del_metodo',imgtk)
		except:
			print('paila perro')
		'''
		if contador>1:
			update_image()
		print('contador=',contador)
		root.after(100,print_im)
	print('eaaaaiiirapaz')
	root.after(100,print_im)
	root.mainloop()


# Global vars:
WIDTH = 400
STEP = int(16/2)
QUIVER = (255, 100, 0)
	
def draw_flow(img, flow,  mag_all,step=STEP):
    h, w = img.shape[:2]
    print('image_size',img.shape)
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    print('fx',fx.shape,'fy',fy.shape)
    print('x',x.shape,'y',y.shape)
    print('fx_inside',fx[100])
    
    x2 = x[800:1000]
    y2 = y[800:1000]
    fx2 = fx[800:1000]
    fy2 = fy[800:1000]
    lines2 = np.vstack([x2, y2, x2 + fx2, y2 + fy2]).T.reshape(-1, 2, 2)
    lines2 = np.int32(lines2 + 0.5)
    
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    print('lines',lines.shape)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    #-----------------------------------------------------------------------------
    horz = cv2.normalize(flow[...,0], None, 0, 255, cv2.NORM_MINMAX)     
    vert = cv2.normalize(flow[...,1], None, 0, 255, cv2.NORM_MINMAX)
    horz = horz.astype('uint8')
    vert = vert.astype('uint8')

    # Change here too
    #cv2.imshow('Horizontal Component', horz)
    #cv2.imshow('Vertical Component', vert)
    #-----------------------------------------------------------------------------
    
    
    
    cv2.polylines(vis, lines, 0, QUIVER,3)
    cv2.putText(img = vis,text = str(mag_all),org = (100, 100),fontFace = cv2.FONT_HERSHEY_DUPLEX,fontScale = 3.0,color = (255, 10, 10),thickness = 3)
    #cv2.circle(vis, (ind[0], ind[1]), 1, (0, 0, 255), 1)
    #cv2.polylines(vis, lines2, 0, (0,0,255),5)
    #for (x1, y1), (_x2, _y2) in lines:
    #    cv2.circle(vis, (x1, y1), 1, (0, 255, 0), 1)
    return vis


def thread_graph():
	#time.sleep(2)
	#sys.setrecursionlimit(2000)
	global da,model,contador
	#cv2.namedWindow("Image_window")
	while True:
		if da != 0:
			break
	algo = 0
	if algo==0:
		try:
			cv_image = np.frombuffer(da.data, dtype=np.uint8).reshape(da.height, da.width, -1)
			cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
			prev = cv_image
			prev = imutils.resize(prev, width=WIDTH)
			prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
		except Exception as e:
			print('nada')
			print(e)
			#time.sleep(1)
			thread_graph()
		else:
			while True:
				try:
					cv_image = np.frombuffer(da.data, dtype=np.uint8).reshape(da.height, da.width, -1)
					cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
				except:
					print('ERROR:',contador)
				else:
					img = cv_image
					img = imutils.resize(img, width=WIDTH)
					gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
					print('im1',prevgray.shape,'im2',gray.shape)
					flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
					mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
					xx = mag.shape[0]
					yy = mag.shape[1]
					izq_abajo = mag[int((xx)*2/3)::,0:int(yy/3)]
					mag_all = np.sum(izq_abajo)
					ind = np.unravel_index(np.argmax(izq_abajo,axis=None),izq_abajo.shape)
					max_izq= izq_abajo.max()
					der_abajo = mag[int((xx)*2/3)::,int(yy*2/3)::]
					print('mag',mag.shape)
					print('izq_abajo',izq_abajo.shape)
					print('maximo_izquierda',izq_abajo.max())
					print('maximo_derecha',der_abajo.max())
					data = asarray(flow)
					print(flow.shape)
					#savetxt('data.csv', flow, delimiter=',')
					prevgray = gray
					cv2.imshow('flow', draw_flow(gray, flow,mag_all=mag_all))
					ch = cv2.waitKey(5)
					if ch == 27:
						break
			
	cv2.destroyAllWindows()
#------------------------------------------------------------------------------------------------------------
class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position


def astar(maze, start, end):
    """Returns a list of tuples as a path from the given start to the given end in the given maze"""

    # Create start and end node
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Add the start node
    open_list.append(start_node)

    # Loop until you find the end
    while len(open_list) > 0:

        # Get the current node
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # Pop current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)

        # Found the goal
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1] # Return reversed path

        # Generate children
        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]: # Adjacent squares

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) -1) or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            if maze[node_position[0]][node_position[1]] != 0:
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:

            # Child is on the closed list
            for closed_child in closed_list:
                if child == closed_child:
                    continue

            # Create the f, g, and h values
            child.g = current_node.g + 1
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
            child.f = child.g + child.h

            # Child is already in the open list
            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    continue

            # Add the child to the open list
            open_list.append(child)


def main_astar(maze,start,end):
	global puntos2,inputt2,pasos,path
	path = astar(maze, start, end)
	print('PATHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH',path)
	pathreal = []
	for i in path:
		pathreal.append((int(512*i[1]/20)+13,int(512*i[0]/20)+13))
	pathrealaprox = []
	coordenadaspiso = []
	for i in pathreal:
		pathrealaprox.append(tuple(inputt2[closest_node2(i,puntos2),:][:2]))
		#coordenadaspiso.append(tuple(inputt2[closest_node2(i,puntos2),:][-2:]))
	#pasos = coordenadaspiso
	coor_reales=[]
	for i in range(len(pathreal)):
		xprov,yprov = pixel_to_world(pathreal[i][0],pathreal[i][1])
		coor_reales.append((xprov[0],yprov[0]))
	pasos = coor_reales
	print(coor_reales)
	print('COOR PISO',coordenadaspiso)
	'''
    maze = [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    start = (0, 0)
    end = (7, 6)
	'''
	#maze = np.array(resized_image/resized_image.max(),dtype=int)
	


#if __name__ == '__main__':
#    main()
#------------------------------------------------------------------------------------------------------------
#pasos=[(0.95,0.10),(1,0.20),(1.1,0.25),(1.2,0.3),(1.4,0.35),(1.6,0.4),(1.96,0.5),(2.27,0.8),(2.77,1.2),(3.5,1.9),(3.4,2.1),(3.1,2.4),(2.9,2.7),(2.7,3)]
#pasos = [(1,1),(1,0), (2,1),(2,0), (3,1),(3,0)]
#pasos = [(0.95,0.10),(1,0.3),(1.8,0.6),(3.4,2.1),(2.7,3)]
global pasos
pasos = [(0,2.5),(4,4)]
print(pasos)

#-------------------------------------------------------------------------
global inputt2
with open('/home/robotica/PG_ws/src/proyecto_de_grado/scripts/matrix4.txt', 'r') as f:
    inputt2 = np.array([[np.float(num) for num in line.split(',')] for line in f])
    print(inputt2)
global puntos2
puntos2 = inputt2[:,0:2]
def closest_node2(node, nodes):
    nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node)**2, axis=1)
    return np.argmin(dist_2)
def dist_node_min(node, nodes):
    nodes = np.asarray(nodes)
    dist_2 = np.sqrt(np.sum((nodes - node)**2, axis=1))
    return dist_2.min()
#-------------------------------------------------------------------------
global ydos, esquivar, modo, delta_espacio,num_fren,delta_theta,dis_obs_min
modo = 'seguir_persona'
esquivar = 0 
num_fren = 0
ydos = 0
delta_espacio = 0 
delta_theta = 0
dis_obs_min=0
def frenar():
	global modo
	modo ='frenar'
def thread_threshold():
	global da,model,contador
	while True:
		if da != 0:
			break
	while True:
		try:
			cv_image = np.frombuffer(da.data, dtype=np.uint8).reshape(da.height, da.width, -1)
			img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
			t = time.time()
			twoDimage = img.reshape((-1,3))
			twoDimage = np.float32(twoDimage)
			criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
			K = 3
			attempts=10
			ret,label,center=cv2.kmeans(twoDimage,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
			center = np.uint8(center)
			res = center[label.flatten()]
			#result_image = res.reshape((img.shape))
			result_image = res.reshape(da.height,da.width,-1)
			#result_image = imutils.resize(result_image, width=200)
			kernel = np.ones((5,5),np.uint8)
			result_image = cv2.dilate(result_image,kernel,iterations=2)

			img = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)
			ret, thresh1 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV)

			
			thresh2 = cv2.cvtColor(thresh1,cv2.COLOR_GRAY2BGR)

			contours = cv2.findContours(thresh1,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			
			contours = contours[0] if len(contours) == 2 else contours[1]

			uvobstacles = []
			for cntr in contours:
				x,y,w,h = cv2.boundingRect(cntr)
				cv2.rectangle(thresh2,(x-30,y),(x+w+30,y+h),(255,0,0),2)
				#cv2.circle(thresh2,(int(x+w/2),int(y+h)),4,(0,255,0),-1)
				uvobstacles.append((int(x+w/2),int(y+h)))

			print(time.time()-t)
			#plt.imshow(thresh1,cmap='gray')
			#----------------------------------------------------------------------------------------
			for cntr in contours:
				x,y,w,h = cv2.boundingRect(cntr)
				cv2.rectangle(thresh2, (x-50, y), (x+w+50, y+h), (255, 255, 255), -1)
			#----------------------------------------------------------------------------------------
			uv_obs_aprox = []
			xy_obs = []
			for i in uvobstacles:
			    uv_obs_aprox.append(tuple(inputt2[closest_node2(i,puntos2),:][:2]))
			    xy_obs.append(tuple(inputt2[closest_node2(i,puntos2),:][-2:]))
			print('UV_OBS_APROX',uv_obs_aprox)
			print('XY_OBS',xy_obs)
			#----------------------------------------------------------------------------------------
			
			xy_obs_world = []
			for i in range(len(xy_obs)):
				x_obs_temp, y_obs_temp = pixel_to_world(uvobstacles[i][0],uvobstacles[i][1])
				xy_obs_world.append([np.round(x_obs_temp[0],2),np.round(y_obs_temp[0],2)])
			#----------------------------------------------------------------------------------------
			font = cv2.FONT_HERSHEY_SIMPLEX
			for i in range(len(xy_obs_world)):
				text_box = str(xy_obs_world[i][0])+','+str(xy_obs_world[i][1])
				#cv2.putText(thresh2,text_box,(uvobstacles[i][0]-100,uvobstacles[i][1]+30), font, 0.7,(255,0,0),2,cv2.LINE_AA)
			#----------------------------------------------------------------------------------------
			
			global x_robot,y_robot, dis_obs_min
			dis_obs_min = dist_node_min([x_robot,y_robot],xy_obs_world)
			print('DIST_MINNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN',dis_obs_min)
			global esq, num_fren
			esq = False
			if (dis_obs_min <= 1.2) and num_fren ==0:
				frenar()
				num_fren = num_fren +1
			global esquivar,delta_espacio,delta_theta
			if (dis_obs_min <= 1.2) and (esquivar ==0) and (delta_espacio<0.05) and(delta_theta<0.0174533):
				#ydos = ydos + 1
				#esq = True
				print('nope')
				global ydos, modo
				modo = 'obstaculo'
				ydos = ydos +1
				tdos = "MODO ESQUIVAR ACTIVADO %s"%ydos
				esquivar=esquivar+1
				#-------------------------------------------------------------------------------------------------
				resized_image = cv2.resize(thresh2,(20,20))
				resized_image = cv2.cvtColor(resized_image,cv2.COLOR_RGB2GRAY)
				ret, resized_image = cv2.threshold(resized_image, 120, 255, cv2.THRESH_BINARY)
				resized_image = np.array(resized_image/resized_image.max(),dtype=int)
				print(resized_image)
				#resized_image = np.array(resized_image/resized_image.max(),dtype=int)
				columna = resized_image[:,10]
				first_elem = np.where(columna==0)[0][0]
				end = (first_elem,10)
				start = (20,10)
				print('llegoooooooooooooooo a staaaaaaaaaaaaaaaaaaaaaaar',end)
				main_astar(maze=resized_image,start=start,end=end)
				global path
				#resized_image = cv2.cvtColor(resized_image,cv2.COLOR_GRAY2RGB)
				'''
				blank_image = np.zeros((20,20))
				for i in (len(path)):
					x = path[i][0]
					y = path[i][1]
					blank_image[x,y]=1
					#plt.savefig()
				'''
				#-------------------------------------------------------------------------------------------------
				plt.figure()
				plt.imshow(resized_image)
				plt.show()
				'''
				plt.figure()
				plt.imshow(blank_image)
				plt.show()
				'''
				#print('llegoooooooooooooooo a staaaaaaaaaaaaaaaaaaaaaaar')
				#-------------------------------------------------------------------------------------------------
				#cv2.putText(thresh2,tdos,(200,200), font, 0.7,(0,0,255),2,cv2.LINE_AA)				#esquivar = 1
			#----------------------------------------------------------------------------------------
			thresh2 = cv2.resize(thresh2,(300,300))
			cv2.imshow('threshold', thresh2)
			ch = cv2.waitKey(5)
			if ch == 27:
				break
		except Exception as e:
			print(e)
	cv2.destroyAllWindows()

def get_target_angle(x1,y1,x2,y2):
	# x1 y y1 son la posicion del punto objetivo en el marco global
	# x2 y y2 son la posicion del robot en el marco global
	y = y1-y2
	x = x1-x2
	angulo = math.degrees(math.atan2(y,x))
	if angulo<0:
		angulo = angulo
		#angulo=angulo+360
	return angulo

def get_error_theta(theta1,theta2):
	#theta 1 es el angulo del punto objetivo en el marco global
	#theta 2 es el angulo del robot en el marco global
	error = theta1-theta2
	return error

def get_distance(x1,y1,x2,y2):
	dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
	return dist


def listener():
	global pasos
	pub=rospy.Publisher('/angular_vel',Float32,queue_size=10)
	pub2=rospy.Publisher('/linear_vel',Float32,queue_size=10)
	rospy.init_node('Receptor_imagenes',anonymous = True)
	rospy.Subscriber("rgb/image_raw",Image,callback)
	rospy.Subscriber("vrep/pubPos",Point,callbackDos)
	rospy.Subscriber('odometry',Odometry,callback_odometry)
	rate = rospy.Rate(10) # 10hz

	thread1 = threading.Thread(target=init_ima)
	thread1.start()

	#time.sleep(1)

	#thread2 = threading.Thread(target=thread_graph)
	#thread2.start()

	thread3 = threading.Thread(target=thread_threshold)
	thread3.start()

	print('THREAD 2 INICIADO')
	
	x_obs = -5
	y_obs = 5
	entro = 0
	x_wa = pasos[0][0]
	y_wa = pasos[0][1]
	paso_actual = 0
	error_distance = 0
	primer_paso = True
	entro_seguir_persona = 0
	while not rospy.is_shutdown():
		vel_msg = Float32()
		#global x_w, y_w, x_robot,y_robot,yaw_z
		global x_robot,y_robot,yaw_z,pasos,x_hist,y_hist,delta_espacio,theta_hist,delta_theta,x_w,y_w,dis_obs_min,per_hist,x_w,y_w
		global x_w2,y_w2
		#if x_w2!=0 and y_w2!=0:
		per_hist.append(np.array([x_w[0],y_w[0]]))
		while dis_obs_min>1.2 and entro_seguir_persona==0:
			vel_msg = Float32()
			global x_w, y_w, x_robot,y_robot,yaw_z
			target_angle = get_target_angle(x1=x_w[0],y1=y_w[0],x2=x_robot,y2=y_robot)
			print('target_angle: ',target_angle)
			angulo_robot = yaw_z
			if angulo_robot<0:
				angulo_robot = angulo_robot
			print('angulo_robot: ',angulo_robot)
			error_angle = math.radians(get_error_theta(target_angle,angulo_robot))
			print('error_angle: ',error_angle)
			k = 2
			#vel_msg.data = 0
			vel_msg.data = k*error_angle
			pub.publish(vel_msg)
			print('distancia: ',get_distance(x1=x_w[0],y1=y_w[0],x2=x_robot,y2=y_robot))
			vel_lin = Float32()
			vel_lin = 0
			if abs(error_angle)<0.174533:
				error_distance = get_distance(x1=x_w[0],y1=y_w[0],x2=x_robot,y2=y_robot)-2.5
				k2 = 2
				vel_lin = k2*error_distance
			pub2.publish(vel_lin)

			if dis_obs_min<1.2:
				entro_seguir_persona=1
				break

		#x_wa = x_w
		#y_wa = y_w
		if primer_paso and len(pasos)>5:
			x_wa = pasos[0][0]
			y_wa = pasos[0][1]
			primer_paso = False
		#target_angle = get_target_angle(x1=x_w[0],y1=y_w[0],x2=x_robot,y2=y_robot)
		x_hist.append(x_robot)
		y_hist.append(y_robot)
		theta_hist.append(yaw_z)
		if len(x_hist)>2 and len(y_hist)>2 and len(theta_hist)>2:
			delta_espacio = np.linalg.norm(np.array([x_hist[-2],y_hist[-2]])-np.array([x_robot,y_robot]))
			delta_theta = abs(yaw_z-theta_hist[-2])
		target_angle = get_target_angle(x1=x_wa,y1=y_wa,x2=x_robot,y2=y_robot)
		print('target_angle: ',target_angle)
		angulo_robot = yaw_z
		if angulo_robot<0:
			angulo_robot = angulo_robot
			#angulo_robot = angulo_robot+360
		print('angulo_robot: ',angulo_robot)
		error_angle = math.radians(get_error_theta(target_angle,angulo_robot))
		print('error_angle: ',error_angle)
		angulo_minimo = 0.174533 # 10 grados
		#if abs(error_angle)<angulo_minimo:
		#	k=0
		k = 2
		#vel_msg.data = 0
		vel_msg.data = k*error_angle
		vel_ang_0 = 0
		pub.publish(vel_msg)
		#pub.publish(vel_ang_0)
		#print('distancia: ',get_distance(x1=x_w[0],y1=y_w[0],x2=x_robot,y2=y_robot))
		vel_lin = Float32()
		vel_lin = 0
		#angulo_minimo = 0.0872665 # 5 grados
		
		if abs(error_angle)<angulo_minimo:
			#error_distance = get_distance(x1=x_w[0],y1=y_w[0],x2=x_robot,y2=y_robot)-2.5
			error_distance = get_distance(x1=x_wa,y1=y_wa,x2=x_robot,y2=y_robot)-0.2
			k2 = 2
			vel_lin = k2*error_distance
		vel_lin_0 = 0
		#pub2.publish(0)
		
		global modo
		if modo == 'frenar':
			modo = 'esquivando_iniciado'
			vel_lin = 0
			c_p = 0
			while c_p<3:
				pub2.publish(vel_lin)
				pub.publish(vel_lin)
				time.sleep(1)
				c_p=c_p+1
		
		pub2.publish(vel_lin)

		#---------------------------------------------------------------------------------------
		print('---------------------------------------------------------------------------------')
		print('error_angulooooooo',error_angle)
		print('error_distamceeeeeee',error_distance)
		print('coordenadas',x_wa,y_wa)
		print('Numero(entro)',entro)
		print('paso_actual: ',paso_actual)
		print('---------------------------------------------------------------------------------')

		#if abs(error_angle)<0.0174533 and error_distance<0.1:
		if abs(error_angle)<0.0174533 and error_distance<0.1:
			print('entrooooooooooooooo')
			if paso_actual < len(pasos)-1:
				paso_actual = paso_actual +1 
				global pasos
				x_wa = pasos[paso_actual][0]
				y_wa = pasos[paso_actual][1]
				if paso_actual == len(pasos)-1:
					while True:
						vel_msg = Float32()
						global x_w, y_w, x_robot,y_robot,yaw_z
						target_angle = get_target_angle(x1=x_w[0],y1=y_w[0],x2=x_robot,y2=y_robot)
						print('target_angle: ',target_angle)
						angulo_robot = yaw_z
						if angulo_robot<0:
							angulo_robot = angulo_robot
						print('angulo_robot: ',angulo_robot)
						error_angle = math.radians(get_error_theta(target_angle,angulo_robot))
						print('error_angle: ',error_angle)
						k = 2
						#vel_msg.data = 0
						vel_msg.data = k*error_angle
						pub.publish(vel_msg)
						print('distancia: ',get_distance(x1=x_w[0],y1=y_w[0],x2=x_robot,y2=y_robot))
						vel_lin = Float32()
						vel_lin = 0
						if abs(error_angle)<0.174533:
							error_distance = get_distance(x1=x_w[0],y1=y_w[0],x2=x_robot,y2=y_robot)-2.5
							k2 = 2
							vel_lin = k2*error_distance
						pub2.publish(vel_lin)
			'''
			if entro == 1:
				if error_distance<0.1:
					print('LLEGAMOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOS')
					x_w = pasos[paso_actual+5][0]
					y_w = pasos[paso_actual+5][1]
						
			if entro == 0:
				x_w = 1.2*(np.cos(math.radians(angulo_robot)))+x_robot
				y_w = 1.2*(np.sin(math.radians(angulo_robot)))+y_robot
				entro = entro + 1
				paso_actual = paso_actual +1
				
			'''
		#---------------------------------------------------------------------------------------
		#pub2.publish(vel_ang_0)
		rate.sleep()

	if rospy.is_shutdown():
		print('murio')
		plt.figure(figsize=(5,5))
		plt.plot(x_hist,y_hist,label='Robot path')

		xest = [i[0] for i in pasos]
		yest = [i[1] for i in pasos]

		plt.scatter(xest,yest,label='A star path',color='red')
		plt.legend()
		plt.xlabel('x[m]')
		plt.ylabel('y[m]')
		plt.title('A star path vs. robot path')

		plt.savefig('rutarobot.jpg',bbox_inches='tight')

	#rospy.spin()

if __name__=='__main__':
	HOGCV = cv2.HOGDescriptor()
	HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
	global root
	listener()
