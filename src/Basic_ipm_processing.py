#!/usr/bin/env python
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from decimal import Decimal
import math
from shapely.geometry import MultiPoint
import sys
import matplotlib.pyplot as plt
import copy
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cbook import get_sample_data
from matplotlib import *
import os
import time
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import time
from timeit import default_timer as timer
from scipy.interpolate import griddata
import laser_geometry.laser_geometry as lg
import sensor_msgs.point_cloud2 as pc2
from cloud_points_processing import cloud_points_processing



class ipm_processes:

    def __init__(self):
        self.bridge = CvBridge()
        #Construct cars limits
        #TODO put car limits in cloud_points class
        self.car_hitbox_max_x=0.1*1000
        self.car_hitbox_min_x=-3.0*1000
        self.car_hitbox_max_y=0.8*1000
        self.car_hitbox_min_y=-0.8*1000
        # Setting some important variables
        self.width = 0
        self.height = 0
        self.Pbox_min_x1 = 0
        self.Pbox_min_y1 = -1500
        self.Pbox_max_x1 = 4000
        self.Pbox_max_y1 = abs(self.Pbox_min_y1)
        #Second box, which is bigger, just for report purposes
        self.Pbox_min_x2 = -3000
        self.Pbox_min_y2 = -4000
        self.Pbox_max_x2 = 9000
        self.Pbox_max_y2 = abs(self.Pbox_min_y2)
        self.x_cloud=[]
        self.y_cloud=[]
        self.flag_debugging_mode = 0
        self.x_cloud=[]
        self.y_cloud=[]
        #Scaling variables
        self.rsf_scale = 15 #IN percentage
        self.rsf_factor = (self.rsf_scale/100.0)
        self.rescaling_factor = 10 #Rescalling of base image
        #Update projection variables with rsf scaling
        self.Pbox_min_x1=int(self.Pbox_min_x1*self.rsf_factor)
        self.Pbox_min_y1 =int(self.Pbox_min_y1*self.rsf_factor)
        self.Pbox_max_x1 = int(self.Pbox_max_x1*self.rsf_factor)
        self.Pbox_max_y1 = int(self.Pbox_max_y1*self.rsf_factor)
        self.Pbox_min_x2=int(self.Pbox_min_x2*self.rsf_factor)
        self.Pbox_min_y2 =int(self.Pbox_min_y2*self.rsf_factor)
        self.Pbox_max_x2 = int(self.Pbox_max_x2*self.rsf_factor)
        self.Pbox_max_y2 = int(self.Pbox_max_y2*self.rsf_factor)
        self.visi_width = self.Pbox_max_x1-self.Pbox_min_x1
        self.offset_height_box = abs(self.Pbox_min_y1)
        self.DISTXmax = self.visi_width
        self.DISTYmax = 2*self.offset_height_box
        self.visi_width2 = self.Pbox_max_x2-self.Pbox_min_x2
        self.offset_height_box2 = abs(self.Pbox_min_y2)
        self.DISTXmax2 = self.visi_width2
        self.DISTYmax2 = 2*self.offset_height_box2


    # Setting up the setting functions
    def set_width_height(self, width, height):
        self.width1 = width
        self.height1 = height


    def image_rescale_debugg(self,k,original_width,original_height):
        #Function that for debugging purposes, where adjust important IPM properties according to inserted images
        img_path = os.path.join(
            "/home/hellxberg/ws_thesis/src/ipm_perception/src/", "the_real_one.png")
        inserted_img = cv2.imread(img_path)
        width_new = np.size(inserted_img, 1)
        height_new = np.size(inserted_img, 0)

        # Calculate scaling factors
        scale_x = original_width/float(width_new)
        scale_y = original_height/float(height_new)
        rescaling_x = scale_x*(self.rescaling_factor)
        rescaling_y = scale_y*(self.rescaling_factor)
        (new_inserted_image,k,width,height)=self.image_rescale(inserted_img,k,original_width,original_height,rescaling_x,rescaling_y)
        return new_inserted_image,k,width,height


    def image_rescale(self,original_img,k,width,height,*argv):
        #Function to rescale an image
        #Prepared for symetric and unsymetric rescalinng
        #argv refers to rescaling factors, can be argv=(rescaling_x,rescaling_y) or argv=rescaling
        if(len(argv)==2):
            rescaling_x=argv[0]
            rescaling_y=argv[1]
        elif(len(argv)==1):
            rescaling_x=argv[0]
            rescaling_y=argv[0]
            
        self.width = int(width/rescaling_x)
        self.height = int(height/rescaling_y)
        new_dsize = (self.width, self.height)
        rescaled_image = cv2.resize(original_img, dsize=new_dsize,interpolation=cv2.INTER_CUBIC)
        k[0][0] = k[0][0] / (rescaling_x*1.0)
        k[0][2] = k[0][2] / (rescaling_x*1.0)
        k[1][1] = k[1][1] / (rescaling_y*1.0)
        k[1][2] = k[1][2] / (rescaling_y*1.0)
        return rescaled_image,k


    def IPM(self, infor):
        print("Begginning of ipm  processing")
        # Function where the main process of IPM happens

        # Colocar a matriz de projecao em formato de array(ori_image_r,k,width,height)
        k = np.array([[infor.P[0], infor.P[1], infor.P[2], infor.P[3]], [infor.P[4], infor.P[5], infor.P[6], infor.P[7]], [infor.P[8], infor.P[9], infor.P[10], infor.P[11]]])
        proj = infor.trans_matrix
        ori_image = infor.cv2_img
        img_width=np.size(ori_image,1)
        img_height=np.size(ori_image,0)
        flag_other_image=0
        if(flag_other_image == 0):
            (ori_image_r,k)=self.image_rescale(ori_image,k,img_width,img_height,self.rescaling_factor)
            #(ori_image_r, k, width, height) = self.preliminary_rescaling(rescaling_factor, rescaling_factor, k, ori_image,width,height)
        else:
            (ori_image_r,k)=self.image_rescale_debugg(k,img_width,img_height)
            #(ori_image_r, k, width, height) = self.adjust_out_imgs(rescaling_factor, k, ori_image,width,height)

        
        # Rescalling proj matrix
        proj[0][3] = proj[0][3]*self.rsf_factor
        proj[1][3] = proj[1][3]*self.rsf_factor
        proj[2][3] = proj[2][3]*self.rsf_factor
        Pro2 = np.dot(k, proj)

        #For interpolation
        x_int=[]
        y_int=[]
        value_int=[]

        start = timer()
        #Create mapping variables
        if(self.flag_debugging_mode==0):
            (output_img,Mapx,Mapy,Mapped,Gmapped)=self.pixel_mapping(Pro2)
        else:
            (output_img,Mapx,Mapy,Mapped,Gmapped,output_img2,Mapped2,Gmapped2,colour_img) =self.pixel_mapping(Pro2)


        # Ciclo para criar a imagem com a nova informacao
        for u in range(self.width):
            for v in range(self.height):
                if(self.flag_debugging_mode==1):
                    if(Mapped2[v][u]):
                        output_img2[int(Mapy[v][u]) + abs(self.Pbox_min_y2)-1][int(Mapx[v][u]) + abs(self.Pbox_min_x2)-1] += (ori_image_r[v][u])/Gmapped2[Mapy[v][u]][Mapx[v][u]]
                        colour_img[int(Mapy[v][u]) + abs(self.Pbox_min_y2)-1][int(Mapx[v][u]) + abs(self.Pbox_min_x2)-1]=[0,0,200]
                if(Mapped[v][u]):
                    output_img[int(Mapy[v][u]) + abs(self.Pbox_min_y1)-1][int(Mapx[v][u]) + abs(self.Pbox_min_x1)-1] = (ori_image_r[v][u])#/Gmapped[Mapy[v][u]][Mapx[v][u]]
                    #Acrescento da interpolacao de pixels, ira ser posteriormente alterado para suportar varias imagens de IPM
                    #Por agora sera aqui aplicado
                    x_int.append(int(Mapx[v][u]) + abs(self.Pbox_min_x1)-1)
                    y_int.append(int(Mapy[v][u]) + abs(self.Pbox_min_y1)-1)
                    value_int.append(ori_image_r[v][u])
                    if(self.flag_debugging_mode==1):
                        colour_img[int(Mapy[v][u]) + abs(self.Pbox_min_y2)-1][int(Mapx[v][u]) + abs(self.Pbox_min_x2)-1]=[0,200,0]
                

        # Cyle for for interpolation
        end = timer()
        print("Time"+str(end-start))
        #self.for_report(output_img)
        
        #print("Images Published")
        if(self.flag_debugging_mode==1):

            cv2.imwrite('/home/hellxberg/ws_thesis/src/ipm_perception/src/Pos-IPM_cropped.jpeg', output_img)
            cv2.imwrite('/home/hellxberg/ws_thesis/src/ipm_perception/src/Pos-IPM_Full.jpeg', output_img2)
            cv2.imwrite('/home/hellxberg/ws_thesis/src/ipm_perception/src/Pos-IPM_colour.jpeg', colour_img)
            cv2.imwrite('/home/hellxberg/ws_thesis/src/ipm_perception/src/Original_image.jpeg',ori_image)

        cv2.imshow("Original IPM image",output_img)
        cv2.waitKey(0)

        
        interpolated_img=self.interpolation_first_phase(self.DISTYmax, self.DISTXmax,x_int,y_int,value_int)
        #self.perspective_visualization(interpolated_img,Pbox_min_x1,Pbox_max_x1,Pbox_min_y1,Pbox_max_y1,rsf_factor)
        self.points_in_image(interpolated_img)
        return (output_img)

    def calculate_new_XY(self, Pro, point_x, point_y):
        D = point_x*(Pro[1][0]*Pro[2][1]-Pro[1][1]*Pro[2][0])+point_y*(Pro[2][0] * Pro[0][1]-Pro[2][1]*Pro[0][0])+Pro[0][0]*Pro[1][1]-Pro[0][1]*Pro[1][0]
        X = (point_x*(Pro[1][1]*Pro[2][3]-Pro[1][3]*Pro[2][1])+point_y*(Pro[0][3]*Pro[2][1]-Pro[0][1]*Pro[2][3])-Pro[0][3]*Pro[1][1]+Pro[0][1]*Pro[1][3])/(D*1.0)
        Y = (point_x*(Pro[1][0]*Pro[2][3]-Pro[1][3]*Pro[2][0])+point_y*(Pro[0][3] * Pro[2][0]-Pro[0][0]*Pro[2][3])+Pro[0][0]*Pro[1][3]-Pro[0][3]*Pro[1][0])/(D*1.0)
        return int(X), int(Y)


    def points_in_image(self,interpolated_img):
        #Function for debugginf purposes
        #functions which puts existing coordinates over ipm image
        plt.imshow(interpolated_img,extent=[self.Pbox_min_x1, self.Pbox_max_x1, self.Pbox_min_y1, self.Pbox_max_y1])
        plt.plot(self.x_cloud*self.rescaling_factor,self.y_cloud*self.rescaling_factor,'o')
        plt.show()

    def interpolation_first_phase(self,height, width,x_int,y_int,value_int):
        #Function to interpolate from given values

        # target grid to interpolate to
        xi = yi = np.arange(0,width,1)
        output_img = np.zeros((height, width, 3), np.uint8)
        for i in range(len(y_int)):
            output_img[y_int[i]][x_int[i]]=value_int[i]

        xi,yi = np.meshgrid(xi,yi)

        # interpolate
        zi = griddata((x_int,y_int),value_int,(xi,yi),method='linear')
        for y in range(height):
            for x in range(width):
                output_img[y][x]=zi[y][x]
        cv2.imshow("Interpolated image",output_img)
        cv2.waitKey(0)
        return output_img
        

    def didactic_IPM_visualization(self,A,Pbox_max_x1,Pbox_max_y1,Pbox_min_x1,Pbox_min_y1):
        width=np.size(A,1)
        height=np.size(A,0)
        stepx,stepy=(Pbox_max_x1-Pbox_min_x1)/float(width),(Pbox_max_y1-Pbox_min_y1)/float(height)
        x1=np.arange(Pbox_min_x1,Pbox_max_x1,stepx)
        y1=np.arange(Pbox_min_y1,Pbox_max_y1,stepy)
        x1,y1=np.meshgrid(x1,y1)
        fig=plt.figure()
        ax=fig.add_subplot(111,projection='3d')
        ax.plot_surface(x1,y1,np.atleast_2d(0.0),rstride=1,cstride=1,facecolors=A/255.0,shade=False)
        plt.show()

    def for_report(self,img):
        counter=0
        height=np.size(img,0)
        width=np.size(img,1)
        for y in range(height):
            for x in range(width):
                if(img[y][x]==[0,0,0]).all():
                    counter+=1
        total_pixels=width*height
        per=(float(counter)/total_pixels)*100.0
        print("NUmber of pixels "+str(total_pixels))
        print("Number of black pixels "+str(counter))
        print("Percentage of black pixels "+str(per))
        print("Width"+str(width))
        print("Height"+str(height))
        

    def pixel_mapping(self,Pro2):
        #Function to create the mapes Mapx & Mapy
        #Pro2-Matrice to transform
        #box_limits1 & box_limits2-matrices containing the limits of the projection box
        #image_measures-matrice with image dimensions
        start = timer()
        if(self.flag_debugging_mode==1):
            Mapped2 = np.zeros((self.height, self.width), dtype=int)
            output_img2 = np.zeros((self.DISTYmax2, self.DISTXmax2, 3), np.uint8)
            Gmapped2= np.zeros((self.DISTYmax2, self.DISTXmax2), dtype=int)
            colour_img = np.ones((self.DISTYmax2, self.DISTXmax2, 3), np.uint8)*255

        #Define here the non debugging variables ------------------------------
        Mapx = np.zeros((self.height, self.width), dtype=int)
        Mapy = np.zeros((self.height, self.width), dtype=int)
        Mapped = np.zeros((self.height, self.width), dtype=int)
        output_img = np.zeros((self.DISTYmax,self.DISTXmax, 3), np.uint8)
        Gmapped = np.zeros((self.DISTYmax,self.DISTXmax), dtype=int)
        print("TIme making the images "+str(timer()-start))
        start=timer()
        for u in range(self.width):
            for v in range(self.height):
                x, y =self.calculate_new_XY(Pro2, u, v)
                if(((x > self.Pbox_min_x1) and (x < self.Pbox_max_x1)) and ((y > self.Pbox_min_y1) and (y <self.Pbox_max_y1))):
                    Mapx[v][u] = x
                    Mapy[v][u] = y
                    Mapped[v][u] = 1
                    Gmapped[Mapy[v][u]][Mapx[v][u]] += 1
                if(self.flag_debugging_mode==1):
                    if(((x > self.Pbox_min_x2) and (x < self.Pbox_max_x2)) and ((y > self.Pbox_min_y2) and (y < self.Pbox_max_y2))):
                        Mapped2[v][u] = 1
                        Gmapped2[Mapy[v][u]][Mapx[v][u]] += 1

        print("Time in the loop "+str(timer()-start))
        if(self.flag_debugging_mode==0):
            return output_img,Mapx,Mapy,Mapped,Gmapped  
        else:
            return (output_img,Mapx,Mapy,Mapped,Gmapped,output_img2,Mapped2,Gmapped2,colour_img)




    def perspective_visualization(self,A,Pbox_min_x,Pbox_max_x,Pbox_min_y,Pbox_max_y,rsf_factor):
        #Visualization of IPM image for validation of results
        fig, ax = plt.subplots()
        marker_position_x=2500
        marker2=3500
        x =[0,marker_position_x*rsf_factor,marker2*rsf_factor]
        y=[0,0,0]
        ax.imshow(A, extent=[Pbox_min_x, Pbox_max_x, Pbox_min_y, Pbox_max_y])
        ax.scatter(x, y,marker='o', s=10, color='r')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Current IPM image')
        plt.show()



    def sensor_fusion(self,left_laser,right_laser,left_tf,right_tf):
        #Function that organizes the cloud point information
        #TODO organize code to accept any number of cloud points
        cpoint=cloud_points_processing()
        print("mehhhhhhhhhhh")
        laser_points=[]
        laser_tf=[]
        laser_points.append(left_laser)
        #laser_points.append(right_laser)
        laser_tf.append(left_tf)
        #laser_tf.append(right_tf)
        (x,y)=cpoint.cloud2cartesian(laser_points,laser_tf)
        self.x_cloud=x
        self.y_cloud=y
        

