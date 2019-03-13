#!/usr/bin/env python
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from decimal import Decimal
import math

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
from numba import jit




class ipm_processes:
    # Setting some important variables
    width = 0
    height = 0
    Pbox_min_x1 = 0
    Pbox_min_y1 = -1500
    Pbox_max_x1 = 4000
    Pbox_max_y1 = abs(Pbox_min_y1)
    #Second box, which is bigger, just for report purposes
    Pbox_min_x2 = -3000
    Pbox_min_y2 = -4000
    Pbox_max_x2 = 9000
    Pbox_max_y2 = abs(Pbox_min_y2)

    flag_debugging_mode = 0

    def __init__(self):
        self.bridge = CvBridge()

    def setFlagDebug(self, flag_debugging_mode):
        self.flag_debugging_mode = flag_debugging_mode

    # Setting up the setting functions

    def set_width_height(self, width, height):
        self.width = width
        self.height = height


    def image_rescale_debugg(self,k,original_width,original_height,rescaling_factor):
        #Function that for debugging purposes, where adjust important IPM properties according to inserted images
        img_path = os.path.join(
            "/home/hellxberg/ws_thesis/src/ipm_perception/src/", "the_real_one.png")
        inserted_img = cv2.imread(img_path)
        width_new = np.size(inserted_img, 1)
        height_new = np.size(inserted_img, 0)

        # Calculate scaling factors
        scale_x = original_width/float(width_new)
        scale_y = original_height/float(height_new)
        rescaling_x = scale_x*(rescaling_factor)
        rescaling_y = scale_y*(rescaling_factor)
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
            
        new_width = int(width/rescaling_x)
        new_height = int(height/rescaling_y)
        new_dsize = (new_width, new_height)
        rescaled_image = cv2.resize(original_img, dsize=new_dsize,interpolation=cv2.INTER_CUBIC)
        k[0][0] = k[0][0] / (rescaling_x*1.0)
        k[0][2] = k[0][2] / (rescaling_x*1.0)
        k[1][1] = k[1][1] / (rescaling_y*1.0)
        k[1][2] = k[1][2] / (rescaling_y*1.0)
        return rescaled_image,k,new_width,new_height


    def IPM(self, infor):
        print("Begginning of ipm  processing")
        # Function where the main process of IPM happens
        
        # Setting up constant variables -------

        # REscalling, default is 100%
        rsf_scale = 15 #IN percentage
        rsf_factor = (rsf_scale/100.0)
        rescaling_factor = 10 #Rescalling of base image
        # If flag_debugging_mode==0 only cropped IPM, else all the IPM didactic images
        flag_debugging_mode = self.flag_debugging_mode

        # Colocar a matriz de projecao em formato de array(ori_image_r,k,width,height)
        k = np.array([[infor.P[0], infor.P[1], infor.P[2], infor.P[3]], [infor.P[4], infor.P[5], infor.P[6], infor.P[7]], [infor.P[8], infor.P[9], infor.P[10], infor.P[11]]])
        proj = infor.trans_matrix
        ori_image = infor.cv2_img
        width=np.size(ori_image,1)
        height=np.size(ori_image,0)
        
        flag_other_image=0
        if(flag_other_image == 0):
            (ori_image_r,k,width,height)=self.image_rescale(ori_image,k,width,height,rescaling_factor)
            #(ori_image_r, k, width, height) = self.preliminary_rescaling(rescaling_factor, rescaling_factor, k, ori_image,width,height)
        else:
            (ori_image_r,k,width,height)=self.image_rescale_debugg(k,width,height,rescaling_factor)
            #(ori_image_r, k, width, height) = self.adjust_out_imgs(rescaling_factor, k, ori_image,width,height)

        
        # Definir variaveis representativas da area de projecao ------------------------------
        image_measures=[]
        image_measures.append(height)
        image_measures.append(width)
        Pbox_min_y1 = int(self.Pbox_min_y1*rsf_factor)
        Pbox_max_y1 = int(self.Pbox_max_y1*rsf_factor)
        Pbox_min_x1 = int(self.Pbox_min_x1*rsf_factor)
        Pbox_max_x1 = int(self.Pbox_max_x1*rsf_factor)
        box_limits1=[]
        box_limits1.append(Pbox_min_x1)
        box_limits1.append(Pbox_max_x1)
        box_limits1.append(Pbox_min_y1)
        box_limits1.append(Pbox_max_y1)
        visi_width = Pbox_max_x1-Pbox_min_x1
        offset_height_box = abs(Pbox_min_y1)
        DISTXmax = visi_width
        DISTYmax = 2*offset_height_box
        measure_main=[]
        measure_main.append(DISTYmax)
        measure_main.append(DISTXmax)


        # Rescalling proj matrix
        proj[0][3] = proj[0][3]*rsf_factor
        proj[1][3] = proj[1][3]*rsf_factor
        proj[2][3] = proj[2][3]*rsf_factor
        Pro2 = np.dot(k, proj)

        #For interpolation
        x_int=[]
        y_int=[]
        value_int=[]

        if(flag_debugging_mode==1):
            #Set up information necessary for a secondary fx = Pbox_min_x1
            Pbox_min_x2 = int(self.Pbox_min_x2*rsf_factor)
            Pbox_min_y2 = int(self.Pbox_min_y2*rsf_factor)
            Pbox_max_x2 = int(self.Pbox_max_x2*rsf_factor)
            Pbox_max_y2 = int(self.Pbox_max_y2*rsf_factor)
            box_limits2=[]
            box_limits2.append(Pbox_min_x2)
            box_limits2.append(Pbox_max_x2)
            box_limits2.append(Pbox_min_y2)
            box_limits2.append(Pbox_max_y2)
            visi_width2 = Pbox_max_x2-Pbox_min_x2
            offset_height_box2 = abs(Pbox_min_y2)
            DISTXmax2 = visi_width2
            DISTYmax2 = 2*offset_height_box2
            measure_debug=[]
            measure_debug.append(DISTYmax2)
            measure_debug.append(DISTXmax2)


        start = timer()
        #Create mapping variables
        if(flag_debugging_mode==0):
            (output_img,Mapx,Mapy,Mapped,Gmapped)=self.pixel_mapping(Pro2,image_measures,box_limits1,measure_main)
        else:
            (output_img,Mapx,Mapy,Mapped,Gmapped,output_img2,Mapped2,Gmapped2,colour_img) =self.pixel_mapping(Pro2,image_measures,box_limits1,measure_main,box_limits2,measure_debug)


        # Ciclo para criar a imagem com a nova informacao
        for u in range(width):
            for v in range(height):
                if(flag_debugging_mode==1):
                    if(Mapped2[v][u]):
                        output_img2[int(Mapy[v][u]) + abs(Pbox_min_y2)-1][int(Mapx[v][u]) + abs(Pbox_min_x2)-1] += (ori_image_r[v][u])/Gmapped2[Mapy[v][u]][Mapx[v][u]]
                        colour_img[int(Mapy[v][u]) + abs(Pbox_min_y2)-1][int(Mapx[v][u]) + abs(Pbox_min_x2)-1]=[0,0,200]
                if(Mapped[v][u]):
                    output_img[int(Mapy[v][u]) + abs(Pbox_min_y1)-1][int(Mapx[v][u]) + abs(Pbox_min_x1)-1] = (ori_image_r[v][u])#/Gmapped[Mapy[v][u]][Mapx[v][u]]
                    #Acrescento da interpolacao de pixels, ira ser posteriormente alterado para suportar varias imagens de IPM
                    #Por agora sera aqui aplicado
                    x_int.append(int(Mapx[v][u]) + abs(Pbox_min_x1)-1)
                    y_int.append(int(Mapy[v][u]) + abs(Pbox_min_y1)-1)
                    value_int.append(ori_image_r[v][u])
                    if(flag_debugging_mode==1):
                        colour_img[int(Mapy[v][u]) + abs(Pbox_min_y2)-1][int(Mapx[v][u]) + abs(Pbox_min_x2)-1]=[0,200,0]
                

        # Cyle for for interpolation
        end = timer()
        print("Time"+str(end-start))
        #self.for_report(output_img)
        
        #print("Images Published")
        if(flag_debugging_mode==1):

            cv2.imwrite('/home/hellxberg/ws_thesis/src/ipm_perception/src/Pos-IPM_cropped.jpeg', output_img)
            cv2.imwrite('/home/hellxberg/ws_thesis/src/ipm_perception/src/Pos-IPM_Full.jpeg', output_img2)
            cv2.imwrite('/home/hellxberg/ws_thesis/src/ipm_perception/src/Pos-IPM_colour.jpeg', colour_img)
            cv2.imwrite('/home/hellxberg/ws_thesis/src/ipm_perception/src/Original_image.jpeg',ori_image)

        # cv2.imwrite('/home/hellxberg/ws_thesis/src/ipm_perception/src/test1-IPM.jpeg',colour_img)
        # cv2.imwrite('/home/hellxberg/ws_thesis/src/ipm_perception/src/test2-IPM.jpeg',out_segment_img)
        #cv2.imshow("meh",output_img)
        #cv2.waitKey(0)
        
        #cv2.imwrite('/home/hellxberg/ws_thesis/src/ipm_perception/src/Pos-IPM.jpeg', output_img)
        
        #interpolated_img=self.interpolation_first_phase(DISTYmax, DISTXmax,x_int,y_int,value_int)
        #self.perspective_visualization(interpolated_img,Pbox_min_x1,Pbox_max_x1,Pbox_min_y1,Pbox_max_y1,rsf_factor)
        return (output_img)

    def calculate_new_XY(self, Pro, point_x, point_y):
        D = point_x*(Pro[1][0]*Pro[2][1]-Pro[1][1]*Pro[2][0])+point_y*(Pro[2][0] * Pro[0][1]-Pro[2][1]*Pro[0][0])+Pro[0][0]*Pro[1][1]-Pro[0][1]*Pro[1][0]
        X = (point_x*(Pro[1][1]*Pro[2][3]-Pro[1][3]*Pro[2][1])+point_y*(Pro[0][3]*Pro[2][1]-Pro[0][1]*Pro[2][3])-Pro[0][3]*Pro[1][1]+Pro[0][1]*Pro[1][3])/(D*1.0)
        Y = (point_x*(Pro[1][0]*Pro[2][3]-Pro[1][3]*Pro[2][0])+point_y*(Pro[0][3] * Pro[2][0]-Pro[0][0]*Pro[2][3])+Pro[0][0]*Pro[1][3]-Pro[0][3]*Pro[1][0])/(D*1.0)
        return int(X), int(Y)

    def interpolation_first_phase(self,height, width,x_int,y_int,value_int):
        #Function to interpolate from given values

        # target grid to interpolate to
        xi = yi = np.arange(0,width,1)
        output_img = np.zeros((height, width, 3), np.uint8)
        for i in range(len(y_int)):
            output_img[y_int[i]][x_int[i]]=value_int[i]

        #cv2.imshow("meh",output_img)
        #cv2.waitKey(0)

        #print(str(xi))
        xi,yi = np.meshgrid(xi,yi)
        #print(str(x))
        print(str(xi))
        # set mask
        #mask = (xi > 0.5) & (xi < 0.6) & (yi > 0.5) & (yi < 0.6)

        # interpolate
        zi = griddata((x_int,y_int),value_int,(xi,yi),method='linear')
        for y in range(height):
            for x in range(width):
                output_img[y][x]=zi[y][x]
        #cv2.imshow("meh",output_img)
        #cv2.waitKey(0)
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
        

    def pixel_mapping(self,Pro2,img_meas,box_limits1,measure_main,*argv):
        #Function to create the mapes Mapx & Mapy
        #Pro2-Matrice to transform
        #box_limits1 & box_limits2-matrices containing the limits of the projection box
        #image_measures-matrice with image dimensions
        start = timer()
        flag_debug1=0
        
        if(len(argv)==0):
            flag_debug1=0

        elif(len(argv)==2):
            measure_debug=argv[1]
            box_limits2=argv[0]
            flag_debug1=1
            Mapped2 = np.zeros((img_meas[0], img_meas[1]), dtype=int)
            output_img2 = np.zeros((measure_debug[0], measure_debug[1], 3), np.uint8)
            Gmapped2= np.zeros((measure_debug[0], measure_debug[1]), dtype=int)
            colour_img = np.ones((measure_debug[0], measure_debug[1], 3), np.uint8)*255

        #Define here the non debugging variables ------------------------------
            
        Mapx = np.zeros((img_meas[0], img_meas[1]), dtype=int)
        Mapy = np.zeros((img_meas[0], img_meas[1]), dtype=int)
        Mapped = np.zeros((img_meas[0], img_meas[1]), dtype=int)
        output_img = np.zeros((measure_main[0],measure_main[1], 3), np.uint8)
        Gmapped = np.zeros((measure_main[0],measure_main[1]), dtype=int)
        print("TIme making the images "+str(timer()-start))
        start=timer()
        for u in range(img_meas[1]):
            for v in range(img_meas[0]):
                x, y =self.calculate_new_XY(Pro2, u, v)
                
                if(((x > box_limits1[0]) and (x < box_limits1[1])) and ((y > box_limits1[2]) and (y <box_limits1[3]))):
                    Mapx[v][u] = x
                    Mapy[v][u] = y
                    Mapped[v][u] = 1
                    Gmapped[Mapy[v][u]][Mapx[v][u]] += 1
                if(flag_debug1==1):
                    if(((x > box_limits2[0]) and (x < box_limits2[1])) and ((y > box_limits2[2]) and (y < box_limits2[3]))):
                        Mapped2[v][u] = 1
                        Gmapped2[Mapy[v][u]][Mapx[v][u]] += 1

        print("Time in the loop "+str(timer()-start))

        if(flag_debug1==0):
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
        plt.show()

