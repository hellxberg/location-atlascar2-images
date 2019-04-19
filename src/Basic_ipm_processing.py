#!/usr/bin/env python
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from decimal import Decimal
import math
from shapely.geometry import MultiPoint
from shapely.geometry import Point
from shapely.geometry import Polygon,MultiPolygon
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
from timeit import default_timer as timer
from scipy.interpolate import griddata
import laser_geometry.laser_geometry as lg
import sensor_msgs.point_cloud2 as pc2
from cloud_points_processing import cloud_points_processing
from matplotlib.path import Path 
import matplotlib.patches as patches


flag_verbose=True

def my_print(str):
    if(flag_verbose):
        print(str)

class ipm_processes:

    def __init__(self,rsf_factor):
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
        self.Pbox_min_y1 = -3000
        self.Pbox_max_x1 = 7000
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
        self.rsf_factor = rsf_factor
        self.rescaling_factor = 8 #Rescalling of base image
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
        self.coll_coord_x=[]
        self.coll_coord_y=[]
        self.center_coll_x=0
        self.center_coll_y=0
        self.offset_cloud_x=0
        self.offset_cloud_y=0
        self.initial_coll_coords=[]


    def set_coll_param(self,coll_coord_x,coll_coord_y):
        min_x=min(coll_coord_x)
        max_x=max(coll_coord_x)
        min_y=min(coll_coord_y)
        max_y=max(coll_coord_y)
        offset_poly_x=abs(min_x)
        offset_poly_y=abs(min_y)
        center_coll_x=int(sum(coll_coord_x)/len(coll_coord_x))+offset_poly_x
        center_coll_y=int(sum(coll_coord_y)/len(coll_coord_y))+offset_poly_y
        self.center_coll_x=center_coll_x
        self.center_coll_y=center_coll_y
        self.offset_cloud_x=offset_poly_x
        self.offset_cloud_y=offset_poly_y
        self.coll_coord_x=coll_coord_x
        self.coll_coord_y=coll_coord_y

    def set_coll_coordinates(self,vertices_x,vertices_y):
        self.vertices_x=vertices_x
        self.vertices_y=vertices_y


    def set_full_polygon_coords(self,coll_coord_x,coll_coord_y):
        self.coll_coord_x=coll_coord_x
        self.coll_coord_y=coll_coord_y 

    def set_initial_coll_coords(self,initial_cloud_coords):
        self.initial_coll_coords=initial_cloud_coords

    # Setting up the setting functions
    def set_width_height(self, width, height):
        self.width1 = width
        self.height1 = height

    def set_polygon_final(self,final_poly):
        self.final_poly=final_poly


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


    def IPM(self, infor,rescaling_factor):
        self.rescaling_factor=rescaling_factor
        # Function where the main process of IPM happens
        start1=timer()
        # Colocar a matriz de projecao em formato de array(ori_image_r,k,width,height)
        k = np.array([[infor.P[0], infor.P[1], infor.P[2], infor.P[3]], [infor.P[4], infor.P[5], infor.P[6], infor.P[7]], [infor.P[8], infor.P[9], infor.P[10], infor.P[11]]])
        proj = infor.trans_matrix
        ori_image = infor.cv2_img
        img_width=np.size(ori_image,1)
        img_height=np.size(ori_image,0)
        flag_other_image=0
        if(flag_other_image == 0):
            (ori_image_r,k)=self.image_rescale(ori_image,k,img_width,img_height,rescaling_factor)
            #(ori_image_r, k, width, height) = self.preliminary_rescaling(rescaling_factor, rescaling_factor, k, ori_image,width,height)
        else:
            (ori_image_r,k)=self.image_rescale_debugg(k,img_width,img_height)
            #(ori_image_r, k, width, height) = self.adjust_out_imgs(rescaling_factor, k, ori_image,width,height)
        print("Rescaling time"+str(timer()-start1))

        # Rescalling proj matrix
        proj[0][3] = proj[0][3]*self.rsf_factor
        proj[1][3] = proj[1][3]*self.rsf_factor
        proj[2][3] = proj[2][3]*self.rsf_factor
        Pro2 = np.dot(k, proj)

        #self.perspective_mapping(copy.deepcopy(Pro2),ori_image_r)
       
        #For interpolation
        x_int=[]
        y_int=[]
        value_int=[]
        
        start2=timer()
        obstacle_mask=self.free_obstacle_masking_v2(copy.deepcopy(k),copy.deepcopy(proj),ori_image_r)
        #obstacle_mask=self.free_obstacle_masking(copy.deepcopy(k),copy.deepcopy(proj),ori_image_r)
        print("Masking time "+str(timer()-start2))
        self.obstacle_mask=obstacle_mask
        start3=timer()
        (output_img,Mapx,Mapy,Mapped)=self.pixel_mapping_mode_e(Pro2,obstacle_mask)
        print("Mapping time "+str(timer()-start3))
        start4=timer()
        #(output_img,Mapx,Mapy,Mapped,Gmapped)=self.pixel_mapping_mode_b(Pro2,obstacle_mask)
        map_condition3=np.where(Mapped==1,True,False)
        #print("Mapping \n")
        #print(str(map_condition3.shape))

        Mapx = (Mapx+abs(self.Pbox_min_x1)-1).astype(int)
        Mapy = (Mapy+abs(self.Pbox_min_y1)-1).astype(int)
        
        Maplinear_out=Mapy*self.DISTXmax+Mapx
        Maplinear_out=Maplinear_out*3
        #print("proper dimensions")
        
        
        n=(self.height)*(self.width)
        Maplinear_in=np.arange(n)
        
        #print("Shape of indexing matrices")
        
        Maplinear_in=Maplinear_in[map_condition3]
        Maplinear_in1=Maplinear_in*3
        Maplinear_in2=Maplinear_in1+1
        Maplinear_in3=Maplinear_in2+1
        Maplinear_out1=Maplinear_out
        Maplinear_out2=Maplinear_out1+1
        Maplinear_out3=Maplinear_out2+1

        #print(str(Maplinear_out.shape))
        #print(str(Maplinear_in.shape))

        output_img.ravel()[Maplinear_out]=ori_image_r.ravel()[Maplinear_in]
        output_img.ravel()[Maplinear_out2]=ori_image_r.ravel()[Maplinear_in3]
        output_img.ravel()[Maplinear_out3]=ori_image_r.ravel()[Maplinear_in3]
        print("Indexing timer "+str(timer()-start4))
        #print(str(output_img.shape))
        #print(str(output_img))
       

        '''
        for u in range(self.width):
            for v in range(self.height):
                if(Mapped[v][u]):
                    #output_img[int(Mapy[v][u]) + abs(self.Pbox_min_y1)-1][int(Mapx[v][u]) + abs(self.Pbox_min_x1)-1] = (ori_image_r[v][u])#/Gmapped[Mapy[v][u]][Mapx[v][u]]
                    
                    output_img[Mapy[v][u]][Mapx[v][u]] = (ori_image_r[v][u])#/Gmapped[Mapy[v][u]][Mapx[v][u]]
                    
                    
                    #Acrescento da interpolacao de pixels, ira ser posteriormente alterado para suportar varias imagens de IPM
                    #Por agora sera aqui aplicado
                    x_int.append(int(Mapx[v][u]) + abs(self.Pbox_min_x1)-1)
                    y_int.append(int(Mapy[v][u]) + abs(self.Pbox_min_y1)-1)
                    value_int.append(ori_image_r[v][u])'''
        

        '''Mapx = Mapx+abs(self.Pbox_min_y1)-1
        Mapy = Mapy+abs(self.Pbox_min_y1)-1
        output_img[Mapy.astype(int)][Mapy.astype(int)] = ori_image_r
        '''
        #cv2.imwrite('/home/hellxberg/ws_thesis/src/ipm_perception/src/IPM_almost_multi_modal.jpeg', output_img)
        # Cyle for for interpolation
        
        
        #self.for_report(output_img)
        
        #print("Images Published")
        if(self.flag_debugging_mode==1):

            cv2.imwrite('/home/hellxberg/ws_thesis/src/ipm_perception/src/Pos-IPM_cropped.jpeg', output_img)
            #cv2.imwrite('/home/hellxberg/ws_thesis/src/ipm_perception/src/Pos-IPM_Full.jpeg', output_img2)
            #cv2.imwrite('/home/hellxberg/ws_thesis/src/ipm_perception/src/Pos-IPM_colour.jpeg', colour_img)
            cv2.imwrite('/home/hellxberg/ws_thesis/src/ipm_perception/src/Original_image.jpeg',ori_image)

        #cv2.imshow("Original IPM image",output_img)
        #cv2.waitKey(0)

        
        #interpolated_img=self.interpolation_first_phase(x_int,y_int,value_int)
        #cv2.imshow("meh",output_img)
        #cv2.waitKey(0)
        #cv2.imwrite('/home/hellxberg/ws_thesis/src/ipm_perception/src/IPM_interpolated.jpeg', interpolated_img)
        #self.perspective_visualization(interpolated_img,Pbox_min_x1,Pbox_max_x1,Pbox_min_y1,Pbox_max_y1,rsf_factor)
        #self.points_in_image_v2(interpolated_img,copy.deepcopy(k),copy.deepcopy(proj),ori_image_r)

        return (self.width,self.height)

    def calculate_new_XY(self, Pro, point_x, point_y):
        #Function that calculates the new pixels coordinates, the basis of the IPM process
        D = point_x*(Pro[1][0]*Pro[2][1]-Pro[1][1]*Pro[2][0])+point_y*(Pro[2][0] * Pro[0][1]-Pro[2][1]*Pro[0][0])+Pro[0][0]*Pro[1][1]-Pro[0][1]*Pro[1][0]
        X = (point_x*(Pro[1][1]*Pro[2][3]-Pro[1][3]*Pro[2][1])+point_y*(Pro[0][3]*Pro[2][1]-Pro[0][1]*Pro[2][3])-Pro[0][3]*Pro[1][1]+Pro[0][1]*Pro[1][3])/(D*1.0)
        Y = (point_x*(Pro[1][0]*Pro[2][3]-Pro[1][3]*Pro[2][0])+point_y*(Pro[0][3] * Pro[2][0]-Pro[0][0]*Pro[2][3])+Pro[0][0]*Pro[1][3]-Pro[0][3]*Pro[1][0])/(D*1.0)
        return int(X), int(Y)


    def perspective_mapping(self,the_matrix,img2over):
        u=[]
        v=[]
        width=self.width
        height=self.height

        #TO check the overlay of cloud over the image the image must be change. Since the y is inverse
        #It's only for debugging purposes
        '''canvas_fig=np.zeros((height,width, 3), np.uint8)
        for uu in range(width):
            for vv in range(height):'''

        plt.imshow(img2over,extent=[0, width, 0, height])

        #time.sleep(1000)
        coll_coord_x=self.coll_coord_x
        coll_coord_y=self.coll_coord_y
        image_scale=0
        for i in range(len(coll_coord_x)):
            vect_coord=np.array([[coll_coord_x[i]],[coll_coord_y[i]],[0],[1]])
            image_coord=np.dot(the_matrix,vect_coord)
            image_scale=image_coord[2]
            u.append(int(image_coord[0]))
            v.append(int(image_coord[1]))
            
        
        plt.plot(v,u,'r')
        plt.show()



    def pixel_mapping_mode_b(self,Pro2,obst_mask):

        #Function to create the mapes Mapx & Mapy
        #Pro2-Matrice to transform
        #box_limits1 & box_limits2-matrices containing the limits of the projection box
        #image_measures-matrice with image dimensions

        #self.obstacle_mask()
        #Define here the non debugging variables ------------------------------
        Mapx = np.zeros((self.height, self.width), dtype=int)
        Mapy = np.zeros((self.height, self.width), dtype=int)
        Mapped = np.zeros((self.height, self.width), dtype=int)
        output_img = np.zeros((self.DISTYmax,self.DISTXmax, 3), np.uint8)
        Gmapped = np.zeros((self.DISTYmax,self.DISTXmax), dtype=int)

        #if(flag_verbose):
            #print()


        start=timer()
        for u in range(self.width):
            for v in range(self.height):
                #if(obst_mask[v][u]==[255,255,255]).all():
                if(obst_mask[v][u]==255):
                    x, y =self.calculate_new_XY(Pro2, u, v)
                    if(((x > self.Pbox_min_x1) and (x < self.Pbox_max_x1)) and ((y > self.Pbox_min_y1) and (y <self.Pbox_max_y1))):
                        Mapx[v][u] = x
                        Mapy[v][u] = y
                        Mapped[v][u] = 1
                        Gmapped[Mapy[v][u]][Mapx[v][u]] += 1

        return output_img,Mapx,Mapy,Mapped,Gmapped 


    def pixel_mapping_mode_c(self,Pro2,obst_mask):

        #Function to create the mapes Mapx & Mapy
        #Pro2-Matrice to transform
        #box_limits1 & box_limits2-matrices containing the limits of the projection box
        #image_measures-matrice with image dimensions

        #self.obstacle_mask()
        #Define here the non debugging variables ------------------------------
        Mapx = np.zeros((self.height, self.width), dtype=int)
        Mapy = np.zeros((self.height, self.width), dtype=int)
        Mapped = np.zeros((self.height, self.width), dtype=int)
        output_img = np.zeros((self.DISTYmax,self.DISTXmax, 3), np.uint8)
        Gmapped = np.zeros((self.DISTYmax,self.DISTXmax), dtype=int)

        #if(flag_verbose):
            #print()


        start=timer()
        for u in range(self.width):
            for v in range(self.height):
                #if(obst_mask[v][u]==[255,255,255]).all():
                if(obst_mask[v][u]==255):
                    x, y =self.calculate_new_XY(Pro2, u, v)
                    if(((x > self.Pbox_min_x1) and (x < self.Pbox_max_x1)) and ((y > self.Pbox_min_y1) and (y <self.Pbox_max_y1))):
                        Mapx[v][u] = x
                        Mapy[v][u] = y
                        Mapped[v][u] = 1
                        Gmapped[Mapy[v][u]][Mapx[v][u]] += 1

        return output_img,Mapx,Mapy,Mapped,Gmapped
    
    def pixel_mapping_mode_d(self,Pro2):

        #Function to create the mapes Mapx & Mapy
        #Pro2-Matrice to transform
        #box_limits1 & box_limits2-matrices containing the limits of the projection box
        #image_measures-matrice with image dimensions
        start=timer()

        
        #self.obstacle_mask()
        #Define here the non debugging variables ------------------------------
        #Mapx = np.zeros((self.height, self.width), dtype=int)
        #Mapy = np.zeros((self.height, self.width), dtype=int)
        #Mapped = np.zeros((self.height, self.width), dtype=int)
        output_img = np.zeros((self.DISTYmax,self.DISTXmax, 3), np.uint8)
        Gmapped = np.zeros((self.DISTYmax,self.DISTXmax), dtype=int)

        trans_vector=np.array([[-Pro2[0][3]],[-Pro2[1][3]],[-Pro2[2][3]],[0]])

        xx,yy=np.meshgrid(np.arange(self.width),np.arange(self.height))
        x=xx.ravel()
        y=yy.ravel()
        #Defining the most important matrices
        p11=Pro2[0][0]
        p12=Pro2[0][1]
        p13=Pro2[0][2]
        p21=Pro2[1][0]
        p22=Pro2[1][1]
        p23=Pro2[1][2]
        p31=Pro2[2][0]
        p32=Pro2[2][1]
        p33=Pro2[2][2]
        #Creating (in a hardcoded manner )the inverse matrix
        rotation_inv=np.array([[(p22 - p32*y)/(p11*p22 - p12*p21 + p21*p32*x - p22*p31*x - p11*p32*y + p12*p31*y),-(p12 - p32*x)/(p11*p22 - p12*p21 + p21*p32*x - p22*p31*x - p11*p32*y + p12*p31*y),-(p22*x - p12*y)/(p11*p22 - p12*p21 + p21*p32*x - p22*p31*x - p11*p32*y + p12*p31*y),(p12*p23 - p13*p22 + p22*p33*x - p23*p32*x - p12*p33*y + p13*p32*y)/(p11*p22 - p12*p21 + p21*p32*x - p22*p31*x - p11*p32*y + p12*p31*y)],[-(p21 - p31*y)/(p11*p22 - p12*p21 + p21*p32*x - p22*p31*x - p11*p32*y + p12*p31*y),(p11 - p31*x)/(p11*p22 - p12*p21 + p21*p32*x - p22*p31*x - p11*p32*y + p12*p31*y),(p21*x - p11*y)/(p11*p22 - p12*p21 + p21*p32*x - p22*p31*x - p11*p32*y + p12*p31*y),-(p11*p23 - p13*p21 + p21*p33*x - p23*p31*x - p11*p33*y + p13*p31*y)/(p11*p22 - p12*p21 + p21*p32*x - p22*p31*x - p11*p32*y + p12*p31*y)],[0,0,0,1],[ -(p21*p32 - p22*p31)/(p11*p22 - p12*p21 + p21*p32*x - p22*p31*x - p11*p32*y + p12*p31*y), (p11*p32 - p12*p31)/(p11*p22 - p12*p21 + p21*p32*x - p22*p31*x - p11*p32*y + p12*p31*y), -(p11*p22 - p12*p21)/(p11*p22 - p12*p21 + p21*p32*x - p22*p31*x - p11*p32*y + p12*p31*y), (p11*p22*p33 - p11*p23*p32 - p12*p21*p33 + p12*p23*p31 + p13*p21*p32 - p13*p22*p31)/(p11*p22 - p12*p21 + p21*p32*x - p22*p31*x - p11*p32*y + p12*p31*y)]])

        result=np.dot(rotation_inv,trans_vector)
        D_result=result[3,0]
        x_result=(result[0,0]/D_result)
        y_result=(result[1,0]/D_result)
        print("X result")
        print(str(x_result))
        mapp=x_result/x_result
        mapping_process=np.where(((x_result > self.Pbox_min_x1) & (x_result < self.Pbox_max_x1)) & ((y_result > self.Pbox_min_y1) & (y_result <self.Pbox_max_y1)),(x_result,y_result,mapp),False)

        Mapx=mapping_process[0].reshape(self.height,self.width)
        Mapy=mapping_process[1].reshape(self.height,self.width)
        Mapped=mapping_process[2].reshape(self.height,self.width)

        print("Mapx\n")
        print(str(Mapx))
        print("Mapy\n")
        print(str(Mapy))
        print("Mapped\n")
        print(str(Mapped))

        return output_img,Mapx,Mapy,Mapped

    def pixel_mapping_mode_e(self,Pro2,obst_mask):

        #Function to create the mapes Mapx & Mapy
        #Pro2-Matrice to transform
        #box_limits1 & box_limits2-matrices containing the limits of the projection box
        #image_measures-matrice with image dimensions
        #start=timer()

        
        #self.obstacle_mask()
        #Define here the non debugging variables ------------------------------
        #Mapx = np.zeros((self.height, self.width), dtype=int)
        #Mapy = np.zeros((self.height, self.width), dtype=int)
        #Mapped = np.zeros((self.height, self.width), dtype=int)
        output_img = np.zeros((self.DISTYmax,self.DISTXmax, 3), np.uint8)
        Gmapped = np.zeros((self.DISTYmax,self.DISTXmax), dtype=int)

        trans_vector=np.array([[-Pro2[0][3]],[-Pro2[1][3]],[-Pro2[2][3]],[0]])

        xx,yy=np.meshgrid(np.arange(self.width),np.arange(self.height))
        point_x=xx.ravel()
        point_y=yy.ravel()
        #Defining the most important matrices
        p11=Pro2[0][0]
        p12=Pro2[0][1]
        p13=Pro2[0][2]
        p21=Pro2[1][0]
        p22=Pro2[1][1]
        p23=Pro2[1][2]
        p31=Pro2[2][0]
        p32=Pro2[2][1]
        p33=Pro2[2][2]
        Pro=Pro2

        obst_mask=obst_mask.ravel()
        #Conditions
        mapp_condi1=np.array(obst_mask==255)

        '''first_mapping=np.vstack([point_x, point_y])
        second_mapping=first_mapping[:,mapp_condi1]
        point_x=second_mapping[0,:]
        point_y=second_mapping[1,:]'''
        first_mapping=mapping_process=np.where(mapp_condi1,(point_x,point_y),False)
        point_x=first_mapping[0]
        point_y=first_mapping[1]

        D = point_x*(Pro[1][0]*Pro[2][1]-Pro[1][1]*Pro[2][0])+point_y*(Pro[2][0] * Pro[0][1]-Pro[2][1]*Pro[0][0])+Pro[0][0]*Pro[1][1]-Pro[0][1]*Pro[1][0]
        X = (point_x*(Pro[1][1]*Pro[2][3]-Pro[1][3]*Pro[2][1])+point_y*(Pro[0][3]*Pro[2][1]-Pro[0][1]*Pro[2][3])-Pro[0][3]*Pro[1][1]+Pro[0][1]*Pro[1][3])/(D*1.0)
        Y = (point_x*(Pro[1][0]*Pro[2][3]-Pro[1][3]*Pro[2][0])+point_y*(Pro[0][3] * Pro[2][0]-Pro[0][0]*Pro[2][3])+Pro[0][0]*Pro[1][3]-Pro[0][3]*Pro[1][0])/(D*1.0)

        '''
        mapp_condi2=np.array(((X > self.Pbox_min_x1) & (X < self.Pbox_max_x1)) & ((Y > self.Pbox_min_y1) & (Y <self.Pbox_max_y1)))
        third_mapping=second_mapping[:,mapp_condi2]
        Mapx=third_mapping[0,:]
        Mapy=third_mapping[1,:]
        '''
        mapp=X/X
        map_condition2=np.array(((X > self.Pbox_min_x1) & (X < self.Pbox_max_x1)) & ((Y > self.Pbox_min_y1) & (Y <self.Pbox_max_y1)))
        #mapping_process=np.where(map_condition2,(X,Y,mapp),False)
        Mapped=np.where(map_condition2,mapp,False)
        '''Mapx=mapping_process[0].reshape(self.height,self.width)
        Mapy=mapping_process[1].reshape(self.height,self.width)
        Mapped=mapping_process[2].reshape(self.height,self.width)'''
        '''Mapx=mapping_process[0,:]
        Mapy=mapping_process[1,:]
        Mapped=mapping_process[2,:]'''
        Mapx=X[map_condition2]
        Mapy=Y[map_condition2]
        #print("Time of mapping process\n")
        #print(str(timer()-start))

        return output_img,Mapx,Mapy,Mapped


    def pixel_mapping_original(self,Pro2):
        #Function to create the mapes Mapx & Mapy
        #Pro2-Matrice to transform
        #box_limits1 & box_limits2-matrices containing the limits of the projection box
        #image_measures-matrice with image dimensions

        #self.obstacle_mask()
        #Define here the non debugging variables ------------------------------
        Mapx = np.zeros((self.height, self.width), dtype=int)
        Mapy = np.zeros((self.height, self.width), dtype=int)
        Mapped = np.zeros((self.height, self.width), dtype=int)
        output_img = np.zeros((self.DISTYmax,self.DISTXmax, 3), np.uint8)
        Gmapped = np.zeros((self.DISTYmax,self.DISTXmax), dtype=int)

        start=timer()
        for u in range(self.width):
            for v in range(self.height):
                
                x, y =self.calculate_new_XY(Pro2, u, v)
                if(((x > self.Pbox_min_x1) and (x < self.Pbox_max_x1)) and ((y > self.Pbox_min_y1) and (y <self.Pbox_max_y1))):
                    Mapx[v][u] = x
                    Mapy[v][u] = y
                    Mapped[v][u] = 1
                    Gmapped[Mapy[v][u]][Mapx[v][u]] += 1
        return output_img,Mapx,Mapy,Mapped,Gmapped 


    def pixel_mapping_mode_a(self,Pro2):
        #Function to create the mapes Mapx & Mapy
        #Pro2-Matrice to transform
        #box_limits1 & box_limits2-matrices containing the limits of the projection box
        #image_measures-matrice with image dimensions

        start = timer()
        #self.obstacle_mask()
        #Define here the non debugging variables ------------------------------
        Mapx = np.zeros((self.height, self.width), dtype=int)
        Mapy = np.zeros((self.height, self.width), dtype=int)
        Mapped = np.zeros((self.height, self.width), dtype=int)
        output_img = np.zeros((self.DISTYmax,self.DISTXmax, 3), np.uint8)
        Gmapped = np.zeros((self.DISTYmax,self.DISTXmax), dtype=int)
        start=timer()
        for u in range(self.width):
            for v in range(self.height):
                x, y =self.calculate_new_XY(Pro2, u, v)
                point=Point(x,-y)
                if(self.final_poly.contains(point)):
                    if(((x > self.Pbox_min_x1) and (x < self.Pbox_max_x1)) and ((y > self.Pbox_min_y1) and (y <self.Pbox_max_y1))):
                        Mapx[v][u] = x
                        Mapy[v][u] = y
                        Mapped[v][u] = 1
                        Gmapped[Mapy[v][u]][Mapx[v][u]] += 1

        return output_img,Mapx,Mapy,Mapped,Gmapped  


    def points_in_image_v2(self,interpolated_img,intri_matrix,extrin_matrix,original_img):
        #Function for visualization of the cloud points over the ipm image created
        #functions which puts existing coordinates over ipm image
        init_coords=self.initial_coll_coords
        height=self.height
        width=self.width
        coll_coord_x=self.coll_coord_x
        coll_coord_y=self.coll_coord_y
        '''x_init1=[p[0] for p in init_coords[0]]
        y_init1=[p[1] for p in init_coords[0]]
        x_init2=[p[0] for p in init_coords[1]]
        y_init2=[p[1] for p in init_coords[1]]
        
        plt.plot(x_init1,y_init1,'b-')
        plt.plot(x_init2,y_init2,'r-')
        plt.show()'''

        #self.imshow1_cloudpoints_image(interpolated_img,init_coords)
        #self.imshow1_cloud_indiv_polygon_image(interpolated_img,init_coords)
        #(x_init,y_init)=self.imshow1_cloud_joint_polygon_image(interpolated_img,coll_coord_x,coll_coord_y)
        #self.imshow_polygon_original_image(transf_matrix,coll_coord_x,coll_coord_y,x_init,y_init,original_img,interpolated_img)
        #self.imshow_polygon_original_image_correct(intri_matrix,extrin_matrix,coll_coord_x,coll_coord_y,x_init,y_init,original_img,interpolated_img)
        #raw_input()
        

    def points_in_image(self,interpolated_img):
        #Function for visualization of the cloud points over the ipm image created
        #functions which puts existing coordinates over ipm image
        
        plt.imshow(interpolated_img,extent=[self.Pbox_min_x1, self.Pbox_max_x1, self.Pbox_min_y1, self.Pbox_max_y1])
        
        #TODO mudar depois para self.vertices_x 
        vertices_x=self.coll_coord_x
        vertices_y=self.coll_coord_y
    
        vertices_rsf_x = [int(x) for x in vertices_x]
        vertices_rsf_y = [int(y) for y in vertices_y]
        #plt.plot(vertices_rsf_x,vertices_rsf_y,'b')
        #plt.show()
        x_cloud_v1=[]
        y_cloud_v1=[]
        for i in range(len(vertices_rsf_x)):
            #if( (vertices_rsf_x[i] < self.Pbox_max_x1 and vertices_rsf_x[i] > self.Pbox_min_x1) and (vertices_rsf_y[i] < self.Pbox_max_y1 and vertices_rsf_y[i] > self.Pbox_min_y1) ):
            x_cloud_v1.append(vertices_rsf_x[i])
            y_cloud_v1.append(vertices_rsf_y[i])
        plt.plot(x_cloud_v1,y_cloud_v1,'o-')
        plt.show()

    def interpolation_first_phase(self,x_int,y_int,value_int):
        #Function to interpolate from given values
        # target grid to interpolate to
        xi = np.arange(0,self.DISTXmax,1)
        yi = np.arange(0,self.DISTYmax,1)
        output_img = np.zeros((self.DISTYmax, self.DISTXmax, 3), np.uint8)
        for i in range(len(y_int)):
            output_img[y_int[i]][x_int[i]]=value_int[i]

        xi,yi = np.meshgrid(xi,yi)
        # interpolate
        zi = griddata((x_int,y_int),value_int,(xi,yi),method='linear')
        for y in range(self.DISTYmax):
            for x in range(self.DISTXmax):
                output_img[y][x]=zi[y][x]
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


    def imshow1_cloudpoints_image(self,interpolated_img,init_coords):
        fig1=plt.figure(1)
        plt.imshow(interpolated_img,extent=[self.Pbox_min_x1, self.Pbox_max_x1, self.Pbox_min_y1, self.Pbox_max_y1])
        #It's prepared for only 2 clous points
        my_color=['b','r']
        i=0
        my_labels=['Points of left laser','Points of right laser']
        for cloud in init_coords:
            x_init=[]
            y_init=[]
            for my_point in cloud:
                x_init.append(my_point[0])
                y_init.append(my_point[1])
            plt.scatter(x_init,y_init,marker='o',c=my_color[i],label=my_labels[i])
        
            i+=1
        plt.title('Overlay of each cloud points over the IPM image')
        plt.legend()
        fig1.show()
    
    def imshow1_cloud_indiv_polygon_image(self,interpolated_img,init_coords):
        fig2=plt.figure(2)
        plt.imshow(interpolated_img,extent=[self.Pbox_min_x1, self.Pbox_max_x1, self.Pbox_min_y1, self.Pbox_max_y1])
        my_color=['b','r']
        j=0
        my_labels=['Points of left laser','Points of right laser']
        for cloud in init_coords:
            x_init=[]
            y_init=[]
            for i in range(len(cloud)):
                x_init.append(cloud[i][0])
                y_init.append(cloud[i][1])
            x_init.append(cloud[0][0])
            y_init.append(cloud[0][1])
            plt.plot(x_init,y_init,marker='o',c=my_color[j],label=my_labels[j])
            j+=1

        plt.title('Overlay of each cloud points over the IPM image')
        plt.legend()
        fig2.show()

    def imshow1_cloud_joint_polygon_image(self,interpolated_img,coll_coord_x,coll_coord_y):
        fig3=plt.figure(3)
        plt.imshow(interpolated_img,extent=[self.Pbox_min_x1, self.Pbox_max_x1, self.Pbox_min_y1, self.Pbox_max_y1])
        x_init=[]
        y_init=[]
        for i in range(-1,len(coll_coord_x)):
            x_init.append(coll_coord_x[i])
            y_init.append(coll_coord_y[i])
        plt.plot(x_init,y_init,marker='o',c='r')        
        plt.legend()
        fig3.show()
        return (x_init,y_init)


    def imshow_polygon_original_image(self,transf_matrix,coll_coord_x,coll_coord_y,x_init,y_init,original_img,interpolated_img):
        fig4=plt.figure(4)
        plt.subplot(131)
        plt.imshow(original_img,extent=[0, self.width, 0, self.height])
        #time.sleep(1000)
        u=[]
        v=[]
        image_scale=0
        for i in range(len(coll_coord_x)):
            vect_coord=np.array([[coll_coord_x[i]],[-coll_coord_y[i]],[0],[1]])
            image_coord=np.dot(transf_matrix,vect_coord)
            image_scale=image_coord[2]
            u.append(int(image_coord[0]/image_scale))
            v.append(int(image_coord[1]/image_scale))
        plt.plot(u,v,'r')
        plt.title('Polygon of cloud points created from the perspective mapping ')
        plt.subplot(133)
        plt.imshow(interpolated_img,extent=[self.Pbox_min_x1, self.Pbox_max_x1, self.Pbox_min_y1, self.Pbox_max_y1])
        plt.plot(x_init,y_init,marker='o',c='r')
        plt.title('Overlay of each cloud points over the IPM image')
        fig4.show()
        fig5=plt.figure(5)
        plt.imshow(original_img,origin='upper',extent=[0, self.width, 0, self.height])
        plt.plot(u,v,'r')
        plt.title('Polygon of cloud points created from the perspective mapping ')
        fig5.show()


    def imshow_polygon_original_image_correct(self,intrin_matrix,extrin_matrix,coll_coord_x,coll_coord_y,x_init,y_init,original_img,interpolated_img):
        #Function to do exacly what is 
        fig6=plt.figure(6)
        plt.gca().invert_yaxis()
        new_width=np.size(original_img,1)
        new_height=np.size(original_img,0)
        plt.imshow(original_img)
        u=[]
        v=[]
        
        free_obst_coord=[]
        image_scale=0
        for i in range(len(coll_coord_x)):
            a_pixel=[]
            vect_coord=np.array([[coll_coord_x[i]],[coll_coord_y[i]],[0],[1]])
            pt_camera3D=np.dot(extrin_matrix,vect_coord)
            if(pt_camera3D[2]>0):
                image_coord=np.dot(intrin_matrix,pt_camera3D)
                image_scale=image_coord[2]
                xpixel=int(image_coord[0]/image_scale)
                ypixel=int(image_coord[1]/image_scale)
                a_pixel.append(xpixel)
                a_pixel.append(ypixel)
                free_obst_coord.append(a_pixel)
                u.append(xpixel)
                v.append(ypixel)

        Poly1=Polygon(free_obst_coord)
        #Next section to make polygon of image
        image_coord=[[0,0],[self.width-1,0],[self.width-1,self.height-1],[0,self.height-1]]
        Poly2=Polygon(image_coord)
        multi_poly=[]
        if(Poly2.intersects(Poly1)):
            
            Poly1 = Poly1.buffer(0)
            intersection=Poly2.intersection(Poly1)
            nonoverlap=Poly2.difference(intersection)
            multi_poly.append(nonoverlap)
        else:
            multi_poly.append(nonoverlap)
        
        #final_poly=MultiPolygon(multi_poly)
        plt.plot(u,v,'r')
        fig6.show()

        fig7=plt.figure(7)
        plt.gca().invert_yaxis()
        plt.imshow(original_img)
        #Dividing Polygon in set of coordinates
        multi_coords=[]
        for a_poly in multi_poly:
            x, y = a_poly.exterior.coords.xy
            plt.plot(x,y,'r')
            multi_coords.append([x,y])

        fig7.show()

    def free_obstacle_masking(self,intrinsic_matrix,extrinsic_matrix,original_img):
        #Function to mask the obstacles in the original image
        #Initialization of important variables
        
        final_mask=np.zeros((self.height,self.width),dtype=np.uint8)
        useless_mask=np.zeros((self.height+2,self.width+2),dtype=np.uint8)
        coll_coord_x=self.coll_coord_x
        coll_coord_y=self.coll_coord_y
        
        free_obst_coord=[]
        image_scale=0
        for i in range(len(coll_coord_x)):
            a_pixel=[]
            vect_coord=np.array([[coll_coord_x[i]],[coll_coord_y[i]],[0],[1]])
            pt_camera3D=np.dot(extrinsic_matrix,vect_coord)
            if(pt_camera3D[2]>0):
                image_coord=np.dot(intrinsic_matrix,pt_camera3D)
                image_scale=image_coord[2]
                xpixel=int(image_coord[0]/image_scale)
                ypixel=int(image_coord[1]/image_scale)
                a_pixel.append(xpixel)
                a_pixel.append(ypixel)
                free_obst_coord.append(a_pixel)

        Poly1=Polygon(free_obst_coord)
        #Next section to make polygon of image
        image_coord=[[0,0],[self.width-1,0],[self.width-1,self.height-1],[0,self.height-1]]

        Poly2=Polygon(image_coord)
        multi_poly=[]
        if(Poly2.intersects(Poly1)):
            Poly1 = Poly1.buffer(0)
            intersection=Poly2.intersection(Poly1)
            use_poly=intersection
            if use_poly.geom_type == 'MultiPolygon':
                multi_poly=list(use_poly)
            elif use_poly.geom_type == 'Polygon':
                multi_poly.append(use_poly)
        else:
            multi_poly.append(Poly2)
        
        #Dividing Polygon in set of coordinates
        
        for a_poly in multi_poly:
            my_mask=np.zeros((self.height,self.width), dtype=np.uint8)
            
            x, y = a_poly.exterior.coords.xy
            for i in range(len(x)):
                #cv2.line(my_mask,(int(x[i-1]),int(y[i-1])),(int(x[i]),int(y[i])),[255,255,255],1)
                cv2.line(my_mask,(int(x[i-1]),int(y[i-1])),(int(x[i]),int(y[i])),255,1)
            sum_x=0
            sum_y=0
            total_n=0
            for x_pixel in range(self.width):
                for y_pixel in range(self.height):
                    if(my_mask[y_pixel][x_pixel]==[255,255,255]).all():
                        sum_x=sum_x+x_pixel
                        sum_y=sum_y+y_pixel
                        total_n=total_n+1
            x_cent=int(sum_x/float(total_n))
            y_cent=int(sum_y/float(total_n))
            cv2.floodFill(my_mask,useless_mask,(x_cent,y_cent),255)
            final_mask=final_mask+my_mask
        #cv2.imshow("final_mask",final_mask)
        #cv2.waitKey(0)
        return final_mask
            
            
    def free_obstacle_masking_v2(self,intrinsic_matrix,extrinsic_matrix,original_img):
        #Function to mask the obstacles in the original image
        #Initialization of important variables
        print("masking_debugging \n")
        start=timer()
        final_mask=np.zeros((self.height,self.width),dtype=np.uint8)
        useless_mask=np.zeros((self.height+2,self.width+2),dtype=np.uint8)
        coll_coord_x=self.coll_coord_x
        coll_coord_y=self.coll_coord_y
        
        free_obst_coord=[]
        image_scale=0
        for i in range(len(coll_coord_x)):
            a_pixel=[]
            vect_coord=np.array([[coll_coord_x[i]],[coll_coord_y[i]],[0],[1]])
            pt_camera3D=np.dot(extrinsic_matrix,vect_coord)
            if(pt_camera3D[2]>0):
                image_coord=np.dot(intrinsic_matrix,pt_camera3D)
                image_scale=image_coord[2]
                xpixel=int(image_coord[0]/image_scale)
                ypixel=int(image_coord[1]/image_scale)
                a_pixel.append(xpixel)
                a_pixel.append(ypixel)
                free_obst_coord.append(a_pixel)

        print("Conversion time "+str(timer()-start))
        start=timer()
        Poly1=Polygon(free_obst_coord)
        #Next section to make polygon of image
        image_coord=[[0,0],[self.width-1,0],[self.width-1,self.height-1],[0,self.height-1]]

        Poly2=Polygon(image_coord)
        multi_poly=[]
        if(Poly2.intersects(Poly1)):
            Poly1 = Poly1.buffer(0)
            intersection=Poly2.intersection(Poly1)
            use_poly=intersection
            if use_poly.geom_type == 'MultiPolygon':
                multi_poly=list(use_poly)
            elif use_poly.geom_type == 'Polygon':
                multi_poly.append(use_poly)
        else:
            multi_poly.append(Poly2)
        
        #Dividing Polygon in set of coordinates
        
        for a_poly in multi_poly:
            my_mask=np.zeros((self.height,self.width), dtype=np.uint8)
            
            x, y = a_poly.exterior.coords.xy
            for i in range(len(x)):
                #cv2.line(my_mask,(int(x[i-1]),int(y[i-1])),(int(x[i]),int(y[i])),[255,255,255],1)
                cv2.line(my_mask,(int(x[i-1]),int(y[i-1])),(int(x[i]),int(y[i])),255,1)
            sum_x=0
            sum_y=0
            total_n=0
            for x_pixel in range(self.width):
                for y_pixel in range(self.height):
                    if(my_mask[y_pixel][x_pixel]==[255,255,255]).all():
                        sum_x=sum_x+x_pixel
                        sum_y=sum_y+y_pixel
                        total_n=total_n+1
            x_cent=int(sum_x/float(total_n))
            y_cent=int(sum_y/float(total_n))
            cv2.floodFill(my_mask,useless_mask,(x_cent,y_cent),255)
            final_mask=final_mask+my_mask
        #cv2.imshow("final_mask",final_mask)
        #cv2.waitKey(0)
        print("Drawing time "+str(timer()-start))
        return final_mask
            





    











