#!/usr/bin/env python
import message_filters as mf
import laser_geometry.laser_geometry as lg
import sensor_msgs.point_cloud2 as pc2
import rospy
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
import tf
import geometry_msgs.msg
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
from Basic_ipm_processing import ipm_processes
from sensor_msgs.msg import LaserScan
import math
import matplotlib.pyplot as plt 
import os
import time
from timeit import default_timer as timer
from cloud_points_processing import cloud_points_processing



class subs_every:
    height = 0
    width = 0
    P = 0
    flag_synch = 0
    flag_laser_left=0
    flag_laser_right=0
    flag_synch_debug=0
    global ipm_x
    global cloud_x

    def __init__(self):
        self.bridge = CvBridge()
        # Initialization of subscritptions
        self.image_sub = mf.Subscriber("/frontal_camera/image_color", Image)
        self.image_info_sub = mf.Subscriber("/frontal_camera/camera_info", CameraInfo)
        self.left_laser=mf.Subscriber("/frontal_laser_left/laserscan",LaserScan)
        self.right_laser=mf.Subscriber("/frontal_laser_right/laserscan",LaserScan)
        #Do the callbacks
        ts = mf.TimeSynchronizer([self.image_sub, self.image_info_sub], 10)
        ts1=mf.ApproximateTimeSynchronizer([self.left_laser,self.right_laser], queue_size=5, slop=0.1)
        ts1.registerCallback(self.callback1)
        ts.registerCallback(self.callback)


    def callback1(self, left_laser, right_laser):
        left_laser_frame="frontal_laser_left"
        right_laser_frame="frontal_laser_right"
        road_frame="road"

        left_tf=self.listening_TF(road_frame,left_laser_frame)
        right_tf=self.listening_TF(road_frame,right_laser_frame)

        if((left_tf is not None) and (right_tf is not None)):
            cloud_x.main_processing_unit(left_laser,right_laser,left_tf,right_tf)
            #pm_x.set_coll_param(cloud_x.poly_coord_x,cloud_x.poly_coord_y)
            ipm_x.set_initial_coll_coords(cloud_x.initial_cloud_coords)
            ipm_x.set_full_polygon_coords(cloud_x.poly_coord_x,cloud_x.poly_coord_y)
            ipm_x.set_polygon_final(cloud_x.final_poly)
            #ipm_x.set_coll_coordinates(cloud_x.vertices_x,cloud_x.vertices_y)
            self.flag_synch_debug=1


    def information_organization(self,laser):
        print(str(laser))
        step=laser.angle_increment
        min_angle=laser.angle_min
        sum_step=0
        x=[]
        y=[]
        for i in laser.ranges:
            an_angle=min_angle+sum_step
            x.append(i*math.cos(an_angle))
            y.append(i*math.sin(an_angle))
            sum_step+=step
        plt.scatter(x,y)
        plt.show(block=True)




    def callback(self, some_image, some_image_info):
        # Rececao de informacao pertinente, no entanto tudo sera processado apenas a rececao de tf
        final_frame="frontal_camera_optical"
        initial_frame="road"
        self.trans_matrix = self.listening_TF(final_frame,initial_frame)
        if(self.trans_matrix is not None):
            self.P = some_image_info.P
            # self.P=some_image_info.K
            try:
                # Converting your ROS Image message to OpenCv2
                self.cv2_img = self.bridge.imgmsg_to_cv2(some_image, "bgr8")
                self.flag_synch = 1
            except CvBridgeError as e:
                print(e)
            else:
                rospy.sleep(0.01)
            


    def listening_TF(self,final_frame,initial_frame):
        listener = tf.TransformListener()
        while not rospy.is_shutdown():
            try:
                now = rospy.Time.now()
                listener.waitForTransform(final_frame, initial_frame, now, rospy.Duration(1.0))
                (trans, rot) = listener.lookupTransform(final_frame, initial_frame, now)
                for i in range(len(trans)):
                    trans[i] = trans[i]*1000

                transf_matrix = listener.fromTranslationRotation(trans, rot)
                break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue
            break
        return transf_matrix


def main():
    #Define the most importante variables
    rsf_scale=10 #In percentage
    rsf_factor = (rsf_scale/100.0)
    global ipm_x
    global cloud_x
    cloud_x=cloud_points_processing(rsf_factor)
    ipm_x= ipm_processes(rsf_factor)
    
    # Initialize important nodes
    img_pub = rospy.Publisher('Full_IPM', Image, queue_size=10)
    rospy.init_node('receive_node')
    info_c = subs_every()
    
    flag_publish = 0
    time1 = rospy.Time.now()
    flag_first_time = 0

    while not rospy.is_shutdown():
        #print(str(info_c.flag_synch))
        if(flag_first_time == 0 and info_c.flag_synch == 1 and info_c.flag_synch_debug==1):
            
            flag_first_time = 1
            print("Begginning of ipm  processing")
            time_file=open('/home/hellxberg/ws_thesis/src/ipm_perception/src/multimodal_IPM_mode_a_no_inter.txt','w')
            #for i in range(5,101,5):

            my_scale=100 #percentage
            rescaling_factor=(1/(my_scale/100.0))
            print("Begin")
            
            start=timer()
            (width,height)=ipm_x.IPM(info_c,rescaling_factor)
            print("Total time "+str(timer()-start))
            #time_file.write(str(my_scale)+" ,"+str(width)+" ,"+str(height)+" ,"+str(time_took)+"\n")
            #print("End of cycle")
            
            #time_file.close()
            #IPM_img = info_c.bridge.cv2_to_imgmsg(IPM_img,"bgr8")
            flag_publish = 1
            exit(0)
        if(flag_publish==1):
            meme_machine=1
            #img_pub.publish(IPM_img)
        rospy.sleep(0.1)


if __name__ == '__main__':
    main()
