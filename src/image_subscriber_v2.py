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

# Instantiate bridge

Pbox_min_x1 = 0
Pbox_min_y1 = 0
Pbox_max_x1 = 0
Pbox_max_y1 = 0


class subs_every:
    height = 0
    width = 0
    P = 0
    flag_synch = 0
    flag_laser_left=0
    flag_laser_right=0


    def __init__(self):
        self.bridge = CvBridge()
        # Initialization of class
        self.image_sub = mf.Subscriber("/frontal_camera/image_color", Image)
        self.image_info_sub = mf.Subscriber("/frontal_camera/camera_info", CameraInfo)
        self.left_laser=mf.Subscriber("/frontal_laser_left/laserscan",LaserScan)
        self.right_laser=mf.Subscriber("/frontal_laser_right/laserscan",LaserScan)
        
        ts = mf.TimeSynchronizer([self.image_sub, self.image_info_sub], 10)
        #ts1=mf.TimeSynchronizer([self.left_laser,self.right_laser],10)
        ts1=mf.ApproximateTimeSynchronizer([self.left_laser,self.right_laser], queue_size=5, slop=0.1)
        ts1.registerCallback(self.callback1)
        ts.registerCallback(self.callback)


    def callback1(self, left_laser, right_laser):
        print(str(right_laser))
        print("meh")



    def process_scan_left(self,left_laser):
        if(self.flag_laser_left==0):
            self.flag_laser_left=1
            lp = lg.LaserProjection()
            pc2_msg=lp.projectLaser(left_laser)
            point_generator=pc2.read_points(pc2_msg)
            x=[]
            y=[]
            for point in point_generator:
                if not math.isnan(point[2]):
                    x.append(point[0])
                    y.append(point[1])
            plt.scatter(x,y)
            plt.show(block=True)
            print(str(point_generator))
            #self.information_organization(left_laser)





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
        self.trans_matrix = self.listening_TF()
        # Caso a tf tenha informacao
        #print(str(self.trans_matrix))
        if(self.trans_matrix is not None):
            self.P = some_image_info.P
            # self.P=some_image_info.K
            # Another

            try:
                # Converting your ROS Image message to OpenCv2
                self.cv2_img = self.bridge.imgmsg_to_cv2(some_image, "bgr8")
                self.flag_synch = 1
                #print("Yep")
            except CvBridgeError as e:
                print(e)
            else:
                rospy.sleep(0.01)
            


    def listening_TF(self):
        listener = tf.TransformListener()
        #listener.waitForTransform("frontal_camera_optical","road", rospy.Time(), rospy.Duration(4.0))
        while not rospy.is_shutdown():
            try:
                now = rospy.Time.now()
                listener.waitForTransform("frontal_camera_optical", "road", now, rospy.Duration(1.0))
                (trans, rot) = listener.lookupTransform("frontal_camera_optical", "road", now)
                for i in range(len(trans)):
                    trans[i] = trans[i]*1000

                transf_matrix = listener.fromTranslationRotation(trans, rot)
                break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue
            break
        return transf_matrix


def main():
    # Initialize important nodes
    img_pub = rospy.Publisher('Full_IPM', Image, queue_size=10)
    rospy.init_node('receive_node')

    info_c = subs_every()
    ipm_x = ipm_processes()
    flag_publish = 0
    time1 = rospy.Time.now()
    flag_first_time = 0

    while not rospy.is_shutdown():
        #print(str(info_c.flag_synch))
        if(flag_first_time == 0 and info_c.flag_synch == 1):
            print("Passa aqui")
            flag_first_time = 1
            IPM_img=ipm_x.IPM(info_c)
            IPM_img = info_c.bridge.cv2_to_imgmsg(IPM_img,"bgr8")
            flag_publish = 1
            #exit(0)
        if(flag_publish==1):
            img_pub.publish(IPM_img)
        rospy.sleep(0.1)


if __name__ == '__main__':
    main()
