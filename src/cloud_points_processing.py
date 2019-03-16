import matplotlib.pyplot as plt
import laser_geometry.laser_geometry as lg
import sensor_msgs.point_cloud2 as pc2
import numpy as np 
import math



class cloud_points_processing:
    car_hitbox_max_x=0.1*1000
    car_hitbox_min_x=-3.0*1000
    car_hitbox_max_y=0.8*1000
    car_hitbox_min_y=-0.8*1000


    def polygon_creation(self,x,y):
        plt.scatter(x,y)
        plt.show()


    def cloud2cartesian(self,lasers,tfss):
        #Change the transforma matrices
        #Especifically put z=0, and accept only rotations in z axes
        '''left_tf=self.adapt_tf_situation1(left_tf)
        right_tf=self.adapt_tf_situation1(right_tf)
        left_points=self.generate_cartesian_points(left_laser,left_tf)'''
        coord_laser=[]
        print("list of lasers,what's wrong")
        print(str(len(lasers)))
        for i in range(len(lasers)):
            the_tf=self.adapt_tf_situation1(tfss[i])
            the_laser=self.generate_cartesian_points(lasers[i],the_tf)
            for point in the_laser:
                #Condition to ignore the car 
                if((point[0]<=self.car_hitbox_max_x and point[0]>=self.car_hitbox_min_x) and (point[1]<=self.car_hitbox_max_y and point[1]>=self.car_hitbox_min_y)):
                    continue
                else:
                    coord_laser.append((point[0],point[1]))
            #poly = MultiPoint(coord_laser).convex_hull
            #print(str(poly))
        xx=[p[0] for p in coord_laser]
        yy=[p[1] for p in coord_laser]
        return (xx,yy)


    def adapt_tf_situation1(self,a_tf):
        #Function that changes the tf matrice for a especific situation
        #More especifically it will the change the matrice from 3d transformations to 2D
        # a_tf=the matrice to be changed 
        a_tf[2][0]=0
        a_tf[2][1]=0
        a_tf[0][2]=0
        a_tf[1][2]=0
        a_tf[2][2]=1
        a_tf[2][3]=0
        new_tf=np.array([[a_tf[0][0],a_tf[0][1],a_tf[0][3]],[a_tf[1][0],a_tf[1][1],a_tf[1][3]],[0,0,1]])
        return new_tf


    def generate_cartesian_points(self,point_cloud,tf_matrice):
        #Fuction to map all the points from the "point_cloud" variable to correct cartesian points
        #point_cloud=cloud of points from which it will be extracted every point individually
        #tf_matrice=Transfrom matrix from the road to the laser sensor
        lp=lg.LaserProjection()
        pc2_msg=lp.projectLaser(point_cloud)
        point_generator=pc2.read_points(pc2_msg)
        cart_points=[]
        for point in point_generator:
            if not math.isnan(point[2]):
                init_coord=np.array([[(point[0])*1000],[(point[1])*1000],[1]])
                final_coord=np.dot(tf_matrice,init_coord)
                cart_points.append(final_coord)
        return cart_points