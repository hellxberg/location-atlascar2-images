# location-atlascar2-images
Tese: Localização do ATLASCAR2 com recurso a imagens de satélite ou aéreas
Repository dedicated to the Industrial Automation Masters

## Associated nformation
Author:Tiago Pereira
Department of Mechanical Engineering (DEM), University of Aveiro (UA)

## Advisor
Miguel Riem de Oliveira [GitHub](https://github.com/miguelriemoliveira/)

# THe Algorithm
## Getting Started 
All that is needed to apply this algorithm in a local machine is a simple "git clone" command of this repository

## Prerequisites
### ROS
The algorithm is being tested in ROS Melodic.
Installation instructions can be found at [www.ros.org](http://www.ros.org/)
### Common python libraries
- Numpy
-Opencv2
- Scipy
- Matplotlib

Can be installed with the following commands:
```
sudo apt install python-numpy
sudo apt-get install python-opencv
sudo apt-get install python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose
```

### Data
The algorithm will search for especific data being publish by ATLASCAR2. 
This data is stored in a bag file that can be downloaded at https://www.dropbox.com/s/7v2xp7yw5z3yby2/target5.bag?dl=0

The user most play the bag file 'target5.bag' with the following command:
```
rosbag play -l target5.bag
```

## Foreseeable Objectives
- Calibrate IPM system, more specifically, the system's precision 
- Create the non-obstacle polygon

## How to run the algorithm
To run the current algorithm apply the following command:
 ```
rosrun ipm_perception image_subscriber_v2.py
 ```

