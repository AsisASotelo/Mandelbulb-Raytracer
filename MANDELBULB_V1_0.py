
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#Asis A Sotelo
#
#24Jul2019 Created Program, Added Raytracing Algorithm and the Plane/Observer
#26Jul2019 Found Distance Estimator equation DanielWhite
#28Jul2019 Decided to Change the Program to two seperate files mandel_func.py  and 
#31Jul2019 Added User Screen and the <ENTER> Prompt



import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import time
import sys


"""
	A program that implements an algorithm to plot the Mandelbulb, a 3D representation of the mandelbrot set. The algorithm developed by Daniel White and Paul Nylander circumvents the limitations of complex numbers 2D representation by rotatig the values of the mandelbrot set utilizing spherical coordinates. The program creates this 3D object and then implements a distance estimator function that the raytracing of points on the mandelbulb. Because by their very nature Mandelbulbs have difficult to specify points the distance estimator allows for efficient calculation of distances from the observer to the mandelbulb and then colors the pixel correctly. 





"""
toolbar_width = 40










###############################################################################
## Gets boundaries of the plane 

def bound(center, span, zoom):
    return center - span/2.**zoom, center + span/2.**zoom
###############################################################################

## Gets the points of the plane on which the image of the mandelbulb will be created 

def planepoints(Q, center, span, zoom, width, height, eps=1e-4):
    x_min, x_max = bound(center[0], span[0], zoom)
    y_min, y_max = bound(center[1], span[1], zoom)
    a, b, c = Q
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    x, y = np.meshgrid(x, y)
    x, y = x.reshape(-1), y.reshape(-1)
    if np.abs(c) > eps:
        z = -(a*x + b*y)/c
        P = np.vstack((x, y, z)).T
    elif np.abs(a) > eps:
        z = -(c*x + b*y)/a
        P = np.vstack((z, y, x)).T
    elif np.abs(b) > eps:
        z = -(a*x + c*y)/b
        P = np.vstack((x, z, y)).T
    return P
##############################################################################

## Determines the distance between the plane,observer and the mandelbulb surface


def DES(positions, max_iter, degree=4, bail=1000):
    m = positions.shape[0]
    xo, yo, zo = np.zeros(m), np.zeros(m), np.zeros(m)
    xp, yp, zp = positions[:, 0], positions[:, 1], positions[:, 2]
    dr = np.zeros(m) + 1
    r = np.zeros(m)
    theta = np.zeros(m)
    phi = np.zeros(m)
    rn = np.zeros(m)
    
    
    ############
    
# HERE IS WHERE MOST OF THE COMPUTATION TIME TAKES PLACE THIS IS REALLY SLOW
    
    ### IMPLEMENTATION OF THE WHITE NYLANDER ALGORITHM
     
    for j in range(max_iter):


        r                = np.sqrt(xo*xo + yo*yo + zo*zo)
        thresh           = r < bail
        dr[thresh]      = np.power(r[thresh], degree - 1) * degree * dr[thresh] + 1.0

        theta[thresh] = np.arctan2( np.sqrt(xo[thresh] * xo[thresh] + np.square(yo[thresh])), zo[thresh])
        phi[thresh]   = np.arctan2( yo[thresh] , xo[thresh] ) # Arctan2 Finds the angle from positive x-axis to point(x,y)!=0

        rn[thresh]    =     r[thresh] ** degree
        theta[thresh] = theta[thresh] *  degree
        phi[thresh]   =   phi[thresh] *  degree

        xo[thresh] = rn[thresh] * np.sin(theta[thresh])    * np.cos(phi[thresh]) + xp[thresh]
        yo[thresh] = rn[thresh] * np.sin(theta[thresh])    * np.sin(phi[thresh]) + yp[thresh]
        zo[thresh] = rn[thresh] * np.cos(theta[thresh])                          + zp[thresh]

    return (0.5 * np.log(r) * r / dr)

###############################################################################

## THE RAYTRACER THIS NEEDS DES IN ORDER TO WORK FOR A THE MANDELBULB SINCE THE EDGES ARE NOT REALLY EDGES



def trace(s, directions, max_steps, min_distance, max_iter, degree, bail, power):

    total_distance = np.zeros(directions.shape[0]) # Takes the distance from the plane_p to the observer values 
    keep_iterations = np.ones_like(total_distance) # 
    steps = np.zeros_like(total_distance)

    toolbar_width = 40
    
    sys.stdout.write("[%s]"% (" " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width + 1))
    
    
    for k in range(toolbar_width):
        time.sleep(0.1)
        for i in range(max_steps): 
        
            
            positions = s[np.newaxis, :] + total_distance[:, np.newaxis] * directions
            distance = DES(positions, max_iter, degree, bail)
            
            keep_iterations[distance < min_distance] = 0
            total_distance += distance * keep_iterations
            steps += keep_iterations
            sys.stdout.write("-")
            sys.stdout.flush()
            
        sys.stdout.write("")
        
        
        return 1 - (steps/max_steps)**power

##############################################################################
## GETS THE DIRECTIONS OF THE OBSERVER WITH THE PLANE 


def direction(plane_p, observer):
    
    
    
    
    vector_obs_to_p = np.array(plane_p - observer)
    vector_obs_to_p = vector_obs_to_p/np.linalg.norm(vector_obs_to_p, axis=1)[:, np.newaxis]
    
    
    return vector_obs_to_p

##############################################################################
def mandelb(a=4, b=np.array([1, 1, -3.]), c=32, d=32, e=32000, f=5e-3, g=0, p=0.2, w=500, h=500, s=[1.2, 1.2], center=[0, 0]):
    
    
    #CHANGES THE MANDELBULB OBJECT IN THE FRAME
    degree = a
    power = p
    max_steps = c
    max_iter =d
    min_distance =f
    bail = e
    
    # CREATES THE OBSERVER VIEW OR THE 2D PLANE 
    OBS = b
    ZOOM =g
    WIDTH =w
    HEIGHT = h
    SPAN =s
    CENTER = center
    
    # CREATION OF THE 2D Image RAYTRACED
    
    plane_points = planepoints(OBS, center=CENTER, span=SPAN, zoom=ZOOM, width=WIDTH, height=HEIGHT)
    directions = direction(plane_points, OBS)
    image = trace(OBS, directions, max_steps, min_distance, max_iter, degree, bail, power)
    image = image.reshape(WIDTH, HEIGHT)




    return image


##############################################################################


## START OF THE USER EXPERIENCE 


print("#####################################################\n")
print("############ Mandelbulb Program #####################\n")
print("#####################################################\n")
print("                                                     \n")
print("This program will plot a 3D object with raytracing it\n")
print("\nshould take 3~4 minutes to process. Version 1.0\n")
input("Press <Enter> to Continue :    \n")
print("Loading image .... \n")
image = mandelb()
scipy.misc.toimage(image, cmin = 0.0, cmax = 1,mode = 'L' ).save("image_deg8.png")
fig = plt.figure()
plt.imshow(image)
fig.show()
print("\nImage Completed...\n")
print("Image saved as 'image.png'\n")
input("Press <ENTER> to Exit ... ")
