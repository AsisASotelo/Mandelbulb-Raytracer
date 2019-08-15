# Mandelbulb-Raytracer
An algorithm based on White and Nylander Algorithm

Version 1.0 - 

Creates an image of a Mandelbulb which is a 3d extension of the 2-d Mandelbrot set. The image is 500 by 500 pixels and implements distance estimatation for the ray tracing allowing some shadowing to appear in the image.

The entirety of the code was written in python and implements standard modules. It is quite lengthy and you have to change the source code to get different types of images. The time to render is ~ 3-4 minutes on the rasipberry pi model 3B+ I ran it on. The next step would be to decrease computation time as the it is quite slow. Perhaps multithreading to utilize the multiple cores of the Raspberry Pi. 
