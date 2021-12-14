from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
import sys
import os
import datetime
from numpy.core.fromnumeric import size
from pycuda import driver, compiler, gpuarray, tools
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit
import time

folder_name = os.path.join(os.getcwd(), 'cuda_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
os.makedirs(folder_name)

# Ransac parameters
ransac_iterations = 20  # number of iterations
ransac_threshold = 3    # threshold
ransac_ratio = 0.6     # ratio of inliers required to assert
                        # that a model fits well to data
 
# generate sparse input data
n_samples = 200               # number of input points ||| Max tested = 3554432
outliers_ratio = 0.3          # ratio of outliers
 
n_inputs = 1
n_outputs = 1

np.random.seed(25)

# generate samples
x = 30*np.random.random((n_samples, n_inputs))
 
# generate line's slope (called here perfect fit)
perfect_fit = 0.5*np.random.normal(size=(n_inputs, n_outputs) )
 
# compute output
y = scipy.dot(x,perfect_fit)
z = 30*np.random.random((n_samples, n_inputs) )

# add a little gaussian noise
x_noise = x + np.random.normal(size=x.shape)
y_noise = y + np.random.normal(size=y.shape)
z_noise = z + np.random.normal(size=z.shape)
 
# add some outliers to the point-set
n_outliers = int(outliers_ratio*n_samples)
indices = np.arange(x_noise.shape[0])
np.random.shuffle(indices)
outlier_indices = indices[:n_outliers]
 
x_noise[outlier_indices] = 30*np.random.random(size=(n_outliers,n_inputs))
 
# gaussian outliers
y_noise[outlier_indices] = 30*np.random.normal(size=(n_outliers,n_outputs))
z_noise[outlier_indices] = 30*np.random.normal(size=(n_outliers,n_outputs))

# non-gaussian outliers (only on one side)
#y_noise[outlier_indices] = 30*(np.random.normal(size=(n_outliers,n_outputs))**2)

def find_line_model(points):
    """ find a line model for the given points
    :param points selected points for model fitting
    :return line model
    """
 
    # [WARNING] vertical and horizontal lines should be treated differently
    #           here we just add some noise to avoid division by zero
 
    a1 = points[1][0] - points[0][0]
    b1 = points[1][1] - points[0][1]
    c1 = points[1][2] - points[0][2]
    a2 = points[2][0] - points[0][0]
    b2 = points[2][1] - points[0][1]
    c2 = points[2][2] - points[0][2]
    a = b1 * c2 - b2 * c1
    b = a2 * c1 - a1 * c2
    c = a1 * b2 - b1 * a2
    d = (- a * points[0][0] - b * points[0][1] - c * points[0][2])
    print ("equation of plane is ",)
    print (a, "x +",)
    print (b, "y +",)
    print (c, "z +",)
    print (d, "= 0.")
 
    return a, b, c, d

def find_intercept_point(m, c, x0, y0):
    """ find an intercept point of the line model with
        a normal from point (x0,y0) to it
    :param m slope of the line model
    :param c y-intercept of the line model
    :param x0 point's x coordinate
    :param y0 point's y coordinate
    :return intercept point
    """
 
    # intersection point with the model
    x = (x0 + m*y0 - m*c)/(1 + m**2)
    y = (m*x0 + (m**2)*y0 - (m**2)*c)/(1 + m**2) + c
 
    return x, y

def ransac_plot(n, x, y, z, a, b, c, d, final=False, x_in=(), y_in=(), z_in = (), points=()):

    fname = "figure_" + str(n) + ".png"
    line_width = 1.
    line_color = '#0080ff'
    title = 'iteration ' + str(n)
 
    if final:
        fname = "final.png"
        line_width = 3.
        line_color = '#ff0000'
        title = 'final solution'

 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
 
    # grid for the plot
    grid = [min(x) - 10, max(x) + 10, min(y) - 20, max(y) + 20, min(z) - 10, max(z) + 10]
    #plt.axis(grid)
 
    # put grid on the plot
    #ax.grid(b=True, which='major', color='0.75', linestyle='--')
    #plt.xticks([i for i in range(min(x) - 10, max(x) + 10, 5)])
    #plt.yticks([i for i in range(min(y) - 20, max(y) + 20, 10)])
 
    # plot input points
    ax.plot(x[:,0], y[:,0], z[:,0], marker='o', label='Input points', color='#00cc00', linestyle='None', alpha=0.4)
 
    # draw the current model
    X,Y = np.meshgrid(x,y)
    Z = (d - a*X - b*Y) / c
    ax.plot_surface(X, Y, Z, alpha = 0.1)
 
    # draw inliers
    if not final:
        ax.plot(x_in, y_in, z_in, marker='o', label='Inliers', linestyle='None', color='#ff0000', alpha=0.6)
 
    # draw points picked up for the modeling
    if not final:
        ax.plot(points[:,0], points[:,1], points[:,2], marker='o', label='Picked points', color='#0000cc', linestyle='None', alpha=0.6)
 
    plt.title(title)
    plt.legend()
    # plt.savefig(os.path.join(folder, fname))
    #if final:
    #plt.show()
    plt.savefig(folder_name + '/' + fname)
    plt.close()

data = np.hstack( (x_noise,y_noise, z_noise)).astype(np.float32)
data = np.array(data)
 
ratio = 0.
model_a = 0.
model_b = 0.
model_c = 0.
model_d = 0.

tik = time.time()
# perform RANSAC iterations
for it in range(ransac_iterations):
 
    # pick up two random points
    n = 3
 
    all_indices = np.arange(x_noise.shape[0])
    np.random.shuffle(all_indices)
 
    indices_1 = all_indices[:n]
    indices_2 = all_indices[n:]
 
    maybe_points = data[indices_1,:]
    test_points = data[indices_2,:]
 
    # find a line model for these points
    a, b, c, d = find_line_model(maybe_points)
 
    x_list = []
    y_list = []
    num = 0

    #print(test_points)
    mod = SourceModule(open('kernel_3d_ransac.cu').read())

    output = np.zeros(shape=(test_points.shape[0]), dtype=np.float32)
    print(output.shape)

    e_start = cuda.Event()
    e_end = cuda.Event()

    points_d = gpuarray.to_gpu(test_points)
    dist_output_d = gpuarray.to_gpu(output)
    blockSize = 1024
    blockDim = (blockSize, 1, 1) 
    gridDim = int((output.shape[0] - 1) / 1024 + 1)
    gridSize = (gridDim, 1, 1)

    dist = mod.get_function('distance')

    dist(points_d, dist_output_d, np.float32(a), np.float32(b), np.float32(c), np.float32(d), np.int32(output.shape[0]), block=blockDim, grid=gridSize)

    e_end.record()
    e_end.synchronize()

    distances = dist_output_d.get()
    #print(distances)
    dists = [np.logical_and([distances > 0], [distances < ransac_threshold])] 
    num = np.sum(dists)
    #print(num)
    print('Num = ', num)
 
    #x_inliers = np.array(x_list)
    #y_inliers = np.array(y_list)
 
    # in case a new model is better - cache it
    if num/float(n_samples-2) > ratio:
        ratio = num/float(n_samples)
        model_a = a
        model_b = b
        model_c = c
        model_d = d
 
    print ('  inlier ratio = ', num/float(n_samples))
    # print ('  model_a = ', model_a)
    # print ('  model_b = ', model_b)
    # print ('  model_c = ', model_c)
    # print ('  model_d = ', model_d)
    print ('  model_a = ', a)
    print ('  model_b = ', b)
    print ('  model_c = ', c)
    print ('  model_d = ', d)
 
    # plot the current step
    # ransac_plot(it, x_noise,y_noise, m, c, False, x_inliers, y_inliers, maybe_points)
 
    # we are done in case we have enough inliers
    if num > n_samples*ransac_ratio:
        print ('The model is found !')
        break

tok = time.time()
print('Time Taken = ', tok - tik)

# plot the final model
ransac_plot(0, x_noise,y_noise, z_noise, a, b, c, d, True) 

print ('\nFinal model:\n')
print ('  ratio = ', ratio)
print ('  model_a = ', model_a)
print ('  model_b = ', model_b)
print ('  model_c = ', model_c)
print ('  model_d = ', model_d)