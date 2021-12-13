import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
import sys
from numpy.core.fromnumeric import size
from pycuda import driver, compiler, gpuarray, tools
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit
import time

# Ransac parameters
ransac_iterations = 20  # number of iterations
ransac_threshold = 3    # threshold
ransac_ratio = 0.6      # ratio of inliers required to assert
                        # that a model fits well to data
 
# generate sparse input data
n_samples = 500               # number of input points
outliers_ratio = 0.4          # ratio of outliers
 
n_inputs = 1
n_outputs = 1

np.random.seed(21)

# generate samples
x = 30*np.random.random((n_samples, n_inputs) )
 
# generate line's slope (called here perfect fit)
perfect_fit = 0.5*np.random.normal(size=(n_inputs, n_outputs) )
 
# compute output
y = scipy.dot(x,perfect_fit)

# add a little gaussian noise
x_noise = x + np.random.normal(size=x.shape)
y_noise = y + np.random.normal(size=y.shape)
 
# add some outliers to the point-set
n_outliers = int(outliers_ratio*n_samples)
indices = np.arange(x_noise.shape[0])
np.random.shuffle(indices)
outlier_indices = indices[:n_outliers]
 
x_noise[outlier_indices] = 30*np.random.random(size=(n_outliers,n_inputs))
 
# gaussian outliers
y_noise[outlier_indices] = 30*np.random.normal(size=(n_outliers,n_outputs))
 
# non-gaussian outliers (only on one side)
#y_noise[outlier_indices] = 30*(np.random.normal(size=(n_outliers,n_outputs))**2)

def find_line_model(points):
    """ find a line model for the given points
    :param points selected points for model fitting
    :return line model
    """
 
    # [WARNING] vertical and horizontal lines should be treated differently
    #           here we just add some noise to avoid division by zero
 
    # find a line model for these points
    m = (points[1,1] - points[0,1]) / (points[1,0] - points[0,0] + sys.float_info.epsilon)  # slope (gradient) of the line
    c = points[1,1] - m * points[1,0]                                     # y-intercept of the line
 
    return m, c

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

def ransac_plot(n, x, y, m, c, final=False, x_in=(), y_in=(), points=()):
    """ plot the current RANSAC step
    :param n      iteration
    :param points picked up points for modeling
    :param x      samples x
    :param y      samples y
    :param m      slope of the line model
    :param c      shift of the line model
    :param x_in   inliers x
    :param y_in   inliers y
    """
 
    fname = "figure_" + str(n) + ".png"
    line_width = 1.
    line_color = '#0080ff'
    title = 'iteration ' + str(n)
 
    if final:
        fname = "final.png"
        line_width = 3.
        line_color = '#ff0000'
        title = 'final solution'
 
    plt.figure("Ransac", figsize=(15., 15.))
 
    # grid for the plot
    grid = [min(x) - 10, max(x) + 10, min(y) - 20, max(y) + 20]
    plt.axis(grid)
 
    # put grid on the plot
    plt.grid(b=True, which='major', color='0.75', linestyle='--')
    #plt.xticks([i for i in range(min(x) - 10, max(x) + 10, 5)])
    #plt.yticks([i for i in range(min(y) - 20, max(y) + 20, 10)])
 
    # plot input points
    plt.plot(x[:,0], y[:,0], marker='o', label='Input points', color='#00cc00', linestyle='None', alpha=0.4)
 
    # draw the current model
    plt.plot(x, m*x + c, 'r', label='Line model', color=line_color, linewidth=line_width)
 
    # draw inliers
    if not final:
        plt.plot(x_in, y_in, marker='o', label='Inliers', linestyle='None', color='#ff0000', alpha=0.6)
 
    # draw points picked up for the modeling
    if not final:
        plt.plot(points[:,0], points[:,1], marker='o', label='Picked points', color='#0000cc', linestyle='None', alpha=0.6)
 
    plt.title(title)
    plt.legend()
    plt.savefig(fname)
    plt.close()

data = np.hstack( (x_noise,y_noise) ).astype(np.float32)
data = np.array(data)
 
ratio = 0.
model_m = 0.
model_c = 0.

all_indices = np.arange(x_noise.shape[0])
maybe_indices1 = np.random.choice(all_indices, size=(ransac_iterations), replace=True)
maybe_indices2 = np.random.choice(all_indices, size=(ransac_iterations), replace=True)
same_indices = (maybe_indices1 == maybe_indices2)
maybe_indices1[same_indices] +=1

# pick up two random points
maybe_points1 = data[maybe_indices1, :]
maybe_points2 = data[maybe_indices2, :]

mod = SourceModule(open('kernel_ransac.cu').read())
maybe_points1_d = gpuarray.to_gpu(maybe_points1)
maybe_points2_d = gpuarray.to_gpu(maybe_points2)
m_d = gpuarray.empty(shape=ransac_iterations, dtype=np.float32)
c_d = gpuarray.empty(shape=ransac_iterations, dtype=np.float32)
blockDim_model = (ransac_iterations, 1, 1) 
gridSize_model = (1, 1, 1)

cuda_find_line_model = mod.get_function('find_line_model')

# find a line model for these points
cuda_find_line_model(maybe_points1_d, maybe_points2_d, m_d, c_d, np.int32(ransac_iterations), block=blockDim_model, grid=gridSize_model)

m_host = m_d.get()
c_host = c_d.get()

x_list = []
y_list = []

# output = np.zeros(shape=(data.shape[0]*ransac_iterations), dtype=np.float32)

e_start = cuda.Event()
e_end = cuda.Event()

points_d = gpuarray.to_gpu(data)
# dist_output_d = gpuarray.to_gpu(output)
dist_output_d = gpuarray.empty(shape=(data.shape[0]*ransac_iterations), dtype=np.float32)
blockSize = 1024
blockDim = (blockSize, 1, 1) 
gridSize = (ransac_iterations, 1, 1)

dist_model_parallel = mod.get_function('distance_model_parallel')

dist_model_parallel(points_d, dist_output_d, m_d, c_d, np.int32(data.shape[0]), block=blockDim, grid=gridSize)

e_end.record()
e_end.synchronize()

distances = dist_output_d.get()
distances = distances.reshape((ransac_iterations, data.shape[0]))
print(distances.shape)

# perform RANSAC iterations
dists = np.logical_and([distances > 0], [distances < ransac_threshold])[0]
num_all = np.sum(dists, axis=1)
num_all -= 2
print('Num = ', num_all, num_all.shape)

for it in range(ransac_iterations):

    m = m_host[it]
    c = c_host[it]
    num = num_all[it]
 
    #x_inliers = np.array(x_list)
    #y_inliers = np.array(y_list)
 
    # in case a new model is better - cache it
    if num/float(n_samples-2) > ratio:
        ratio = num/float(n_samples)
        model_m = m
        model_c = c
 
    print ('  inlier ratio = ', num/float(n_samples))
    print ('  model_m = ', m)
    print ('  model_c = ', c)
 
    # plot the current step
    # ransac_plot(it, x_noise,y_noise, m, c, False, x_inliers, y_inliers, maybe_points)
 
    # we are done in case we have enough inliers
    if num > n_samples*ransac_ratio:
        print ('The model is found !')
        break

# plot the final model
ransac_plot(0, x_noise,y_noise, model_m, model_c, True)
 
print ('\nFinal model:\n')
print ('  ratio = ', ratio)
print ('  model_m = ', model_m)
print ('  model_c = ', model_c)