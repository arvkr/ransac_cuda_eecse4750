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

folder_name = os.path.join(os.getcwd(), 'serial_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
os.makedirs(folder_name)

# Ransac parameters
ransac_iterations = 20  # number of iterations
ransac_threshold = 3    # threshold
ransac_ratio = 0.6      # ratio of inliers required to assert
                        # that a model fits well to data
 
# generate sparse input data
n_samples = 200               # number of input points ||| Max tested = 3554432
outliers_ratio = 0.4          # ratio of outliers

# What is the purpose of these 2 variables below:
n_inputs = 1
n_outputs = 1

np.random.seed(42)

# generate samples
x = 30*np.random.random((n_samples, n_inputs))
 
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
    plt.savefig(folder_name + '/' + fname)
    plt.close()

data = np.hstack( (x_noise,y_noise) ).astype(np.float32)
data = np.array(data)
 
ratio = 0.
model_m = 0.
model_c = 0.

kernelwrapper = """
    __global__ void distance(const float *points, float *output, float m, float c, int N){

        int i = blockIdx.x * blockDim.x + threadIdx.x;

        if (i < N){

            // 2 is hard coded but it is 2d points
            float x0 = points[i*2];
            float y0 = points[i*2 + 1];
            //printf("%0.2f ", x0);
            //printf("%0.2f ", y0);

            // intersection point with the model
            float x1 = (x0 + (m*y0) - (m*c))/(1 + (m*m));
            float y1 = ((m*x0) + ((m*m)*y0) - ((m*m)*c))/(1 + (m*m)) + c;
            float dist = sqrt(((x1 - x0)*(x1 - x0)) + ((y1 - y0)*(y1 - y0)));
            //printf("%0.2f ", x1);
            //printf("%0.2f ", y1);
            //printf("%0.2f", dist);

            output[i] = dist;

            // __syncthreads();

            // 3 is threshold. Pass in as param
            /*if (dist < 3){
                x_list.append(x0)
                y_list.append(y0)
                num += 1
            }*/
        } 
    }

    """

tik = time.time()
# perform RANSAC iterations
for it in range(ransac_iterations):
 
    # pick up two random points
    n = 2
 
    all_indices = np.arange(x_noise.shape[0])
    np.random.shuffle(all_indices)
 
    indices_1 = all_indices[:n]
    indices_2 = all_indices[n:]
 
    maybe_points = data[indices_1,:]
    test_points = data[indices_2,:]
 
    # find a line model for these points
    m, c = find_line_model(maybe_points)
 
    x_list = []
    y_list = []
    num = 0

    #print(test_points)
    mod = SourceModule(kernelwrapper)

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

    dist(points_d, dist_output_d, np.float32(m), np.float32(c), np.int32(output.shape[0]), block=blockDim, grid=gridSize)

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
        model_m = m
        model_c = c
 
    print ('  inlier ratio = ', num/float(n_samples))
    print ('  model_m = ', model_m)
    print ('  model_c = ', model_c)
 
    # plot the current step
    # ransac_plot(it, x_noise,y_noise, m, c, False, x_inliers, y_inliers, maybe_points)
 
    # we are done in case we have enough inliers
    if num > n_samples*ransac_ratio:
        print ('The model is found !')
        break

tok = time.time()
print('Time Taken = ', tok - tik)

# plot the final model
ransac_plot(0, x_noise,y_noise, model_m, model_c, True)
 
print ('\nFinal model:\n')
print ('  ratio = ', ratio)
print ('  model_m = ', model_m)
print ('  model_c = ', model_c)