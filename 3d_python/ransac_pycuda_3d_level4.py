#!/usr/bin/env python
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import scipy
from numpy.core.fromnumeric import size
from pycuda import driver, compiler, gpuarray, tools
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit
import time

def find_intercept_point(m, c, x0, y0):
    """ find the distance from point (x0,y0,z0) to the modelled plane
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
    # plt.savefig(folder_name + '/' + fname)
    plt.close()

def do_ransac_3d(x_noise,y_noise, z_noise, ransac_iterations, ransac_threshold, ransac_ratio, n_samples, maybe_indices1, maybe_indices2, maybe_indices3, blockSize=1024):

    data = np.hstack( (x_noise,y_noise, z_noise))
    data = np.array(data)
    
    # Variable to check if current model has more inliers than the best model so far
    ratio = 0.

    # Execution time benchmarking
    tik = cuda.Event()
    tok = cuda.Event()
    st_mem1 = cuda.Event()
    st_mem2 = cuda.Event()
    st_mem3 = cuda.Event()
    st_mem4 = cuda.Event()
    en_mem1 = cuda.Event()
    en_mem2 = cuda.Event()
    en_mem3 = cuda.Event()
    en_mem4 = cuda.Event()

    tik.record()
    start = time.time()
    # Make sure the randomly picked points for modelling are not the same
    same_indices = (maybe_indices1 == maybe_indices2)
    maybe_indices1[same_indices] +=1
    same_indices = (maybe_indices1 == maybe_indices3)
    maybe_indices1[same_indices] +=1
    same_indices = (maybe_indices2 == maybe_indices3)
    maybe_indices2[same_indices] +=1


    # pick up three random points
    maybe_points1 = data[maybe_indices1, :]
    maybe_points2 = data[maybe_indices2, :]
    maybe_points3 = data[maybe_indices3, :]

    mod = SourceModule(open('kernel_3d_ransac.cu').read())
    st_mem1.record()
    
    # Allocate GPU memory to store model parameters
    maybe_points1_d = gpuarray.to_gpu(maybe_points1)
    maybe_points2_d = gpuarray.to_gpu(maybe_points2)
    maybe_points3_d = gpuarray.to_gpu(maybe_points3)
    a_d = gpuarray.empty(shape=ransac_iterations, dtype=np.float64)
    b_d = gpuarray.empty(shape=ransac_iterations, dtype=np.float64)
    c_d = gpuarray.empty(shape=ransac_iterations, dtype=np.float64)
    d_d = gpuarray.empty(shape=ransac_iterations, dtype=np.float64)
    en_mem1.record()
    en_mem1.synchronize()

    # One thread calculates model parameters for one model (ransac_iterations define the number of models)
    blockDim_model = (ransac_iterations, 1, 1) 
    gridSize_model = (1, 1, 1)

    cuda_find_plane_model = mod.get_function('find_plane_model')

    # Call kernel to find plane model in parallel for all models
    cuda_find_plane_model(maybe_points1_d, maybe_points2_d, maybe_points3_d, a_d, b_d, c_d, d_d, np.int32(ransac_iterations), block=blockDim_model, grid=gridSize_model)

    st_mem2.record()
    # Retreive the calculated model parameters from GPU
    a_host = a_d.get()
    b_host = b_d.get()
    c_host = c_d.get()
    d_host = d_d.get()
    en_mem2.record()
    en_mem2.synchronize()

    end = time.time()
    time1 = end - start
    start = time.time()
    output = np.zeros(shape=(data.shape[0]), dtype=np.float64)

    e_start = cuda.Event()
    e_end = cuda.Event()

    # Allocate memory to calculate error distances
    st_mem3.record()
    points_d = gpuarray.to_gpu(data)
    dist_output_d = gpuarray.empty(shape=(data.shape[0]*ransac_iterations), dtype=np.float64)
    en_mem3.record()
    en_mem3.synchronize()

    # Each thread calculates error for one data sample in one model
    blockDim = (blockSize, 1, 1) 
    # Grid_x determined number of blocks for one model
    # Grid_y determines the number of models
    gridSize = (((data.shape[0] - 1)//blockSize + 1), ransac_iterations, 1)

    dist_3d_model_parallel_large = mod.get_function('distance_3d_model_parallel_large')
    # Call kernel tp find error distances for all models and all points
    dist_3d_model_parallel_large(points_d, dist_output_d, a_d, b_d, c_d, d_d, np.int32(data.shape[0]), block=blockDim, grid=gridSize)

    e_end.record()
    e_end.synchronize()

    st_mem4.record()
    distances = dist_output_d.get()
    en_mem4.record()
    en_mem4.synchronize()

    end = time.time()
    time2 = end - start
    start = time.time()
    distances = distances.reshape((ransac_iterations, data.shape[0]))

    # Calucalate the number of inlier points for each model (NumPy code - can be parallelized in the future using scan kernel)
    dists = np.logical_and([distances > 0], [distances < ransac_threshold])[0]
    num_all = np.sum(dists, axis=1)
    # Remove the 3 points used for modeling
    num_all -= 3

    # Find the best model by checking the number of inlier points for all models - Can be parallelized in the future
    for it in range(ransac_iterations):
        
        a = a_host[it]
        b = b_host[it]
        c = c_host[it]
        d = d_host[it]
        num = num_all[it]
    
        # in case a new model is better - cache it
        if num/float(n_samples-2) > ratio:
            ratio = num/float(n_samples)
            model_a = a
            model_b = b
            model_c = c
            model_d = d
    
        # print ('\n  inlier ratio = ', num/float(n_samples))
        # print ('  model_a = ', a)
        # print ('  model_b = ', b)
        # print ('  model_c = ', c)
        # print ('  model_d = ', d)
    
        # plot the current step
        # ransac_plot(it, x_noise,y_noise, m, c, False, x_inliers, y_inliers, maybe_points)
    
    end = time.time()
    time3 = end - start
    tok.record()
    tok.synchronize()
    mem_transfer_time = (st_mem1.time_till(en_mem1) + st_mem2.time_till(en_mem2) + st_mem3.time_till(en_mem3) + st_mem4.time_till(en_mem4)) / 1000
    # print('Time Taken = ', tok - tik)
    # print('Mem Transfer Time = ', mem_transfer_time)

    # plot the final model
    # ransac_plot(0, x_noise,y_noise, z_noise, a, b, c, d, True) 

    # print ('\nFinal model:\n')
    # print ('  ratio = ', ratio)
    # print ('  model_a = ', model_a)
    # print ('  model_b = ', model_b)
    # print ('  model_c = ', model_c)
    # print ('  model_d = ', model_d)
    return tik.time_till(tok)/1000, mem_transfer_time

if __name__ == "__main__":

    # folder_name = os.path.join(os.getcwd(), 'cuda_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    # os.makedirs(folder_name)

    # Ransac parameters
    ransac_iterations = 20  # number of iterations
    ransac_threshold = 3    # threshold
    ransac_ratio = 0.6     # ratio of inliers required to assert
                            # that a model fits well to data
    
    # generate sparse input data
    n_samples = 4096             # number of input points ||| Max tested = 3554432
    outliers_ratio = 0.3          # ratio of outliers
    
    n_inputs = 1
    n_outputs = 1

    np.random.seed(25)

    # generate samples
    x = 30*np.random.random((n_samples, n_inputs)).astype(np.float64)
    
    # generate line's slope (called here perfect fit)
    perfect_fit = 0.5*np.random.normal(size=(n_inputs, n_outputs)).astype(np.float64)
    
    # compute output
    y = scipy.dot(x,perfect_fit)
    z = 30*np.random.random((n_samples, n_inputs)).astype(np.float64)

    # add a little gaussian noise
    x_noise = x + np.random.normal(size=x.shape).astype(np.float64)
    y_noise = y + np.random.normal(size=y.shape).astype(np.float64)
    z_noise = z + np.random.normal(size=z.shape).astype(np.float64)
    
    # add some outliers to the point-set
    n_outliers = int(outliers_ratio*n_samples)
    indices = np.arange(x_noise.shape[0])
    np.random.shuffle(indices)
    outlier_indices = indices[:n_outliers]
    
    x_noise[outlier_indices] = 30*np.random.random(size=(n_outliers,n_inputs)).astype(np.float64)
    
    # gaussian outliers
    y_noise[outlier_indices] = 30*np.random.normal(size=(n_outliers,n_outputs)).astype(np.float64)
    z_noise[outlier_indices] = 30*np.random.normal(size=(n_outliers,n_outputs)).astype(np.float64)

    # Choose random 3 points per ransac_iteration to find the plane model
    all_indices = np.arange(x_noise.shape[0])
    maybe_indices1 = np.random.choice(all_indices, size=(ransac_iterations), replace=True)
    maybe_indices2 = np.random.choice(all_indices, size=(ransac_iterations), replace=True)
    maybe_indices3 = np.random.choice(all_indices, size=(ransac_iterations), replace=True)

    # Perform best-level4 parallelized RANSAC
    do_ransac_3d(x_noise,y_noise, z_noise, ransac_iterations, ransac_threshold, ransac_ratio, n_samples, maybe_indices1, maybe_indices2, maybe_indices3)


