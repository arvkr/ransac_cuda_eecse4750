import numpy as np
import scipy
import matplotlib.pyplot as plt
import os

from python_ransac_3d import do_ransac_3d as naive_ransac_3d
from ransac_pycuda_3d_level4 import do_ransac_3d as cuda_ransac_3d

def plot_benchmark(sizes, naive_time, cuda_time, fname, xlabel, ylabel, title, logscale=True):

    fig, ax = plt.subplots(figsize=(10, 10))
    
    # ax.plot(sizes, naive_time, label = 'naive', marker='o')
    ax.plot(sizes, cuda_time, label = 'cuda', marker='o')
    if logscale:
        ax.set_xscale('log', base=2)
    ax.legend()
    ax.grid()
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    plt.savefig(fname)

if __name__ == "__main__":

    n_samples_test = False
    ransac_iterations_test = True

    if n_samples_test:
        # Ransac parameters
        ransac_iterations = 20  # number of iterations
        ransac_threshold = 3    # threshold
        ransac_ratio = 0.6     # ratio of inliers required to assert
                                # that a model fits well to data
        
        # generate sparse input data
        n_samples = 200
        # n_samples_all =  [128, 512, 1024, 4096, 8192, 16384, 65536, 262144, 1048576]          # number of input points||| Max tested = 3554432
        n_samples_all =  [128, 512, 1024, 4096, 8192, 16384, 65536, 262144, 1048576, 2097152, 4194304, 8388608]          # number of input points||| Max tested = 3554432
        # n_samples_all =  [128, 512, 1024, 4096]
        outliers_ratio = 0.3          # ratio of outliers
        
        naive_time = []
        cuda_time = []

        fname = 'ransac_3d_samples_cuda.png'
        xlabel='Number of data samples'
        ylabel='Execution Time (seconds)'
        title='RANSAC for 3D points (num_models=20)'

        # fname = 'ransac_2d_models.png'
        # xlabel='Number of models'
        # ylabel='Execution Time (seconds)'
        # title='RANSAC for 2D points varying the number of models (num_samples = 1024)'
        
        # for i, ransac_iterations in enumerate(ransac_iterations_all):
        for i, n_samples in enumerate(n_samples_all):

            n_inputs = 1
            n_outputs = 1

            np.random.seed(25)
            # print('itr', i)
            print(f"\n\nIteration {i}; n_samples {n_samples}; num_models:{ransac_iterations} ")

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

            # non-gaussian outliers (only on one side)
            #y_noise[outlier_indices] = 30*(np.random.normal(size=(n_outliers,n_outputs))**2)

            all_indices = np.arange(x_noise.shape[0])
            maybe_indices1 = np.random.choice(all_indices, size=(ransac_iterations), replace=True)
            maybe_indices2 = np.random.choice(all_indices, size=(ransac_iterations), replace=True)
            maybe_indices3 = np.random.choice(all_indices, size=(ransac_iterations), replace=True)

            num_runs = 5

            # avg_time = 0.0
            # for i in range(num_runs):
            #     t1 = naive_ransac_3d(x_noise,y_noise, z_noise, ransac_iterations, ransac_threshold, ransac_ratio, n_samples, maybe_indices1, maybe_indices2, maybe_indices3)
            #     avg_time += t1
            # naive_time.append(avg_time/num_runs)
            
            t_ = cuda_ransac_3d(x_noise,y_noise, z_noise, ransac_iterations, ransac_threshold, ransac_ratio, n_samples, maybe_indices1, maybe_indices2, maybe_indices3)
            
            avg_time = 0.0
            for i in range(num_runs):
                t2 = cuda_ransac_3d(x_noise,y_noise, z_noise, ransac_iterations, ransac_threshold, ransac_ratio, n_samples, maybe_indices1, maybe_indices2, maybe_indices3)
                avg_time += t2
            cuda_time.append(avg_time/num_runs)

        plot_benchmark(n_samples_all, naive_time, cuda_time, fname, xlabel, ylabel, title)
        # plot_benchmark(ransac_iterations_all, naive_time, cuda_time, fname, xlabel, ylabel, title, False)

# ----------
# ----------
    if ransac_iterations_test:
        # Ransac parameters
        ransac_iterations = 20  # number of iterations
        ransac_iterations_all = [10, 20, 40, 80, 100, 200, 400, 500, 700, 900, 1000]  # number of iterations
        # ransac_iterations_all = [10, 20, 40]  # number of iterations
        
        ransac_threshold = 3    # threshold
        ransac_ratio = 0.6     # ratio of inliers required to assert
                                # that a model fits well to data
        
        # generate sparse input data
        n_samples = 1024
        # n_samples_all =  [128, 512, 1024, 4096, 8192, 16384, 65536, 262144, 1048576]          # number of input points||| Max tested = 3554432
        n_samples_all =  [128, 512, 1024, 4096]
        outliers_ratio = 0.3          # ratio of outliers
        
        naive_time = []
        cuda_time = []

        # fname = 'ransac_3d.png'
        # xlabel='Number of data samples'
        # ylabel='Execution Time (seconds)'
        # title='RANSAC for 3D points (num_models=20)'

        fname = 'ransac_3d_models_cuda.png'
        xlabel='Number of models'
        ylabel='Execution Time (seconds)'
        title='RANSAC for 3D points varying the number of models (num_samples = 1024)'
        
        for i, ransac_iterations in enumerate(ransac_iterations_all):
        # for i, n_samples in enumerate(n_samples_all):

            n_inputs = 1
            n_outputs = 1

            np.random.seed(25)
            # print('itr', i)
            print(f"\n\nIteration {i}; n_samples {n_samples}; num_models:{ransac_iterations} ")

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

            # non-gaussian outliers (only on one side)
            #y_noise[outlier_indices] = 30*(np.random.normal(size=(n_outliers,n_outputs))**2)

            all_indices = np.arange(x_noise.shape[0])
            maybe_indices1 = np.random.choice(all_indices, size=(ransac_iterations), replace=True)
            maybe_indices2 = np.random.choice(all_indices, size=(ransac_iterations), replace=True)
            maybe_indices3 = np.random.choice(all_indices, size=(ransac_iterations), replace=True)

            num_runs = 5

            # avg_time = 0.0
            # for i in range(num_runs):
            #     t1 = naive_ransac_3d(x_noise,y_noise, z_noise, ransac_iterations, ransac_threshold, ransac_ratio, n_samples, maybe_indices1, maybe_indices2, maybe_indices3)
            #     avg_time += t1
            # naive_time.append(avg_time/num_runs)
            
            t_ = cuda_ransac_3d(x_noise,y_noise, z_noise, ransac_iterations, ransac_threshold, ransac_ratio, n_samples, maybe_indices1, maybe_indices2, maybe_indices3)
            
            avg_time = 0.0
            for i in range(num_runs):
                t2 = cuda_ransac_3d(x_noise,y_noise, z_noise, ransac_iterations, ransac_threshold, ransac_ratio, n_samples, maybe_indices1, maybe_indices2, maybe_indices3)
                avg_time += t2
            cuda_time.append(avg_time/num_runs)

        # plot_benchmark(n_samples_all, naive_time, cuda_time, fname, xlabel, ylabel, title)
        plot_benchmark(ransac_iterations_all, naive_time, cuda_time, fname, xlabel, ylabel, title, False)

