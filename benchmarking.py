import numpy as np
import scipy
import matplotlib.pyplot as plt

from python_ransac import do_ransac as naive_ransac
from ransac_pycuda_level4 import do_ransac as cuda_ransac_level4

def plot_benchmark(sizes, naive_time, cuda_time, fname, xlabel, ylabel, title, logscale=True):

    fig, ax = plt.subplots(figsize=(10, 10))
    
    ax.plot(sizes, naive_time, label = 'naive', marker='o')
    ax.plot(sizes, cuda_time, label = 'cuda', marker='o')
    if logscale:
        ax.set_xscale('log', base=2)
    ax.legend()
    ax.grid()
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    plt.savefig(fname)



if __name__ == "__main__":

    # Flags to profile change in execution time by varying number of data samples, number of ransac iterations
    n_samples_test = True
    ransac_iterations_test = True
    do_naive = True

    if n_samples_test:

        # Ransac parameters
        ransac_iterations = 20  # number of iterations
        ransac_threshold = 3    # threshold
        ransac_ratio = 0.6      # ratio of inliers required to assert
                                # that a model fits well to data
        
        # generate sparse input data
        # n_samples_all =  [128, 512, 1024, 4096, 8192, 16384, 65536, 262144, 1048576]          # number of input points||| Max tested = 3554432
        n_samples_all =  [128, 512, 1024, 4096]          # number of input points||| Max tested = 3554432
        
        naive_time = []
        cuda_time = []
        fname = 'ransac_2d.png'
        xlabel='Number of data samples'
        ylabel='Execution Time (seconds)'
        title='RANSAC for 2D points'
        n_samples = n_samples_all[0]

        for i, n_samples in enumerate(n_samples_all):
            print(f'\n\nIteration {i}; n_samples {n_samples}; num_models:{ransac_iterations}')
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
            
            all_indices = np.arange(x_noise.shape[0])
            maybe_indices1 = np.random.choice(all_indices, size=(ransac_iterations), replace=True)
            maybe_indices2 = np.random.choice(all_indices, size=(ransac_iterations), replace=True)
            
            num_runs = 5
            avg_time = 0.0
            for i in range(num_runs):
                t1 = naive_ransac(x_noise, y_noise, ransac_iterations, ransac_threshold, n_samples, maybe_indices1, maybe_indices2)
                avg_time += t1
            naive_time.append(avg_time/num_runs)
            
            t2 = cuda_ransac_level4(x_noise, y_noise, ransac_iterations, ransac_threshold, n_samples, maybe_indices1, maybe_indices2)
            avg_time = 0.0
            for i in range(num_runs):
                t2 = cuda_ransac_level4(x_noise, y_noise, ransac_iterations, ransac_threshold, n_samples, maybe_indices1, maybe_indices2)
                avg_time += t2
            cuda_time.append(avg_time/num_runs)

        plot_benchmark(n_samples_all, naive_time, cuda_time, fname, xlabel, ylabel, title)

    if ransac_iterations_test:
        
        # Ransac parameters
        ransac_iterations_all = [10, 20, 40, 80, 100, 200, 400, 500, 700, 900, 1000]  # number of iterations
        # ransac_iterations_all = [10, 20, 40, 80]  # number of iterations
        ransac_threshold = 3    # threshold
        ransac_ratio = 0.6      # ratio of inliers required to assert
                                # that a model fits well to data
        
        # generate sparse input data
        n_samples_all = [2048]
        
        naive_time = []
        cuda_time = []
        n_samples = n_samples_all[0]

        fname = 'ransac_2d_models.png'
        xlabel='Number of models'
        ylabel='Execution Time (seconds)'
        title='RANSAC for 2D points varying the number of models (num_samples = 1024)'

        for i, ransac_iterations in enumerate(ransac_iterations_all):
        
            print(f'\n\nIteration {i}; n_samples {n_samples}; num_models:{ransac_iterations}')
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
            
            all_indices = np.arange(x_noise.shape[0])
            maybe_indices1 = np.random.choice(all_indices, size=(ransac_iterations), replace=True)
            maybe_indices2 = np.random.choice(all_indices, size=(ransac_iterations), replace=True)
            
            num_runs = 5
            avg_time = 0.0
            for i in range(num_runs):
                t1 = naive_ransac(x_noise, y_noise, ransac_iterations, ransac_threshold, n_samples, maybe_indices1, maybe_indices2)
                avg_time += t1
            naive_time.append(avg_time/num_runs)
            
            t2 = cuda_ransac_level4(x_noise, y_noise, ransac_iterations, ransac_threshold, n_samples, maybe_indices1, maybe_indices2)
            avg_time = 0.0
            for i in range(num_runs):
                t2 = cuda_ransac_level4(x_noise, y_noise, ransac_iterations, ransac_threshold, n_samples, maybe_indices1, maybe_indices2)
                avg_time += t2
            cuda_time.append(avg_time/num_runs)

        plot_benchmark(ransac_iterations_all, naive_time, cuda_time, fname, xlabel, ylabel, title, False)