# EECSE4750 Heterogeneous Computing for Signals and Data

# Parallelization of RANSAC

* **Name: Arvind Kanesan Rathna (ak4728) and Kyle Coelho (kc3415)**

* CUDA has been used for implementation. Therefore, make sure the CUDA environment is activated:
```
$ cudaEnv
```
## Description and Organization of the major files

* It is possible to individually run any of the .py files below and view the sample outputs:

```
# For example,
$ python ransac_pycuda_level4.py
```
#### 3D RANSAC
- `3d_python/python_ransac_3d.py` - Serial implementation of RANSAC for 3D datapoints
- `3d_python/ransac_pycuda_3d_level4.py` - Fully parallelized of RANSAC for 3D datapoints. This version is called level 4 in the report and presentation.
- `3d_python/kernel_3d_ransac.cu` - The CUDA kernels for 3D RANSAC
- `3d_python/ransac_pycuda_3d_level2.py` - Level 2 parallelized of RANSAC for 3D datapoints. More description in the report/presentation.
- `3d_python/ransac_pycuda_3d_level1.py` - Level 1 parallelized of RANSAC for 3D datapoints. More description in the report/presentation.

#### 2D RANSAC
- `python_ransac.py` - Serial implementation of RANSAC for 2D datapoints
- `ransac_pycuda_level4.py` - Fully parallelized of RANSAC for 2D datapoints. This version is called level 4 in the report and presentation.
- `kernel_ransac.cu` - The CUDA kernels for 2D RANSAC
- `ransac_pycuda_level3.py` - Level 3 parallelized of RANSAC for 2D datapoints. More description in the report/presentation.
- `ransac_pycuda_level2.py` - Level 2 parallelized of RANSAC for 2D datapoints. More description in the report/presentation.
- `ransac_pycuda_level1.py` - Level 1 parallelized of RANSAC for 2D datapoints. More description in the report/presentation.

## Benchmarking/Stress Tests for 3D RANSAC and 2D RANSAC
* To reproduce the benchmarks graphs for the 3D version of RANSAC as shown in the presentation and report, run the below command:
    * First graph generated is named `ransac_3d_samples_cuda.png` by default. This plots the execution time for serial and cuda as we vary the size of the dataset.
    * Second graphs generated is named `ransac_3d_models_cuda.png` by default. This plots the execution time for serial and cuda as we vary the number of RANSAC models.
    * By setting the boolean `plot_cuda_mem` to True, we can choose to also print the execution of split-up between memory transfer and computation time for CUDA.
    * It has been verified that the outputs between serial and CUDA match perfectly. Randomization has been taken into account be setting an initial seed value.
```
$ cd 3d_python/
$ python 3d_benchmarking.py
```
* To profile the code using NSight, first set the flags `n_samples_test` and `ransac_iterations_test` to `False` in `3d_benchmarking.py`. Then run:
    * In NSight, we can view the profiling information for SM%, Mem%, etc, as we vary the block size.
```
$ cd 3d_python/
$ nv-nsight-cu-cli -o metrics python 3d_benchmarking.py > output.txt
$ nv-nsight-cu metrics.nsight-cuprof-report
```

* In a similar vein, we can run the benchmarks for the 2D version of RANSAC and generate similar plots as shown in the report:

```
$ python benchmarking.py 
```