#include "float.h"
__global__ void distance(const float *points, float *output, float m, float c, int N){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N){

        // 2 is hard coded but it is 2d points
        float x0 = points[i*2];
        float y0 = points[i*2 + 1];
        // intersection point with the model
        float x1 = (x0 + (m*y0) - (m*c))/(1 + (m*m));
        float y1 = ((m*x0) + ((m*m)*y0) - ((m*m)*c))/(1 + (m*m)) + c;
        float dist = sqrt(((x1 - x0)*(x1 - x0)) + ((y1 - y0)*(y1 - y0)));
        output[i] = dist;
    } 
}

__global__ void find_line_model(const float *maybe_points1, const float *maybe_points2, float *m, float *c, int num_models){

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_models){

        float x1 = maybe_points1[i*2];
        float y1 = maybe_points1[i*2 + 1];

        float x2 = maybe_points2[i*2];
        float y2 = maybe_points2[i*2 + 1];

        m[i] = (y2 - y1)/(x2 - x1 + FLT_MIN);
        c[i] = y2 - m[i]*x2;

    }

}

__global__ void distance_model_parallel(const float *points, float *output, float *m_all, float *c_all, int num_samples){

    int tx = threadIdx.x;
    float m = m_all[blockIdx.x];
    float c = c_all[blockIdx.x];
    int op_idx = blockIdx.x*num_samples + tx;

    if (tx < num_samples){

        float x0 = points[tx*2];
        float y0 = points[tx*2 + 1];

        // intersection point with the model
        float x1 = (x0 + (m*y0) - (m*c))/(1 + (m*m));
        float y1 = ((m*x0) + ((m*m)*y0) - ((m*m)*c))/(1 + (m*m)) + c;
        float dist = sqrt(((x1 - x0)*(x1 - x0)) + ((y1 - y0)*(y1 - y0)));
        output[op_idx] = dist;
    } 
}

__global__ void distance_model_parallel_large(const float *points, float *output, float *m_all, float *c_all, int num_samples){

    int tx = threadIdx.x;
    int point_idx = blockIdx.x * blockDim.x + tx;

    float m = m_all[blockIdx.y];
    float c = c_all[blockIdx.y];
    int op_idx = blockIdx.y*num_samples + point_idx;

    if (point_idx < num_samples){

        float x0 = points[point_idx*2];
        float y0 = points[point_idx*2 + 1];

        // intersection point with the model
        float x1 = (x0 + (m*y0) - (m*c))/(1 + (m*m));
        float y1 = ((m*x0) + ((m*m)*y0) - ((m*m)*c))/(1 + (m*m)) + c;
        float dist = sqrt(((x1 - x0)*(x1 - x0)) + ((y1 - y0)*(y1 - y0)));
        output[op_idx] = dist;
    } 
}