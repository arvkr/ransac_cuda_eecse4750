__global__ void distance(const float *points, float *output, float m, float c, int N){

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N){

        // 2 is hard coded but it is 2d points
        float x0 = points[i*2];
        float y0 = points[i*2 + 1];
        // printf("%0.2f ", x0);
        // printf("%0.2f ", y0);

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

// __global__ void find_line_model(const float *maybe_points1, const float *maybe_points2, float *m, float *c){

//     int i = blockIdx.x * blockDim.x + threadIdx.x;

// }