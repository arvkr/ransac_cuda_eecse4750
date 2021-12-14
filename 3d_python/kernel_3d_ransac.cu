__global__ void distance(const double *points, double *output, double a, double b, double c, double d, int N){

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N){

        double x0 = points[i*3];
        double y0 = points[i*3 + 1];
        double z0 = points[i*3 + 2];
        //printf("%0.2f ", x0);
        //printf("%0.2f ", y0);

        // intersection point with the model
        double numer = abs(((a * x0) + (b * y0) + (c * z0) + d));
        double denom = sqrt((a * a) + (b * b) + (c * c));
        double dist = numer / denom;
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

__global__ void find_plane_model(const double *maybe_points1, const double *maybe_points2, const double *maybe_points3, double *a, double *b, double *c, double *d, int num_models){

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_models){

        double x1 = maybe_points1[i*3];
        double y1 = maybe_points1[i*3 + 1];
        double z1 = maybe_points1[i*3 + 2];

        double x2 = maybe_points2[i*3];
        double y2 = maybe_points2[i*3 + 1];
        double z2 = maybe_points2[i*3 + 2];

        double x3 = maybe_points3[i*3];
        double y3 = maybe_points3[i*3 + 1];
        double z3 = maybe_points3[i*3 + 2];

        double a1 = x2 - x1;
        double b1 = y2 - y1;
        double c1 = z2 - z1;
        double a2 = x3 - x1;
        double b2 = y3 - y1;
        double c2 = z3 - z1;
        a[i] = b1 * c2 - b2 * c1;
        b[i] = a2 * c1 - a1 * c2;
        c[i] = a1 * b2 - b1 * a2;
        d[i] = (- a[i] * x1 - b[i] * y1 - c[i] * z1);

    }

}