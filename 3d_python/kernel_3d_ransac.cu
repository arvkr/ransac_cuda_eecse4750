// Calculates the error distance of a point for one plane model. Each thread is responsible for one point
__global__ void distance(const double *points, double *output, double a, double b, double c, double d, int N){

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N){

        double x0 = points[i*3];
        double y0 = points[i*3 + 1];
        double z0 = points[i*3 + 2];

        // intersection point with the model
        double numer = abs(((a * x0) + (b * y0) + (c * z0) + d));
        double denom = sqrt((a * a) + (b * b) + (c * c));
        double dist = numer / denom;

        output[i] = dist;
    } 
}

// Given 3 points, find the equation of plane passing through it. Each thread is responsible for one plane equation
__global__ void find_plane_model(const double *maybe_points1, const double *maybe_points2, 
                                const double *maybe_points3, double *a, double *b,
                                double *c, double *d, int num_models){

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

        // a, b, c, d - plane equation parameters
        a[i] = b1 * c2 - b2 * c1;
        b[i] = a2 * c1 - a1 * c2;
        c[i] = a1 * b2 - b1 * a2;
        d[i] = (- a[i] * x1 - b[i] * y1 - c[i] * z1);

    }

}

// More advanced version where error distance is calculated for all points and all models in parallel.
__global__ void distance_3d_model_parallel_large(const double *points, double *output, double *a_all, double *b_all, double *c_all, double *d_all, int num_samples){

    int tx = threadIdx.x;
    int point_idx = blockIdx.x * blockDim.x + tx;

    // There are a total of blockDim.y models
    double a = a_all[blockIdx.y];
    double b = b_all[blockIdx.y];
    double c = c_all[blockIdx.y];
    double d = d_all[blockIdx.y];

    // The output index for error calculation of one point for one model
    int op_idx = blockIdx.y*num_samples + point_idx;

    if (point_idx < num_samples){

        double x0 = points[point_idx*3];
        double y0 = points[point_idx*3 + 1];
        double z0 = points[point_idx*3 + 2];

        // intersection point with the model
        double numer = abs(((a * x0) + (b * y0) + (c * z0) + d));
        double denom = sqrt((a * a) + (b * b) + (c * c));
        double dist = numer / denom;
        output[op_idx] = dist;
    } 
}