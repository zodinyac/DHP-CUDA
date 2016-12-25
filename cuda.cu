#include <cstdio>
#include <cstdlib>
#include "cuda.h"
#include "cuda_utils.h"

static double *cuda_acc;

void cuda_init(int rank)
{
    int device_count;
    cudaGetDeviceCount(&device_count);
    
    array(int) devices = array_new(ARRAY_VECTOR, int);
    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp device_prop;
        cudaGetDeviceProperties(&device_prop, i);
        if (device_prop.major >= 2) {
            array_push(&devices, i);
        }
    }
    
    if (array_size(devices) == 0) {
        exit(1);
    }
    
    int selected_device = devices[0];
    cudaSetDevice(selected_device);
    
    //printf("(%d) cuda selected device: %d\n", rank, selected_device);
    
    array_delete(&devices);
    
    // accumulator for reduce operations
    cudaMalloc(&cuda_acc, sizeof(double));
}

void cuda_finalize()
{
    cudaFree(cuda_acc);
}

void *cuda_create_load_array_to_device(array(double) array, int (*indexes)[D], int mesh_n[D])
{
    void *cuda_array;
    
    cudaMalloc(&cuda_array, array_size(array) * array_item_size(array));
    cuda_load_array_to_device(cuda_array, array, indexes, mesh_n);
    
    return cuda_array;
}

void cuda_load_array_to_device(void *cuda_array, array(double) array, int (*indexes)[D], int mesh_n[D])
{
    for (int j = 0; j < CALC_JC(indexes); j++) {
        int ic = CALC_IC(indexes);
        cudaMemcpy((double *)cuda_array + j * ic, array + (j + indexes[Y][START] - 1) * mesh_n[X] + (indexes[X][START] - 1), ic * array_item_size(array), cudaMemcpyHostToDevice);
    }
}

void cuda_load_array_from_device(void *cuda_array, array(double) array, int (*indexes)[D], int mesh_n[D])
{
    for (int j = 0; j < CALC_JC(indexes) - 2; j++) {
        int ic = CALC_IC(indexes);
        cudaMemcpy(array + (j + indexes[Y][START]) * mesh_n[X] + indexes[X][START], (double *)cuda_array + (j + 1) * ic + 1, (ic - 2) * array_item_size(array), cudaMemcpyDeviceToHost);
    }
}

void cuda_delete_array(void *cuda_array)
{
    cudaFree(cuda_array);
}

__global__ void calc_residual_vector(double *cuda_res_vect, double *cuda_sol_vect, double *cuda_rhs_vect, int ic, int jc, double hx, double hy)
{
    if (INDEX_X > 0 && INDEX_Y > 0 && INDEX_X < ic - 1 && INDEX_Y < jc - 1) {
        double lp;
        if (cuda_sol_vect == NULL) {
            lp = 0.0;
        } else {
            lp = left_part(cuda_sol_vect, INDEX_X, INDEX_Y);
        }
        cuda_res_vect[INDEX(ic)] = lp - cuda_rhs_vect[INDEX(ic)];
    }
}

void cuda_calc_residual_vector(void *cuda_res_vect, void *cuda_sol_vect, void *cuda_rhs_vect, int (*indexes)[D], double h[D])
{
    int ic = CALC_IC(indexes);
    int jc = CALC_JC(indexes);
    
    calc_residual_vector<<<GRID_DIM(ic, jc), BLOCK_DIM>>>((double *)cuda_res_vect, (double *)cuda_sol_vect, (double *)cuda_rhs_vect, ic, jc, h[X], h[Y]);
}

__global__ void calc_product(double *cuda_vect1, double *cuda_vect2, double *cuda_acc, int ic, int jc)
{
    extern __shared__ double shared[];
    
    double tacc = 0.0;
    if (INDEX_X > 0 && INDEX_Y > 0 && INDEX_X < ic - 1 && INDEX_Y < jc - 1) {
        tacc = cuda_vect1[INDEX(ic)] * cuda_vect2[INDEX(ic)];
    }
    
    cuda_reduce_sum(shared, THREAD_IDX, tacc);
    if (THREAD_IDX == 0) {
        atomicAdd(cuda_acc, shared[0]);
	}
}

void cuda_calc_product(void *cuda_vect1, void *cuda_vect2, double *acc, int (*indexes)[D])
{
    cudaMemset(cuda_acc, 0, sizeof(double));
    
    int ic = CALC_IC(indexes);
    int jc = CALC_JC(indexes);
    
    calc_product<<<GRID_DIM(ic, jc), BLOCK_DIM, BLOCK_SIZE * sizeof(double)>>>((double *)cuda_vect1, (double *)cuda_vect2, (double *)cuda_acc, ic, jc);

    cudaMemcpy(acc, cuda_acc, sizeof(double), cudaMemcpyDeviceToHost);
}

__global__ void calc_Aproduct(double *cuda_vect1, double *cuda_vect2, double *cuda_acc, int ic, int jc, double hx, double hy)
{
    extern __shared__ double shared[];
    
    double tacc = 0.0;
    if (INDEX_X > 0 && INDEX_Y > 0 && INDEX_X < ic - 1 && INDEX_Y < jc - 1) {
        tacc = left_part(cuda_vect1, INDEX_X, INDEX_Y) * cuda_vect2[INDEX(ic)];
    }
    
    cuda_reduce_sum(shared, THREAD_IDX, tacc);
    if (THREAD_IDX == 0) {
        atomicAdd(cuda_acc, shared[0]);
	}
}

void cuda_calc_Aproduct(void *cuda_vect1, void *cuda_vect2, double *acc, int (*indexes)[D], double h[D])
{
    cudaMemset(cuda_acc, 0, sizeof(double));
    
    int ic = CALC_IC(indexes);
    int jc = CALC_JC(indexes);
    
    calc_Aproduct<<<GRID_DIM(ic, jc), BLOCK_DIM, BLOCK_SIZE * sizeof(double)>>>((double *)cuda_vect1, (double *)cuda_vect2, (double *)cuda_acc, ic, jc, h[X], h[Y]);

    cudaMemcpy(acc, cuda_acc, sizeof(double), cudaMemcpyDeviceToHost);
}

__global__ void calc_it0_solution_vector(double *cuda_sol_vect, double *cuda_res_vect, int ic, int jc, double tau)
{
    if (INDEX_X > 0 && INDEX_Y > 0 && INDEX_X < ic - 1 && INDEX_Y < jc - 1) {
        cuda_sol_vect[INDEX(ic)] = -tau * cuda_res_vect[INDEX(ic)];
    }
}

void cuda_calc_it0_solution_vector(void *cuda_sol_vect, void *cuda_res_vect, int (*indexes)[D], double tau)
{
    int ic = CALC_IC(indexes);
    int jc = CALC_JC(indexes);
    
    calc_it0_solution_vector<<<GRID_DIM(ic, jc), BLOCK_DIM>>>((double *)cuda_sol_vect, (double *)cuda_res_vect, ic, jc, tau);
}

__global__ void calc_itn_solution_vector(double *cuda_sol_vect, double *cuda_basis_vect, double *cuda_acc, int ic, int jc, double tau)
{
    extern __shared__ double shared[];
    
    double terr = 0.0;
    if (INDEX_X > 0 && INDEX_Y > 0 && INDEX_X < ic - 1 && INDEX_Y < jc - 1) {
        double new_value = cuda_sol_vect[INDEX(ic)] - tau * cuda_basis_vect[INDEX(ic)];
        terr = fabs(new_value - cuda_sol_vect[INDEX(ic)]);
        cuda_sol_vect[INDEX(ic)] = new_value;
    }
    
    cuda_reduce_max(shared, THREAD_IDX, terr);
    if (THREAD_IDX == 0) {
        atomicMax(cuda_acc, shared[0]);
	}
}

void cuda_calc_itn_solution_vector(void *cuda_sol_vect, void *cuda_basis_vect, double *err, int (*indexes)[D], double tau)
{
    cudaMemset(cuda_acc, 0, sizeof(double));
    
    int ic = CALC_IC(indexes);
    int jc = CALC_JC(indexes);
    
    calc_itn_solution_vector<<<GRID_DIM(ic, jc), BLOCK_DIM, BLOCK_SIZE * sizeof(double)>>>((double *)cuda_sol_vect, (double *)cuda_basis_vect, (double *)cuda_acc, ic, jc, tau);
    
    cudaMemcpy(err, cuda_acc, sizeof(double), cudaMemcpyDeviceToHost);
}

__global__ void calc_basis_vector(double *cuda_basis_vect, double *cuda_res_vect, int ic, int jc, double alpha)
{
    if (INDEX_X > 0 && INDEX_Y > 0 && INDEX_X < ic - 1 && INDEX_Y < jc - 1) {
        cuda_basis_vect[INDEX(ic)] = cuda_res_vect[INDEX(ic)] - alpha * cuda_basis_vect[INDEX(ic)];
    }
}

void cuda_calc_basis_vector(void *cuda_basis_vect, void *cuda_res_vect, int (*indexes)[D], double alpha)
{
    int ic = CALC_IC(indexes);
    int jc = CALC_JC(indexes);
    
    calc_basis_vector<<<GRID_DIM(ic, jc), BLOCK_DIM>>>((double *)cuda_basis_vect, (double *)cuda_res_vect, ic, jc, alpha);
}

__global__ void calc_error(double *cuda_sol_vect, double *cuda_acc, int ic, int jc, double hx, double hy, double offset_x, double offset_y)
{
    extern __shared__ double shared[];
    
    double terr = 0.0;
    if (INDEX_X > 0 && INDEX_Y > 0 && INDEX_X < ic - 1 && INDEX_Y < jc - 1) {
        double bv = boundary_value((INDEX_X + offset_x - 1) * hx, (INDEX_Y + offset_y - 1) * hy);
        terr = fabs(bv - cuda_sol_vect[INDEX(ic)]);
    }
    
    cuda_reduce_max(shared, THREAD_IDX, terr);
    if (THREAD_IDX == 0) {
        atomicMax(cuda_acc, shared[0]);
	}
}

void cuda_calc_error(void *cuda_sol_vect, double *err, int (*indexes)[D], double h[D])
{
    cudaMemset(cuda_acc, 0, sizeof(double));
    
    int ic = CALC_IC(indexes);
    int jc = CALC_JC(indexes);
    
    calc_error<<<GRID_DIM(ic, jc), BLOCK_DIM, BLOCK_SIZE * sizeof(double)>>>((double *)cuda_sol_vect, (double *)cuda_acc, ic, jc, h[X], h[Y], indexes[X][START], indexes[Y][START]);
    
    cudaMemcpy(err, cuda_acc, sizeof(double), cudaMemcpyDeviceToHost);
}
