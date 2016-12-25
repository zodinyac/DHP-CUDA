#ifndef CUDA_H
#define CUDA_H

#include "array.h"
#include "definitions.h"

#ifdef __cplusplus
extern "C" {
#endif
    void cuda_init(int rank);
    void cuda_finalize();
    
    void *cuda_create_load_array_to_device(array(double) array, int (*indexes)[D], int mesh_n[D]);
    void cuda_load_array_to_device(void *cuda_array, array(double) array, int (*indexes)[D], int mesh_n[D]);
    void cuda_load_array_from_device(void *cuda_array, array(double) array, int (*indexes)[D], int mesh_n[D]);
    void cuda_delete_array(void *cuda_array);
    
    void cuda_calc_residual_vector(void *cuda_res_vect, void *cuda_sol_vect, void *cuda_rhs_vect, int (*indexes)[D], double h[D]);
    void cuda_calc_product(void *cuda_vect1, void *cuda_vect2, double *acc, int (*indexes)[D]);
    void cuda_calc_Aproduct(void *cuda_vect1, void *cuda_vect2, double *acc, int (*indexes)[D], double h[D]);
    
    void cuda_calc_it0_solution_vector(void *cuda_sol_vect, void *cuda_res_vect, int (*indexes)[D], double tau);
    void cuda_calc_itn_solution_vector(void *cuda_sol_vect, void *cuda_basis_vect, double *err, int (*indexes)[D], double tau);
    
    void cuda_calc_basis_vector(void *cuda_basis_vect, void *cuda_res_vect, int (*indexes)[D], double alpha);
    
    void cuda_calc_error(void *cuda_sol_vect, double *err, int (*indexes)[D], double h[D]);
#ifdef __cplusplus
}
#endif

#endif
