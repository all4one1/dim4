#pragma once
#include "cuda_runtime.h"
#include <device_launch_parameters.h>
#include "../types_project.h"


extern __constant__ Configuration dev;


__global__ void check();
__global__ void swap_one(double* f_old, double* f_new);
__global__ void swap_two(double* f_old, double* f_new, double* f2_old, double* f2_new);
__global__ void swap_three(double* f_old, double* f_new, double* f2_old, double* f2_new, double* f3_old, double* f3_new);

namespace dim4
{
	__global__ void disturb(unsigned int q, double* f, double val);
	__global__ void check4(unsigned int q, double *T);
	__global__ void temperature_check(double* T, double* T0);
	__global__ void temperature(double* T, double* T0, double* vx, double* vy, double* vz, double* vs);
	__global__ void quasi_velocity(double* T,
		double* ux, double* uy, double* uz, double* us,
		double* vx, double* vy, double* vz, double* vs);
	__global__ void velocity_correction(double* p, 
		double* ux, double* uy, double* uz, double *us,
		double* vx, double* vy, double* vz, double *vs);

	__global__ void poisson(double* p, double* p0, double* ux, double* uy, double* uz, double* us);
}