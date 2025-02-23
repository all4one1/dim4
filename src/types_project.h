#pragma once

enum class Side
{
	center,
	west, east,
	south, north,
	front, back
};


enum bc_type
{
	closed = 0,
	periodic = 1,
	Dirichlet = 2,
	Neumann = 3
};

enum class MathBoundary
{
	Dirichlet,
	Neumann
};


struct Configuration
{
	double hx, hy, hz, hs, Sx, Sy, Sz, Ss, Lx, Ly, Lz, Ls, dV;
	double tau, tau_p;
	double alpha, beta, gamma;
	double sinA, cosA, sinB, cosB, sinG, cosG;
	double vibr_x, vibr_y, vibr_z, vibr_s;
	double grav_x, grav_y, grav_z, grav_s;
	double density_x, density_y, density_z, density_s;
	double Sc11, Sc12, Sc21, Sc22, psi1, psi2, psiS, Q, Ra, Rav, K, Pr, Le;
	double Re, Pe, M, A, Ca, Gr;
	unsigned int dim, nx, ny, nz, ns, N, offset, offset2, offset3, Nbytes;
	bc_type xbc, ybc, zbc, sbc;
};


struct StatValues 
{
	double ksi_max = 0, ksi_sum = 0, omega_max = 0, omega_sum = 0, C_sum = 0, C_sum_signed = 0;
	double NuTop = 0, NuDown = 0, ShrTop = 0, ShrDown = 0;
	double Vmax = 0, Vx = 0, Vy = 0, Vz = 0, Vs = 0, Ek = 0;
	double Cu = 0, Pe = 0;
};


struct Arrays
{
	double* p, * p0, * ux, * uy, * uz, *us, * vx, * vy, * vz, *vs, * buffer, *buffer2, *rhs;
	double* T, * T0, * C, * C0, * C2, * C20, * C3, * C30, * mu, * mu0;
	double* omega, * omega0, * ksi, * ksi0;
	double* src_x, * src_y, * src_z;
	size_t total_bytes, Nbytes, N;
};

