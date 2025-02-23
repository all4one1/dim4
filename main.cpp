#include "src/Extras.h"
#include "src/ExtrasCuda.h"
#include "src/types_project.h"
#include "src/kernels/convection.h"
#include "src/CuPoisson/CuPoisson.h"
#include "cuda_runtime.h"
//#include <device_launch_parameters.h>
#include <vector>
#include <iostream>
using std::cout;
using std::endl;

__constant__ Configuration dev;
Configuration host;
#include "src/init.h"

RUN_STATE run;

int main(int argc, char** argv)
{
	ReadingFile par("parameters.txt");
	par.reading<int>(run.read_database, "continue", 0);
	par.reading<int>(run.cpu_only, "cpu", 0);
	par.reading<double>(run.timeq_limit, "time_limit", 100);
	par.reading<double>(run.timeq_minimal, "time_minimal", 0);
	par.reading_string(run.note, "note", "");

	GPU_ gpu(0);
	init_parameters(host);
	run.tau = host.tau;
	unsigned int &N = host.N;

	Arrays hostptr, devptr;
	CudaLaunchSetup launch(host.N, host.nx, host.ny, host.nz);
	#define KERNEL1D launch.grid1d, launch.block1d
	FuncTimer ftimer;
	StatValues stat;
	Backup backup("recovery", true);
	std::ofstream w_final, w_temporal;

	allocate_host_arrays({ &hostptr.T, &hostptr.T0, &hostptr.p, &hostptr.p0,
		&hostptr.ux, &hostptr.uy, &hostptr.uz, &hostptr.us,
		&hostptr.vx,&hostptr.vy,&hostptr.vz,&hostptr.vs }, host.N);
	allocate_device_arrays({ &devptr.T, &devptr.T0, &devptr.p, &devptr.p0,
		&devptr.ux, &devptr.uy, &devptr.uz, &devptr.us,
		&devptr.vx, &devptr.vy, &devptr.vz, &devptr.vs }, host.N);
	
	CuPoisson poisson;
	void* args[] = { &devptr.p, &devptr.p0, &devptr.ux, &devptr.uy, &devptr.uz, &devptr.us };
	poisson.set_kernel(dim4::poisson, args, KERNEL1D);
	poisson.set_main_field(devptr.p, devptr.p0, host.N);


	if (run.read_database)
	{
		w_final.open("w_final.dat", std::ofstream::app);
		w_temporal.open("w_temporal.dat", std::ofstream::app);
		backup.read(run.iter, run.timeq, run.call_i, host, { hostptr.vx, hostptr.vy, hostptr.vz, hostptr.vs, hostptr.p, hostptr.T });
		
		copyArrayFromHostToDevice({ devptr.vx, devptr.ux }, hostptr.vx, host.N);
		copyArrayFromHostToDevice({ devptr.vy, devptr.uy }, hostptr.vy, host.N);
		copyArrayFromHostToDevice({ devptr.vz, devptr.uz }, hostptr.vz, host.N);
		copyArrayFromHostToDevice({ devptr.vs, devptr.us }, hostptr.vs, host.N);	
		copyArrayFromHostToDevice({ devptr.p, devptr.p0 }, hostptr.p, host.N);
		copyArrayFromHostToDevice({ devptr.T, devptr.T0 }, hostptr.T, host.N);
	}
	else
	{
		w_final.open("w_final.dat");
		w_temporal.open("w_temporal.dat");
		deleteFilesInDirectory(L"fields");

		init_fields(host, hostptr, devptr);
	}

	//cudaMemcpyToSymbol(dev, &host, sizeof(Configuration), 0, cudaMemcpyHostToDevice);	check << <1, 1 >> > ();	cudaDeviceSynchronize(); pause

	if (run.read_database == 0)		dim4::disturb << <1, 1 >> > (QUARTER, devptr.vy, 0.01);
reset:
	if (run.call_i >= 100) return 0;
	if (run.call_i > 0 && run.iter == 0)	host.Ra += -100;
	
	Checker Check_Vmax(&stat.Vmax, &run.timeq, Checker::ExitType::Relative, "Vmax", 1e-5);


	cudaMemcpyToSymbol(dev, &host, sizeof(Configuration), 0, cudaMemcpyHostToDevice);

	ftimer.start("main");
	while (run.stop_signal == 0)
	{
		run.iter++;
		run.timeq += host.tau;
		if (run.timeq > run.timeq_limit) run.stop_signal = 2;

		ftimer.start("calc");
		{
			dim4::temperature << <KERNEL1D >> > (devptr.T, devptr.T0, devptr.vx, devptr.vy, devptr.vz, devptr.vs);
			swap_one << <KERNEL1D >> > (devptr.T0, devptr.T);
			dim4::quasi_velocity << <KERNEL1D >> > (devptr.T, devptr.ux, devptr.uy, devptr.uz, devptr.us, devptr.vx, devptr.vy, devptr.vz, devptr.vs);
			poisson.solve();
			dim4::velocity_correction << <KERNEL1D >> > (devptr.p, devptr.ux, devptr.uy, devptr.uz, devptr.us, devptr.vx, devptr.vy, devptr.vz, devptr.vs);
		}
		ftimer.end("calc");
		

		/* output and postprocessing */
		//if (run.every(100) || run.iter == 1)
		if (run.every_time(0.5) || run.iter == 1)
		{
			cudaMemcpy(hostptr.p, devptr.p, host.Nbytes, cudaMemcpyDeviceToHost);
			cudaMemcpy(hostptr.T, devptr.T, host.Nbytes, cudaMemcpyDeviceToHost);
			cudaMemcpy(hostptr.vx, devptr.vx, host.Nbytes, cudaMemcpyDeviceToHost);
			cudaMemcpy(hostptr.vy, devptr.vy, host.Nbytes, cudaMemcpyDeviceToHost);
			cudaMemcpy(hostptr.vz, devptr.vz, host.Nbytes, cudaMemcpyDeviceToHost);
			cudaMemcpy(hostptr.vs, devptr.vs, host.Nbytes, cudaMemcpyDeviceToHost);

			velocity_stats(host, hostptr.vx, hostptr.vy, hostptr.vz, hostptr.vs, stat);

			cout << endl << "Ra = " << host.Ra << ", t= " << run.timeq << ", " << run.iter;// << endl;
			cout << ",  T,vy = " << hostptr.T[CENTER] << " " << hostptr.vy[CENTER] << endl;
	
			if (run.iter == 1) w_temporal << "t, time(sec), time(sec)v2, T2, vy2, T4, Vy4, Tmax, VYmax, Vmax, Ek" << endl;
			w_temporal << run.timeq << " " << ftimer.update_and_get("main") << " " << ftimer.get("calc");
			w_temporal << " " << hostptr.T[CENTER] << " " << hostptr.vy[CENTER];
			w_temporal << " " << hostptr.T[QUARTER] << " " << hostptr.vy[QUARTER];
			w_temporal << " " << absmaxsigned(hostptr.T, N) << " " << absmaxsigned(hostptr.vy, N);
			w_temporal << " " << stat.Vmax << " " << stat.Ek;
			w_temporal << endl;

			write_1d_section(run.iter, host, { hostptr.T, hostptr.vy }, "T, vy");



			Check_Vmax.update();
			cout << "Check_Vmax: abs=" << Check_Vmax.dif << ", rel=" << Check_Vmax.dif_rel << endl;
			if (run.timeq >= run.timeq_minimal)
			{
				if (Check_Vmax.ready_to_exit) run.stop_signal = 1;
			}
		}

		/* backup saves */
		if (run.every_time(10))
		{
			backup.save(run.iter, run.timeq, run.call_i, host, { hostptr.vx, hostptr.vy, hostptr.vz, hostptr.vs, hostptr.p, hostptr.T }, "vx, vy, vz, vs, p, T");
		}

		/* upon finish */
		if (run.stop_signal > 0)
		{
			if (run.call_i == 0)	w_final << "Ra, Vmax, Ek, Vy_max, Vs_max, time(sec), t" << endl;

			w_final << host.Ra << " " << stat.Vmax << " " << stat.Ek << " " << stat.Vy << " " << stat.Vs
				<< " " << ftimer.update_and_get("main") << " " << run.timeq
				<< endl;
			
			backup.save(run.iter, run.timeq, run.call_i, host,	{ hostptr.vx, hostptr.vy, hostptr.vz, hostptr.vs, hostptr.p, hostptr.T },	"vx, vy, vz, vs, p, T");
			
			if (run.stop_signal == 1)
			{
				run.iter = 0; run.timeq = 0; run.stop_signal = 0; 
				run.call_i++;
				goto reset;
			}
			if (run.stop_signal == 2) break;
		}


		if (run.stop_signal == -1)
		{
			break;
		}
	}
	return 0;
}
