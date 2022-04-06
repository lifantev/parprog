#include<stdio.h>
#include <stdlib.h>
#include<iostream>
#include<cmath>
#include<vector>
#include<fstream>
#include<mpi.h>


int N = 1000; 
int K = 1000;

double tau = 0.01;
double h = 0.01;

double phi(double x) {
	return 0;
}

double ksi(double t) {
	return t * std::sin(t);
}

double f(double t, double x) {
	return t * std::sin(x + t);
}



double solve_cross(double left, double right, double bottom, double f) {
	return bottom + 2 * tau * (f + (left - right) / (2 * h));
}

double solve_3(double left, double right, double f) {
	return 0.5 * (left + right) + tau * (f + (left - right) / (2 * h));
}

double solve_left_corner(double left, double central, double f) {
	return central + tau * (f + (left - central) / h);
}



void data_send(double** data, MPI_Status& stat, int proc_num, int proc_rank, int N_per_process, int k) {
	double send_right, send_left;

	if (proc_rank == 0) {
		send_right = data[k][N_per_process - 1];
		MPI_Send(&send_right, sizeof(send_right), MPI_BYTE, 1, 1, MPI_COMM_WORLD);
	}
	else if (proc_rank == proc_num - 1) {
		send_left = data[k][0];
		MPI_Send(&send_left, sizeof(send_left), MPI_BYTE, proc_rank - 1, 1, MPI_COMM_WORLD);
	}
	else {
		send_right = data[k][N_per_process - 1];
		MPI_Send(&send_right, sizeof(send_right), MPI_BYTE, proc_rank + 1, 1, MPI_COMM_WORLD);
		send_left = data[k][0];
		MPI_Send(&send_left, sizeof(send_left), MPI_BYTE, proc_rank - 1, 1, MPI_COMM_WORLD);
	}
}

std::pair<double, double> data_recive(double** data, MPI_Status& stat, int proc_num, int proc_rank, int N_per_process, int k) {
	double recv_right, recv_left;

	if (proc_rank == 0) {
		MPI_Recv(&recv_right, sizeof(recv_right), MPI_BYTE, 1, 1, MPI_COMM_WORLD, &stat);
		recv_left = 0;
	}
	else if (proc_rank == proc_num - 1) {
		MPI_Recv(&recv_left, sizeof(recv_left), MPI_BYTE, proc_rank - 1, 1, MPI_COMM_WORLD, &stat);
		recv_right = 0;
	}
	else {
		MPI_Recv(&recv_right, sizeof(recv_right), MPI_BYTE, proc_rank + 1, 1, MPI_COMM_WORLD, &stat);
		MPI_Recv(&recv_left, sizeof(recv_left), MPI_BYTE, proc_rank - 1, 1, MPI_COMM_WORLD, &stat);
	}

	return std::pair<double, double>(recv_left, recv_right);
}



double** new_matrix(int height, int width) {
	double** matrix = new double * [height];
	for (int i = 0; i < height; ++i) {
		matrix[i] = new double[width];
	}
	return matrix;
}

void delete_matrix(double** matrix, int height) {
	for (int i = 0; i < height; ++i) {
		delete[] matrix[i];
	}
	delete[] matrix;
}



int main(int argc, char* argv[]) {

	int proc_num;
	int proc_rank;
	MPI_Status stat;
	double time_d = 0;
	double time_start = 0;
	double time_end = 0;
	int N_per_process, N_add;
	int first_x;

	if (argc >= 2) {
		N = K = atoi(argv[1]);
	}

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &proc_num);

	if (proc_rank == 0) {
		time_start = MPI_Wtime();
	}

	N_per_process = N / proc_num;
	N_add = N % proc_num;

	if (proc_rank < N_add) {
		N_per_process++;
		first_x = N_per_process * proc_rank;
	}
	else {
		first_x = N_add + N_per_process * proc_rank;
	}

	double** data = new_matrix(K, N_per_process);

	for (int n = 0; n < N_per_process; ++n) 	
		data[0][n] = phi((n + first_x) * h);

	data_send(data, stat, proc_num, proc_rank, N_per_process, 0);

	for (int n = 1; n < N_per_process - 1; ++n) 
		data[1][n] = solve_3(data[0][n - 1], data[0][n + 1], f(tau, (first_x + n) * h));

	std::pair<double, double> data_recv = data_recive(data, stat, proc_num, proc_rank, N_per_process, 0);


	if (proc_rank == 0)	
		data[1][0] = ksi(tau);
	else				
		data[1][0] = solve_3(data_recv.first, data[0][1], f(tau, first_x * h));

	if (proc_rank == proc_num - 1)	
		data[1][N_per_process - 1] = solve_left_corner(data[0][N_per_process - 2], data[0][N_per_process - 1], f(tau, h * (first_x + N_per_process - 1)));
	else							
		data[1][N_per_process - 1] = solve_3(data[0][N_per_process - 2], data_recv.second, f(tau, h * (first_x + N_per_process - 1)));


	for (int k = 2; k < K; ++k) {

		data_send(data, stat, proc_num, proc_rank, N_per_process, k - 1);

		for (int n = 1; n < N_per_process - 1; ++n) 
			data[k][n] = solve_cross(data[k - 1][n - 1], data[k - 2][n], data[k - 1][n + 1], f(k * tau, (first_x + n) * h));

		std::pair<double, double> data_recv = data_recive(data, stat, proc_num, proc_rank, N_per_process, k - 1);

		if (proc_rank == 0)	
			data[k][0] = ksi(k * tau);
		else				
			data[k][0] = solve_cross(data_recv.first, data[k - 2][0], data[k - 1][1], f(k * tau, first_x));

		if (proc_rank == proc_num - 1)	
			data[k][N_per_process - 1] = solve_left_corner(data[k - 1][N_per_process - 2], data[k - 1][N_per_process - 1], f(k * tau, (first_x + N_per_process - 1) * h));
		else							
			data[k][N_per_process - 1] = solve_cross(data[k - 1][N_per_process - 2], data[k - 2][N_per_process - 1], data_recv.second, f(k * tau, (first_x + N_per_process - 1) * h));
	}


	if (proc_rank != 0) {
		double* part_data = new double[N_per_process * K];

		for (int iy = 0; iy < K; ++iy)
			for (int ix = 0; ix < N_per_process; ++ix)
				part_data[iy * N_per_process + ix] = data[iy][ix];

		MPI_Send(&N_per_process, sizeof(N_per_process), MPI_BYTE, 0, 1, MPI_COMM_WORLD); 
		MPI_Send(&first_x, sizeof(first_x), MPI_BYTE, 0, 1, MPI_COMM_WORLD);
		MPI_Send(part_data, N_per_process * K * sizeof(double), MPI_BYTE, 0, 1, MPI_COMM_WORLD);

		delete[] part_data;
	}
	else {
		double** result_data = new_matrix(K, N);

		for (int iy = 0; iy < K; ++iy)
			for (int ix = 0; ix < N_per_process; ++ix)
				result_data[iy][ix] = data[iy][ix];

		for (int i = 1; i < proc_num; ++i) {
			int rec_N_per_process;
			MPI_Recv(&rec_N_per_process, sizeof(rec_N_per_process), MPI_BYTE, i, 1, MPI_COMM_WORLD, &stat);

			int proc_first_x;
			MPI_Recv(&proc_first_x, sizeof(proc_first_x), MPI_BYTE, i, 1, MPI_COMM_WORLD, &stat);

			double* reciving_data = new double[rec_N_per_process * K];
			MPI_Recv(reciving_data, rec_N_per_process * K * sizeof(double), MPI_BYTE, i, 1, MPI_COMM_WORLD, &stat);

			for (int iy = 0; iy < K; ++iy)
				for (int ix = 0; ix < rec_N_per_process; ++ix)
					result_data[iy][ix + proc_first_x] = reciving_data[iy * rec_N_per_process + ix];

		}

		if (proc_rank == 0) {
			time_end = MPI_Wtime();
			time_d = time_end - time_start;

			std::cout << "time " << time_d << " seconds" << std::endl;
		}

		std::ofstream out;
		char file_name[] = "out.txt";
		out.open(file_name);
		for (int iy = 0; iy < K; ++iy) {
			for (int ix = 0; ix < N; ++ix) {
				out << result_data[iy][ix] << " ";
			}
			out << std::endl;
		}
		out.close();
		
		std::cout << "data saved in " << file_name << std::endl;

		delete_matrix(result_data, K);
	}


	delete_matrix(data, K);
	MPI_Finalize();
	return 0;
}