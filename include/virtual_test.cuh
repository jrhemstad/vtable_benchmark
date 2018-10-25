#ifndef VIRTUAL_H
#define VIRTUAL_H

template <typename input_type>
double run_cpu_virtual_test(int input_size, int num_iterations);


template <typename input_type>
double run_gpu_virtual_test(int input_size, int num_iterations);


#endif
