#ifndef VIRTUAL_H
#define VIRTUAL_H

template <typename input_type>
std::vector<input_type> run_cpu_virtual_test(std::vector<input_type> left,
                                             std::vector<input_type> right,
                                             const int num_iterations);


template <typename input_type>
std::vector<input_type> run_gpu_virtual_test(std::vector<input_type> left,
                                             std::vector<input_type> right,
                                             const int num_iterations);


#endif
