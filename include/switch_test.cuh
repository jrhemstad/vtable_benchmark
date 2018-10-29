#ifndef SWITCH_H
#define SWITCH_H
#include <vector>

template <typename input_type>
std::vector<input_type> run_cpu_switch_test(std::vector<input_type> left,
                                            std::vector<input_type> right,
                                            const int num_iterations);

template <typename input_type>
double run_gpu_switch_test(const int input_size, const int num_iterations);

#endif
