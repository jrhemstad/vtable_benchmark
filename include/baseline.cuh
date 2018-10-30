#ifndef BASELINE_H
#define BASELINE_H
#include <vector>

template <typename input_type>
std::vector<input_type> run_cpu_baseline(std::vector<input_type>  left,
                                         std::vector<input_type>  right,
                                         const int num_iterations);

template <typename input_type>
std::vector<input_type> run_gpu_baseline(std::vector<input_type>  left,
                                         std::vector<input_type>  right,
                                         const int num_iterations);


#endif
