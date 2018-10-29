#ifndef BASELINE_H
#define BASELINE_H
#include <vector>

template <typename input_type>
std::vector<input_type> run_cpu_baseline(std::vector<input_type> const& left,
                                         std::vector<input_type> const& right,
                                         const int num_iterations, 
                                         bool record_time = true );

template <typename input_type>
std::vector<input_type> run_gpu_baseline(std::vector<input_type> const& left,
                                         std::vector<input_type> const& right,
                                         const int num_iterations,
                                         bool record_time = true);


#endif
