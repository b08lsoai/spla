/**********************************************************************************/
/* This file is part of spla project */
/* https://github.com/SparseLinearAlgebra/spla */
/**********************************************************************************/
/* MIT License */
/*                                                                                */
/* Copyright (c) 2023 SparseLinearAlgebra */
/*                                                                                */
/* Permission is hereby granted, free of charge, to any person obtaining a copy
 */
/* of this software and associated documentation files (the "Software"), to deal
 */
/* in the Software without restriction, including without limitation the rights
 */
/* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell */
/* copies of the Software, and to permit persons to whom the Software is */
/* furnished to do so, subject to the following conditions: */
/*                                                                                */
/* The above copyright notice and this permission notice shall be included in
 * all */
/* copies or substantial portions of the Software. */
/*                                                                                */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR */
/* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, */
/* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 */
/* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER */
/* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 */
/* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 */
/* SOFTWARE. */
/**********************************************************************************/

#include "common.hpp"
#include "options.hpp"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <spla.hpp>
#include <vector>

namespace {

    double mean(const std::vector<double>& v) {
        return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
    }

    double stdev(const std::vector<double>& v) {
        if (v.size() < 2) return 0.0;// как statistics.stdev, но не бросаем исключение
        double m      = mean(v);
        double sq_sum = 0.0;
        for (double x : v) sq_sum += (x - m) * (x - m);
        return std::sqrt(sq_sum / static_cast<double>(v.size() - 1));// sample stdev, n-1
    }

    void write_csv_row(std::ofstream& out, const std::string& tool,
                       const std::string&         graph_name,
                       const std::vector<double>& times_ms,
                       double                     mst_weight) {
        double avg = mean(times_ms);
        double sd  = stdev(times_ms);
        double mn  = *std::min_element(times_ms.begin(), times_ms.end());
        double mx  = *std::max_element(times_ms.begin(), times_ms.end());
        out << tool << "," << graph_name << "," << avg << "," << sd << "," << mn << "," << mx << "," << mst_weight << "\n";
    }

}// namespace

int main(int argc, const char* const* argv) {
    auto options = make_options(
            "mst", "Boruvka's Minimum Spanning Tree algorithm with spla library");

    cxxopts::ParseResult args;
    int                  ret;

    if (parse_options(argc, argv, options, args, ret)) {
        std::cerr << "failed to parse options" << std::endl;
        return ret;
    }

    spla::Timer     timer_total;
    spla::Timer     timer_gpu;
    spla::Timer     timer_cpu;
    spla::Timer     timer_ref;
    spla::MtxLoader loader;

    timer_total.start();

    if (!loader.load(args["mtxpath"].as<std::string>())) {
        std::cerr << "failed to load graph";
        return 1;
    }

    std::string    acc_info;
    spla::Library* library = spla::Library::get();

    library->set_platform(args["platform"].as<int>());
    library->set_device(args["device"].as<int>());
    library->set_queues_count(1);
    library->get_accelerator_info(acc_info);
    std::cout << "env: " << acc_info << std::endl;

    const spla::uint N = loader.get_n_rows();
    auto             S = spla::Matrix::make(N, N, spla::PAIR);

    const auto& Ai = loader.get_Ai();
    const auto& Aj = loader.get_Aj();
    const auto& Aw = loader.get_Aw();

    for (std::size_t k = 0; k < loader.get_n_values(); ++k) {
        S->set_pair(Ai[k], Aj[k], spla::T_PAIR(Aw[k], Aj[k]));
    }

    auto T_gpu = spla::Matrix::make(N, N, spla::FLOAT);
    auto T_cpu = spla::Matrix::make(N, N, spla::FLOAT);

    auto desc = spla::Descriptor::make();

    const int n_iters = args["niters"].as<int>();
    const int n_warm  = 10;

    double total_weight_gpu = 0.0;
    double total_weight_cpu = 0.0;

    std::vector<double> cpu_times_ms, gpu_times_ms;

    if (args["run-cpu"].as<bool>()) {
        library->set_force_no_acceleration(true);

        // warm up
        for (int i = 0; i < n_warm; ++i) {
            T_cpu->clear();
            S = spla::Matrix::make(N, N, spla::PAIR);
            for (std::size_t k = 0; k < loader.get_n_values(); ++k) {
                S->set_pair(Ai[k], Aj[k], spla::T_PAIR(Aw[k], Aj[k]));
            }
            spla::mst(T_cpu, S, desc, nullptr);
        }

        for (int i = 0; i < n_iters; ++i) {
            spla::Timer t_cpu;
            T_cpu->clear();
            S = spla::Matrix::make(N, N, spla::PAIR);
            for (std::size_t k = 0; k < loader.get_n_values(); ++k) {
                S->set_pair(Ai[k], Aj[k], spla::T_PAIR(Aw[k], Aj[k]));
            }
            timer_cpu.lap_begin();
            t_cpu.start();
            spla::mst(T_cpu, S, desc, nullptr);
            t_cpu.stop();
            timer_cpu.lap_end();
            cpu_times_ms.push_back(t_cpu.get_elapsed_ms());
        }

        total_weight_cpu = 0;
        for (spla::uint i = 0; i < N; ++i) {
            for (spla::uint j = i + 1; j < N; ++j) {
                float w;
                T_cpu->get_float(i, j, w);
                if (w != 0.0) {
                    total_weight_cpu += w;
                }
            }
        }

        std::cout << "CPU MST total weight: " << total_weight_cpu << std::endl;
    }

    if (args["run-gpu"].as<bool>()) {
        library->set_force_no_acceleration(false);

        // warm up
        for (int i = 0; i < n_warm; ++i) {
            T_gpu->clear();
            S = spla::Matrix::make(N, N, spla::PAIR);
            for (std::size_t k = 0; k < loader.get_n_values(); ++k) {
                S->set_pair(Ai[k], Aj[k], spla::T_PAIR(Aw[k], Aj[k]));
            }
            spla::mst(T_gpu, S, desc, nullptr);
        }

        for (int i = 0; i < n_iters; ++i) {
            spla::Timer t_gpu;
            T_gpu->clear();
            S = spla::Matrix::make(N, N, spla::PAIR);
            for (std::size_t k = 0; k < loader.get_n_values(); ++k) {
                S->set_pair(Ai[k], Aj[k], spla::T_PAIR(Aw[k], Aj[k]));
            }
            timer_gpu.lap_begin();
            t_gpu.start();
            spla::mst(T_gpu, S, desc, nullptr);
            t_gpu.stop();
            timer_gpu.lap_end();
            gpu_times_ms.push_back(t_gpu.get_elapsed_ms());
        }

        total_weight_gpu = 0;
        for (spla::uint i = 0; i < N; ++i) {
            for (spla::uint j = i + 1; j < N; ++j) {
                float w;
                T_gpu->get_float(i, j, w);
                if (w != 0.0) {
                    total_weight_gpu += w;
                }
            }
        }

        std::cout << "GPU MST total weight: " << total_weight_gpu << std::endl;
    }

    std::ofstream csv_out("mst_profile.csv");
    csv_out << "tool,graph,avg,sd,min,max,mst_weight\n";
    const std::string& graph_name = "nemeth15";

    if (!cpu_times_ms.empty()) {
        write_csv_row(csv_out, "spla_cpu", graph_name, cpu_times_ms, total_weight_cpu);
    }
    if (!gpu_times_ms.empty()) {
        write_csv_row(csv_out, "spla_gpu", graph_name, gpu_times_ms, total_weight_gpu);
    }
    csv_out.close();

    spla::Library::get()->finalize();

    timer_total.stop();

    std::cout << "\n=== Timing Results ===" << std::endl;
    std::cout << "total(ms):" << timer_total.get_elapsed_ms() << std::endl;
    std::cout << "cpu(ms): ";
    timer_cpu.print();
    std::cout << std::endl;
    std::cout << "gpu(ms): ";
    timer_gpu.print();
    std::cout << std::endl;

    return 0;
}