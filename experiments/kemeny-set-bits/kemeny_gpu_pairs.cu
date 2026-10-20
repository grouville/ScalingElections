#include "kemeny.cuh"
#include <chrono>
#include <iostream>
#include <random>

kemeny_solution_t gpu_baseline(std::uint64_t const*, candidate_index_t, backend_t);
kemeny_solution_t gpu_changed(std::uint64_t const*, candidate_index_t, backend_t);
std::vector<std::uint64_t> gpu_costs_baseline(std::uint64_t const*, candidate_index_t);
std::vector<std::uint64_t> gpu_costs_changed(std::uint64_t const*, candidate_index_t);

void same(kemeny_solution_t const& left, kemeny_solution_t const& right) {
    if (left.score != right.score || left.ranking != right.ranking || left.winners != right.winners ||
        left.multiplicity != right.multiplicity)
        throw std::runtime_error("Kemeny scores, rankings, winners or multiplicity disagree");
}

void verify() {
    std::mt19937_64 random(42);
    for (candidate_index_t n = 1; n <= 10; ++n)
        for (bool tied : {false, true}) {
            std::vector<std::uint64_t> counts(std::size_t(n) * n);
            for (auto& value : counts) value = tied ? 4 : random() % 8;
            for (candidate_index_t c = 0; c < n; ++c) counts[c * n + c] = 0;
            auto expected = gpu_baseline(counts.data(), n, backend_t::cpu_k);
            same(expected, gpu_baseline(counts.data(), n, backend_t::gpu_k));
            same(expected, gpu_changed(counts.data(), n, backend_t::gpu_k));
            auto preferences = square_view<std::uint64_t const, candidate_index_t>(counts.data(), n, n);
            kemeny_sums<std::uint64_t> sums(preferences);
            auto oracle = compute_kemeny_costs_cpu<std::uint64_t>(preferences, sums);
            if (gpu_costs_baseline(counts.data(), n) != oracle || gpu_costs_changed(counts.data(), n) != oracle)
                throw std::runtime_error("GPU cost tables disagree with the CPU oracle");
        }
    std::cerr << "PASS: both GPU kernels match CPU full cost tables and metadata for 20 small profiles.\n";
}

int main(int argc, char** argv) {
    try {
        auto properties = current_device_properties_();
        std::cerr << "Kemeny GPU device=" << properties.name << " cc=" << properties.major << '.' << properties.minor
                  << " sms=" << properties.multiProcessorCount << '\n';
        verify();
        if (argc > 1 && std::string_view(argv[1]) == "--verify-only") return 0;
        std::mt19937_64 random(42);
        std::cout << "kind,candidates,repeat,path,nanoseconds\n";
        for (candidate_index_t n : {18, 20, 22, 24}) {
            std::vector<std::uint64_t> counts(std::size_t(n) * n);
            for (auto& value : counts) value = random() % 350000001;
            for (candidate_index_t c = 0; c < n; ++c) counts[c * n + c] = 0;
            auto expected = gpu_baseline(counts.data(), n, backend_t::gpu_k);
            for (bool control : {true, false}) {
                auto other = control ? gpu_baseline : gpu_changed;
                for (int warmup = 0; warmup < 2; ++warmup) {
                    same(expected, gpu_baseline(counts.data(), n, backend_t::gpu_k));
                    same(expected, other(counts.data(), n, backend_t::gpu_k));
                }
                for (int repeat = 0; repeat < 7; ++repeat)
                    for (int half = 0; half < 2; ++half) {
                        bool changed = ((repeat + half) % 2) != 0;
                        auto call = changed ? other : gpu_baseline;
                        auto start = std::chrono::steady_clock::now();
                        auto actual = call(counts.data(), n, backend_t::gpu_k);
                        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - start).count();
                        same(expected, actual);
                        std::cout << (control ? "control" : "experiment") << ',' << n << ',' << repeat << ','
                                  << (changed ? "candidate" : "baseline") << ',' << elapsed << '\n';
                    }
            }
        }
        return 0;
    }
    catch (std::exception const& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
