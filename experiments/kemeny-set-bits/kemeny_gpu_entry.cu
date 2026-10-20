#include <kemeny.cuh>

kemeny_solution_t ENTRY_POINT(std::uint64_t const* data, candidate_index_t n, backend_t backend) {
    auto preferences = square_view<std::uint64_t const, candidate_index_t>(data, n, n);
    return compute_kemeny_ranking<std::uint64_t>(preferences, backend);
}

std::vector<std::uint64_t> COSTS_ENTRY(std::uint64_t const* data, candidate_index_t n) {
    auto preferences = square_view<std::uint64_t const, candidate_index_t>(data, n, n);
    kemeny_sums<std::uint64_t> sums(preferences);
    auto costs = compute_kemeny_costs_gpu<std::uint64_t>(preferences, sums);
    return {costs.data(), costs.data() + costs.size()};
}
