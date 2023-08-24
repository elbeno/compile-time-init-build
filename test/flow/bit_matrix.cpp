#include <flow/bit_matrix.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <string_view>

TEST_CASE("default construct", "[bit matrix]") {
    flow::bit_matrix<32> m{};
    static_assert(sizeof(m) == 128);
}

TEST_CASE("size", "[bit matrix]") {
    static_assert(sizeof(flow::bit_matrix<1>) == 8);
    static_assert(sizeof(flow::bit_matrix<2>) == 8);
    static_assert(sizeof(flow::bit_matrix<8>) == 8);
    static_assert(sizeof(flow::bit_matrix<9>) == 32);
    static_assert(sizeof(flow::bit_matrix<16>) == 32);
    static_assert(sizeof(flow::bit_matrix<17>) == 128);
}

TEST_CASE("index compute (block 0,0 extrema)", "[bit matrix]") {
    {
        auto const [ai, bi] = flow::detail::compute_index<8, 2>(0, 0);
        CHECK(ai == 0);
        CHECK(bi == 0);
    }
    {
        auto const [ai, bi] = flow::detail::compute_index<8, 2>(0, 7);
        CHECK(ai == 0);
        CHECK(bi == 7);
    }
    {
        auto const [ai, bi] = flow::detail::compute_index<8, 2>(7, 0);
        CHECK(ai == 0);
        CHECK(bi == 56);
    }
    {
        auto const [ai, bi] = flow::detail::compute_index<8, 2>(7, 7);
        CHECK(ai == 0);
        CHECK(bi == 63);
    }
}

TEST_CASE("index compute (block 0,1 extrema)", "[bit matrix]") {
    {
        auto const [ai, bi] = flow::detail::compute_index<8, 2>(0, 8);
        CHECK(ai == 1);
        CHECK(bi == 0);
    }
    {
        auto const [ai, bi] = flow::detail::compute_index<8, 2>(0, 15);
        CHECK(ai == 1);
        CHECK(bi == 7);
    }
    {
        auto const [ai, bi] = flow::detail::compute_index<8, 2>(7, 8);
        CHECK(ai == 1);
        CHECK(bi == 56);
    }
    {
        auto const [ai, bi] = flow::detail::compute_index<8, 2>(7, 15);
        CHECK(ai == 1);
        CHECK(bi == 63);
    }
}

TEST_CASE("index compute (block 1,0 extrema)", "[bit matrix]") {
    {
        auto const [ai, bi] = flow::detail::compute_index<8, 2>(8, 0);
        CHECK(ai == 2);
        CHECK(bi == 0);
    }
    {
        auto const [ai, bi] = flow::detail::compute_index<8, 2>(8, 7);
        CHECK(ai == 2);
        CHECK(bi == 7);
    }
    {
        auto const [ai, bi] = flow::detail::compute_index<8, 2>(15, 0);
        CHECK(ai == 2);
        CHECK(bi == 56);
    }
    {
        auto const [ai, bi] = flow::detail::compute_index<8, 2>(15, 7);
        CHECK(ai == 2);
        CHECK(bi == 63);
    }
}

TEST_CASE("index compute (block 1,1 extrema)", "[bit matrix]") {
    {
        auto const [ai, bi] = flow::detail::compute_index<8, 2>(8, 8);
        CHECK(ai == 3);
        CHECK(bi == 0);
    }
    {
        auto const [ai, bi] = flow::detail::compute_index<8, 2>(8, 15);
        CHECK(ai == 3);
        CHECK(bi == 7);
    }
    {
        auto const [ai, bi] = flow::detail::compute_index<8, 2>(15, 8);
        CHECK(ai == 3);
        CHECK(bi == 56);
    }
    {
        auto const [ai, bi] = flow::detail::compute_index<8, 2>(15, 15);
        CHECK(ai == 3);
        CHECK(bi == 63);
    }
}

TEST_CASE("indexing", "[bit matrix]") {
    constexpr flow::bit_matrix<2> m{};
    static_assert(not m.index(0, 0));
}

TEST_CASE("set bit", "[bit matrix]") {
    flow::bit_matrix<2> m{};
    m.set(0, 0);
    CHECK(m.index(0, 0));
    CHECK(not m.index(0, 1));
    CHECK(not m.index(1, 0));
    CHECK(not m.index(1, 1));
}

TEST_CASE("set whole matrix", "[bit matrix]") {
    flow::bit_matrix<8> m{};
    m.set();
    for (auto i = std::size_t{}; i < 8u; ++i) {
        for (auto j = std::size_t{}; j < 8u; ++j) {
            CHECK(m.index(i, j));
        }
    }
}

TEST_CASE("clear whole matrix", "[bit matrix]") {
    flow::bit_matrix<16> m{};
    m.set();
    m.clear();
    for (auto i = std::size_t{}; i < 16u; ++i) {
        for (auto j = std::size_t{}; j < 16u; ++j) {
            CHECK(not m.index(i, j));
        }
    }
}

TEST_CASE("equality", "[bit matrix]") {
    auto m = flow::bit_matrix<12>{};
    m.set();
    auto n = flow::bit_matrix<12>{};
    for (auto i = std::size_t{}; i < 12u; ++i) {
        for (auto j = std::size_t{}; j < 12u; ++j) {
            n.set(i, j);
        }
    }
    CHECK(m == n);
}

TEST_CASE("identity matrix", "[bit matrix]") {
    auto m = flow::bit_matrix<16>::identity();
    CHECK(not m.index(0, 1));
    CHECK(not m.index(1, 0));
    for (auto i = std::size_t{}; i < 16u; ++i) {
        CHECK(m.index(i, i));
    }
}

TEST_CASE("create from string_view"
          "[bit_matrix]") {
    using namespace std::string_view_literals;
    constexpr flow::bit_matrix<3> m{"100"
                                    "010"
                                    "001"sv};
    static_assert(m == flow::bit_matrix<3>::identity());
}

TEST_CASE("or", "[bit matrix]") {
    using namespace std::string_view_literals;
    constexpr flow::bit_matrix<3> m{"100"
                                    "010"
                                    "001"sv};
    constexpr flow::bit_matrix<3> n{"011"
                                    "101"
                                    "110"sv};
    constexpr flow::bit_matrix<3> r{"111"
                                    "111"
                                    "111"sv};
    static_assert((m | n) == r);
}

TEST_CASE("and", "[bit matrix]") {
    using namespace std::string_view_literals;
    constexpr flow::bit_matrix<3> m{"101"
                                    "010"
                                    "101"sv};
    constexpr flow::bit_matrix<3> n{"010"
                                    "101"
                                    "010"sv};
    static_assert((m & n) == flow::bit_matrix<3>{});
}

TEST_CASE("complement", "[bit matrix]") {
    using namespace std::string_view_literals;
    constexpr auto m = ~flow::bit_matrix<2>::identity();
    constexpr flow::bit_matrix<2> r{"01"
                                    "10"sv};
    static_assert(m == r);
}

TEST_CASE("xor", "[bit matrix]") {
    auto m = flow::bit_matrix<2>::identity();
    auto n = m ^ m;
    CHECK(n == flow::bit_matrix<2>{});
}

TEST_CASE("transpose identity", "[bit matrix]") {
    auto m = flow::bit_matrix<2>::identity();
    CHECK(transpose(m) == flow::bit_matrix<2>::identity());
}

TEST_CASE("transpose", "[bit matrix]") {
    auto m = flow::bit_matrix<14>{};
    m.set(0, 0);
    m.set(12, 4);
    m.set(13, 0);
    auto n = flow::bit_matrix<14>{};
    n.set(0, 0);
    n.set(4, 12);
    n.set(0, 13);
    CHECK(transpose(m) == n);
}

TEST_CASE("add", "[bit matrix]") {
    auto m = flow::bit_matrix<2>{};
    m.set(0, 0);
    auto n = flow::bit_matrix<2>{};
    n.set(1, 1);
    CHECK(m + n == flow::bit_matrix<2>::identity());
}

// TODO:
// split into 4 submatrices for recursion

// TEST_CASE("multiply", "[bit matrix]") {
//     auto m = flow::bit_matrix<2>{};
//     m.set(0, 0);
//     m.set(1, 0);
//     CHECK(m * m == m);
// }

// TEST_CASE("power", "[bit matrix]") {
//     auto m = flow::bit_matrix<3>{};
//     m.set(0, 0);
//     m.set(0, 2);
//     m.set(1, 2);
//     m.set(2, 0);
//     m.set(2, 1);

//     auto n = flow::bit_matrix<3>{};
//     n.set(0, 0);
//     n.set(0, 1);
//     n.set(0, 2);
//     n.set(1, 0);
//     n.set(1, 2);
//     n.set(2, 0);
//     n.set(2, 1);
//     n.set(2, 2);
//     CHECK(pow(m, 3) == n);
// }

// TEST_CASE("power (identity)", "[bit matrix]") {
//     constexpr auto m = flow::bit_matrix<32>::identity();
//     static_assert(pow(m, 32) == m);
// }
