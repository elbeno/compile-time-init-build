#pragma once

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <climits>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <numeric>
#include <ostream>
#include <span>
#include <string_view>

namespace flow {
namespace detail {
struct index_t {
    std::size_t array_index;
    std::size_t bit_index;
};

[[nodiscard]] constexpr auto to_morton(std::size_t x, std::size_t y) {
    return (((x * 0x0101'0101'0101'0101ull & 0x8040'2010'0804'0201ull) *
                 0x0102'0408'1020'4081ull >>
             49u) &
            0x5555ull) |
           (((y * 0x0101'0101'0101'0101ull & 0x8040'2010'0804'0201ull) *
                 0x0102'0408'1020'4081ull >>
             48u) &
            0xAAAAull);
}

template <std::size_t block_size, std::size_t extent>
[[nodiscard]] constexpr auto compute_index(std::size_t row, std::size_t col) {
    auto const block_row_index = row / block_size;
    auto const block_col_index = col / block_size;
    auto const ai = to_morton(block_col_index, block_row_index);

    auto const block_row_subindex = row % block_size;
    auto const block_col_subindex = col % block_size;
    auto const bi = block_row_subindex * block_size + block_col_subindex;

    return index_t{ai, bi};
}

[[nodiscard]] constexpr auto transpose8x8(std::uint64_t x) {
    std::uint64_t t = (x ^ (x >> 7u)) & 0x00aa'00aa'00aa'00aaull;
    x ^= t ^ (t << 7u);
    t = (x ^ (x >> 14u)) & 0x0000'cccc'0000'ccccull;
    x ^= t ^ (t << 14u);
    t = (x ^ (x >> 28u)) & 0x0000'0000'f0f0'f0f0ull;
    x ^= t ^ (t << 28u);
    return x;
}

[[nodiscard]] constexpr auto multiply8x8(std::uint64_t x, std::uint64_t y) {
    constexpr std::uint64_t row = 0xffull;
    constexpr std::uint64_t col = 0x0101'0101'0101'0101ull;

    std::uint64_t r{};
    while (y != 0) {
        std::uint64_t yrow = y & row;
        std::uint64_t xcol = x & col;
        r |= yrow * xcol;
        y >>= 8u;
        x >>= 1u;
    }
    return r;
}

template <std::size_t N>
[[nodiscard]] constexpr auto multiplyNxN(std::span<std::uint64_t const, N> x,
                                         std::span<std::uint64_t const, N> y,
                                         std::span<std::uint64_t, N> dst) {
    if constexpr (N == 1) {
        dst[0] = multiply8x8(x[0], y[0]);
    } else if constexpr (N == 2) {
        dst[0] =
            detail::multiply8x8(x[0], y[0]) | detail::multiply8x8(y[1], y[2]);
        dst[1] =
            detail::multiply8x8(x[0], y[1]) | detail::multiply8x8(x[1], y[3]);
        dst[2] =
            detail::multiply8x8(x[2], y[0]) | detail::multiply8x8(x[3], y[2]);
        dst[3] =
            detail::multiply8x8(x[2], y[1]) | detail::multiply8x8(x[3], y[3]);
    } else {
        constexpr auto M = N / 4;
        using subsrc = std::span<std::uint64_t const, M>;

        auto const a1 = subsrc{std::begin(x), M};
        auto const b1 = subsrc{std::next(std::begin(x), M), M};
        auto const c1 = subsrc{std::next(std::begin(x), M * 2), M};
        auto const d1 = subsrc{std::next(std::begin(x), M * 3), M};

        auto const a2 = subsrc{std::begin(y), M};
        auto const b2 = subsrc{std::next(std::begin(y), M), M};
        auto const c2 = subsrc{std::next(std::begin(y), M * 2), M};
        auto const d2 = subsrc{std::next(std::begin(y), M * 3), M};

        using subdst = std::span<std::uint64_t, M>;
        auto const dst1 = subdst{std::begin(dst), M};
        auto const dst2 = subdst{std::next(std::begin(dst), M), M};
        auto const dst3 = subdst{std::next(std::begin(dst), M * 2), M};
        auto const dst4 = subdst{std::next(std::begin(dst), M * 3), M};

        std::array<std::uint64_t, M> intermediate{};

        multiplyNxN(a1, a2, std::span{intermediate});
        multiplyNxN(b1, c2, std::span{dst1});
        std::transform(std::begin(intermediate), std::end(intermediate),
                       std::begin(dst1), std::begin(dst1), std::bit_or{});

        multiplyNxN(a1, b2, std::span{intermediate});
        multiplyNxN(b1, d2, std::span{dst2});
        std::transform(std::begin(intermediate), std::end(intermediate),
                       std::begin(dst2), std::begin(dst2), std::bit_or{});

        multiplyNxN(c1, a2, std::span{intermediate});
        multiplyNxN(d1, c2, std::span{dst3});
        std::transform(std::begin(intermediate), std::end(intermediate),
                       std::begin(dst3), std::begin(dst3), std::bit_or{});

        multiplyNxN(c1, b2, std::span{intermediate});
        multiplyNxN(d1, d2, std::span{dst4});
        std::transform(std::begin(intermediate), std::end(intermediate),
                       std::begin(dst4), std::begin(dst4), std::bit_or{});
    }
}
} // namespace detail

template <std::size_t N> class bit_matrix {
    using elem_type = std::uint64_t;

    constexpr static auto block_size = 8u;
    constexpr static auto num_blocks =
        std::bit_ceil((N + block_size - 1) / block_size);

    constexpr static auto bit = elem_type{1u};
    constexpr static auto allbits = std::numeric_limits<elem_type>::max();
    constexpr static elem_type identity_elem = 0x80'40'20'10'08'04'02'01;

    constexpr static auto leftover_rows = N % block_size;
    constexpr static elem_type bottom_row_mask =
        (leftover_rows != 0) ? (bit << (leftover_rows * block_size)) - 1
                             : allbits;

    constexpr static auto leftover_cols = N % block_size;
    constexpr static elem_type right_column_mask8 =
        (leftover_cols != 0) ? (bit << leftover_cols) - 1 : 0xffu;
    constexpr static elem_type right_column_mask =
        right_column_mask8 | (right_column_mask8 << 8u) |
        (right_column_mask8 << 16u) | (right_column_mask8 << 24u) |
        (right_column_mask8 << 32u) | (right_column_mask8 << 40u) |
        (right_column_mask8 << 48u) | (right_column_mask8 << 56u);

    std::array<elem_type, num_blocks * num_blocks> storage{};

    constexpr auto set(detail::index_t i, bool value) -> bit_matrix & {
        auto const [ai, bi] = i;
        if (value) {
            storage[ai] |= (bit << bi);
        } else {
            storage[ai] &= ~(bit << bi);
        }
        return *this;
    }

    constexpr auto get(detail::index_t i) const -> bool {
        auto const [ai, bi] = i;
        return (storage[ai] & (bit << bi)) != 0;
    }

    struct proxy : detail::index_t {
        bit_matrix *m;
        constexpr auto operator=(bool value) -> proxy & {
            m->set(*this, value);
            return *this;
        }

        constexpr explicit(true) operator bool() const { return m->get(*this); }
    };

    friend constexpr auto operator==(bit_matrix const &lhs,
                                     bit_matrix const &rhs) -> bool = default;

    friend constexpr auto operator|(bit_matrix lhs, bit_matrix const &rhs)
        -> bit_matrix {
        lhs |= rhs;
        return lhs;
    }

    friend constexpr auto operator&(bit_matrix lhs, bit_matrix const &rhs)
        -> bit_matrix {
        lhs &= rhs;
        return lhs;
    }

    friend constexpr auto operator^(bit_matrix lhs, bit_matrix const &rhs)
        -> bit_matrix {
        lhs ^= rhs;
        return lhs;
    }

    friend constexpr auto operator~(bit_matrix const &m) -> bit_matrix {
        bit_matrix r{};
        std::transform(std::begin(m.storage), std::end(m.storage),
                       std::begin(r.storage), std::bit_not{});
        r.mask_edges();
        return r;
    }

    friend constexpr auto transpose(bit_matrix const &m) -> bit_matrix {
        bit_matrix result{};
        for (auto i = std::size_t{}; i < num_blocks; ++i) {
            for (auto j = std::size_t{}; j < num_blocks; ++j) {
                result.storage[j * num_blocks + i] =
                    detail::transpose8x8(m.storage[i * num_blocks + j]);
            }
        }
        return result;
    }

    friend constexpr auto operator+(bit_matrix lhs, bit_matrix const &rhs)
        -> bit_matrix {
        lhs += rhs;
        return lhs;
    }

    friend constexpr auto operator*(bit_matrix lhs, bit_matrix const &rhs)
        -> bit_matrix {
        lhs *= rhs;
        return lhs;
    }

    friend constexpr auto pow(bit_matrix const &m, std::size_t n)
        -> bit_matrix {
        if (n == 1) {
            return m;
        }
        if (n % 2 == 0) {
            auto const r = pow(m, n / 2);
            return r * r;
        }
        return m * pow(m, n - 1);
    }

    friend auto operator<<(std::ostream &os, bit_matrix const &m)
        -> std::ostream & {
        for (auto i = std::size_t{}; i < N; ++i) {
            for (auto j = std::size_t{}; j < N; ++j) {
                os << (m.index(i, j) ? '1' : '0');
            }
            os << '\n';
        }
        return os;
    }

  public:
    constexpr bit_matrix() = default;
    constexpr bit_matrix(std::string_view v) {
        auto i = std::size_t{};
        auto j = std::size_t{};
        for (auto c : v) {
            if (c == '1') {
                set(i, j);
            }
            if (++j == N) {
                j = 0;
                ++i;
            }
        }
    }

    [[nodiscard]] constexpr static auto identity() -> bit_matrix {
        bit_matrix m{};
        for (auto i = std::size_t{}; i < num_blocks; ++i) {
            m.storage[detail::to_morton(i, i)] = identity_elem;
        }
        m.storage.back() &= right_column_mask & bottom_row_mask;
        return m;
    }

    [[nodiscard]] constexpr auto index(std::size_t row, std::size_t col) const
        -> bool {
        return get(detail::compute_index<block_size, num_blocks>(row, col));
    }

    [[nodiscard]] constexpr auto index(std::size_t row, std::size_t col)
        -> proxy {
        return {detail::compute_index<block_size, num_blocks>(row, col), this};
    }

    constexpr auto set(std::size_t row, std::size_t col, bool value = true)
        -> bit_matrix & {
        return set(detail::compute_index<block_size, num_blocks>(row, col),
                   value);
    }

    constexpr auto set() -> bit_matrix & {
        storage.fill(allbits);
        mask_edges();
        return *this;
    }

    constexpr auto clear() -> bit_matrix & {
        storage.fill(0u);
        return *this;
    }

    constexpr auto operator|=(bit_matrix const &rhs) -> bit_matrix & {
        std::transform(std::begin(storage), std::end(storage),
                       std::begin(rhs.storage), std::begin(storage),
                       std::bit_or{});
        return *this;
    }

    constexpr auto operator&=(bit_matrix const &rhs) -> bit_matrix & {
        std::transform(std::begin(storage), std::end(storage),
                       std::begin(rhs.storage), std::begin(storage),
                       std::bit_and{});
        return *this;
    }

    constexpr auto operator^=(bit_matrix const &rhs) -> bit_matrix & {
        std::transform(std::begin(storage), std::end(storage),
                       std::begin(rhs.storage), std::begin(storage),
                       std::bit_xor{});
        return *this;
    }

    constexpr auto operator+=(bit_matrix const &rhs) -> bit_matrix & {
        return this->operator^=(rhs);
    }

    constexpr auto operator*=(bit_matrix const &m) -> bit_matrix & {
        detail::multiplyNxN(std::span{std::as_const(storage)},
                            std::span{m.storage}, std::span{storage});
        return *this;
    }

    constexpr auto mask_edges() -> void {
        constexpr auto right_col = num_blocks - 1;
        for (auto i = std::size_t{}; i < num_blocks; ++i) {
            auto const idx = detail::to_morton(right_col, i);
            storage[idx] &= right_column_mask;
        }
        constexpr auto bottom_row = num_blocks - 1;
        for (auto i = std::size_t{}; i < num_blocks; ++i) {
            auto const idx = detail::to_morton(i, bottom_row);
            storage[idx] &= bottom_row_mask;
        }
    }
};
} // namespace flow
