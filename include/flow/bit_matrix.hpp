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

template <std::size_t block_size, std::size_t extent>
[[nodiscard]] constexpr auto compute_index(std::size_t row, std::size_t col) {
    auto const block_row_index = row / block_size;
    auto const block_col_index = col / block_size;
    auto const ai = block_row_index * extent + block_col_index;

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

    friend constexpr auto
    copy_from_larger_block(bit_matrix &dst, std::span<std::uint64_t const> src,
                           std::size_t row_offset, std::size_t col_offset) {
        std::size_t c{};
        for (auto i = std::size_t{}; i < num_blocks; ++i) {
            for (auto j = std::size_t{}; j < num_blocks; ++j) {
                dst.storage[c++] =
                    src[(row_offset + i) * num_blocks * 2 + col_offset + j];
            }
        }
    }

    friend constexpr auto
    copy_from_smaller_block(bit_matrix &dst, std::span<std::uint64_t const> src,
                            std::size_t n, std::size_t row_offset,
                            std::size_t col_offset) {
        std::size_t c{};
        for (auto i = std::size_t{}; i < n; ++i) {
            for (auto j = std::size_t{}; j < n; ++j) {
                dst.storage[(row_offset + i) * num_blocks + col_offset + j] =
                    src[c++];
            }
        }
    }

    constexpr static auto halfsize = block_size * num_blocks / 2;

    [[nodiscard]] friend constexpr auto to_sub_blocks(bit_matrix const &m) {
        std::array<bit_matrix<halfsize>, 4> blocks{};
        auto const src = std::span{std::data(m.storage), std::size(m.storage)};
        copy_from_larger_block(blocks[0], src, 0, 0);
        copy_from_larger_block(blocks[1], src, 0, num_blocks / 2);
        copy_from_larger_block(blocks[2], src, num_blocks / 2, 0);
        copy_from_larger_block(blocks[3], src, num_blocks / 2, num_blocks / 2);
        return blocks;
    }

    template <std::size_t M>
    [[nodiscard]] friend constexpr auto
    from_sub_blocks(bit_matrix const &m0, bit_matrix const &m1,
                    bit_matrix const &m2, bit_matrix const &m3) {
        bit_matrix<M> r{};
        constexpr auto n = std::bit_ceil((M + block_size - 1) / block_size) / 2;

        copy_from_smaller_block(
            r, std::span{std::data(m0.storage), std::size(m0.storage)},
            num_blocks, 0, 0);
        copy_from_smaller_block(
            r, std::span{std::data(m1.storage), std::size(m1.storage)},
            num_blocks, 0, n);
        copy_from_smaller_block(
            r, std::span{std::data(m2.storage), std::size(m2.storage)},
            num_blocks, n, 0);
        copy_from_smaller_block(
            r, std::span{std::data(m3.storage), std::size(m3.storage)},
            num_blocks, n, n);

        r.mask_edges();
        return r;
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
            m.storage[i * num_blocks + i] = identity_elem;
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

    constexpr auto operator*=(bit_matrix const &m) -> bit_matrix &
        requires(num_blocks == 1)
    {
        storage[0] = detail::multiply8x8(storage[0], m.storage[0]);
        return *this;
    }

    constexpr auto operator*=(bit_matrix const &m) -> bit_matrix &
        requires(num_blocks > 1)
    {
        auto const a = to_sub_blocks(*this);
        auto const b = to_sub_blocks(m);

        auto const sub_mults =
            std::array{a[0] * b[0] | a[1] * b[2], a[0] * b[1] | a[1] * b[3],
                       a[2] * b[0] | a[3] * b[2], a[2] * b[1] | a[3] * b[3]};
        *this = from_sub_blocks<N>(sub_mults[0], sub_mults[1], sub_mults[2],
                                   sub_mults[3]);
        return *this;
    }

    constexpr auto mask_edges() -> void {
        constexpr auto right_col = num_blocks - 1;
        for (auto i = std::size_t{}; i < num_blocks; ++i) {
            storage[i * num_blocks + right_col] &= right_column_mask;
        }
        constexpr auto bottom_row_start = num_blocks * (num_blocks - 1);
        for (auto i = std::size_t{}; i < num_blocks; ++i) {
            storage[bottom_row_start + i] &= bottom_row_mask;
        }
    }
};
} // namespace flow
