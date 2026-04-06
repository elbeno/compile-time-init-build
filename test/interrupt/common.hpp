#pragma once

#include <interrupt/config.hpp>
#include <interrupt/fwd.hpp>
#include <interrupt/policies.hpp>

#include <groov/groov.hpp>
#include <groov/test.hpp>

#include <stdx/compiler.hpp>
#include <stdx/concepts.hpp>
#include <stdx/ct_string.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <string_view>

using interrupt::operator""_irq;

namespace {
template <typename interrupt::irq_num_t> bool enabled{};
template <typename interrupt::irq_num_t> std::size_t priority{};
bool inited{};

using namespace stdx::literals;
template <stdx::ct_string S> using en_field_t = stdx::cts_t<"enable."_cts + S>;
template <stdx::ct_string S> using st_field_t = stdx::cts_t<"status."_cts + S>;

namespace detail {
template <typename Flow> struct service_t {
    template <typename Nexus>
    constexpr static auto active = Nexus::template get_service<Flow>().active;

    template <typename Nexus> constexpr static auto run() -> void {
        Nexus::template service<Flow>();
    }
};

template <stdx::ct_string Name> struct service_t<stdx::cts_t<Name>> {
    template <typename Nexus>
    constexpr static auto active = Nexus::template get_service<Name>().active;

    template <typename Nexus> constexpr static auto run() -> void {
        Nexus::template service<Name>();
    }
};
} // namespace detail

template <typename Group> struct test_hal {
    static auto init() -> void { inited = true; }

    template <bool Enable, interrupt::irq_num_t IrqNumber, std::size_t Priority>
    static auto irq_init() -> void {
        enabled<IrqNumber> = Enable;
        priority<IrqNumber> = Priority;
    }

    template <typename Field>
    CONSTEVAL static auto get_field() -> groov::pathlike auto {
        return groov::make_path<Field::value>();
    }
    template <typename Field>
    CONSTEVAL static auto get_register() -> groov::pathlike auto {
        return groov::parent(get_field<Field>());
    }

    template <groov::pathlike Register>
    using register_datatype_t =
        typename decltype(groov::resolve(Group{}, Register{}))::type_t;

    template <groov::pathlike Register, typename Field>
    constexpr static register_datatype_t<Register> mask =
        groov::resolve(Group{}, groov::make_path<Field::value>())
            .template mask<register_datatype_t<Register>>;

    template <groov::pathlike P>
    static auto write(P p, auto raw_value) -> void {
        groov::sync_write(Group{}(p = raw_value));
    }
    template <groov::pathlike P> static auto read(P p) -> bool {
        auto const value = groov::test::get_value<Group>(groov::parent(p));
        REQUIRE(value);
        using Field = decltype(groov::resolve(Group{}, P{}));
        return Field::extract(*value);
    }
    template <groov::pathlike P> static auto clear(P p) -> void {
        groov::sync_write(Group{}(p = groov::clear));
    }

    template <typename Nexus, typename Flow>
    CONSTEVAL static auto active() -> bool {
        return detail::service_t<Flow>::template active<Nexus>;
    }

    template <typename Nexus, typename Flow>
    constexpr static auto run() -> void {
        detail::service_t<Flow>::template run<Nexus>();
    }
};

namespace detail {
template <template <typename> typename Flow> struct test_nexus {
    template <stdx::ct_string Name>
    constexpr static auto service_v = Flow<stdx::cts_t<Name>>{};
    template <stdx::ct_string Name>
    CONSTEVAL static auto get_service() -> auto & {
        return service_v<Name>;
    }
    template <stdx::ct_string Name> constexpr static auto service() {
        return get_service<Name>()();
    }
};
} // namespace detail
} // namespace

template <stdx::ct_string> inline bool flow_run{};

template <typename T> struct flow_t {
    auto operator()() const { flow_run<T::value> = true; }
    constexpr static bool active =
        std::string_view{T::value}.starts_with("true");
};

using test_nexus = detail::test_nexus<flow_t>;
