#pragma once

#include <interrupt/concepts.hpp>

#include <stdx/bitset.hpp>
#include <stdx/compiler.hpp>
#include <stdx/concepts.hpp>
#include <stdx/tuple.hpp>
#include <stdx/tuple_algorithms.hpp>
#include <stdx/type_traits.hpp>
#include <stdx/utility.hpp>

#include <conc/concurrency.hpp>

#include <boost/mp11/algorithm.hpp>
#include <boost/mp11/list.hpp>
#include <boost/mp11/utility.hpp>

#include <type_traits>
#include <utility>

template <typename...> struct undef;

namespace interrupt {
namespace detail {
template <typename Irq> using get_resources_t = typename Irq::resources_t;
template <typename Irq> using get_flows_t = typename Irq::flows_t;
template <typename Irq>
using get_name_list_t = stdx::type_list<typename Irq::name_t>;

template <typename Irq>
using has_real_enable_field =
    std::bool_constant<not is_no_field_v<typename Irq::enable_field_t>>;

template <typename Irq>
using has_enable_field = boost::mp11::mp_eval_if_not<
    std::bool_constant<requires { typename Irq::enable_field_t; }>,
    std::false_type, has_real_enable_field, Irq>;

template <typename Irq>
using enable_fields_t =
    boost::mp11::mp_remove_if<stdx::type_list<typename Irq::enable_field_t>,
                              is_no_field>;

template <typename Irq>
using get_enable_field_t =
    boost::mp11::mp_eval_if_not<has_enable_field<Irq>, stdx::type_list<>,
                                enable_fields_t, Irq>;

template <typename Root>
using descendants_t = std::remove_cvref_t<decltype(Root::descendants)>;

template <typename Root, template <typename...> typename F,
          template <typename...> typename Result = stdx::type_bitset,
          template <typename> typename Desc = descendants_t>
using collect_t = boost::mp11::mp_apply<
    Result, boost::mp11::mp_unique<decltype(std::declval<Desc<Root>>().apply(
                []<typename... Irqs>(Irqs const &...) {
                    return boost::mp11::mp_append<F<Irqs>...>{};
                }))>>;

template <typename Name> struct has_name {
    template <typename Irq> using fn = std::is_same<Name, typename Irq::name_t>;
};

template <typename Irq> using get_name_t = typename Irq::name_t;

template <typename Resource> struct has_resource {
    template <typename Irq>
    using fn = boost::mp11::mp_contains<get_resources_t<Irq>, Resource>;
};

template <typename Irqs> struct with_resource {
    template <typename Resource>
    using fn = stdx::tt_pair<
        Resource, boost::mp11::mp_transform<
                      get_name_t,
                      boost::mp11::mp_copy_if_q<Irqs, has_resource<Resource>>>>;
};

template <typename Flow> struct has_flow {
    template <typename Irq>
    using fn = boost::mp11::mp_contains<get_flows_t<Irq>, Flow>;
};

template <typename Irqs> struct with_flow {
    template <typename Flow>
    using fn = stdx::tt_pair<
        Flow, boost::mp11::mp_transform<
                  get_name_t, boost::mp11::mp_copy_if_q<Irqs, has_flow<Flow>>>>;
};

template <typename Irqs, typename Resources>
using resource_map_impl = boost::mp11::mp_apply<
    stdx::type_map,
    boost::mp11::mp_transform_q<detail::with_resource<Irqs>, Resources>>;

template <typename Irqs, typename Resources>
constexpr auto make_resource_map = [] {
    struct RM : resource_map_impl<Irqs, Resources> {};
    return RM{};
};

template <typename Irqs, typename Resources>
using resource_map_t = decltype(make_resource_map<Irqs, Resources>());

template <typename Irqs, typename Flows>
using flow_map_impl =
    boost::mp11::mp_apply<stdx::type_map, boost::mp11::mp_transform_q<
                                              detail::with_flow<Irqs>, Flows>>;

template <typename Irqs, typename Flows>
constexpr auto make_flow_map = [] {
    struct FM : flow_map_impl<Irqs, Flows> {};
    return FM{};
};

template <typename Irqs, typename Flows>
using flow_map_t = decltype(make_flow_map<Irqs, Flows>());

template <typename Hal> struct get_register_q {
    template <typename Field>
    using from_field = decltype(Hal::template get_register<Field>());

    template <typename Irq>
    using from_irq =
        decltype(Hal::template get_register<typename Irq::enable_field_t>());
};

namespace dynamic_hal_concept {
template <typename Cfg>
using enable_fields_t = collect_t<Cfg, get_enable_field_t>;

template <typename Cfg>
using first_enable_field_t = boost::mp11::mp_first<enable_fields_t<Cfg>>;

template <typename Hal, typename Cfg>
using register_type_t =
    decltype(Hal::template get_register<first_enable_field_t<Cfg>>());

template <typename Hal, typename Cfg>
using register_datatype_t =
    typename Hal::template register_datatype_t<register_type_t<Hal, Cfg>>;
} // namespace dynamic_hal_concept

template <typename Hal, typename Cfg>
concept dynamic_hal_for =
    boost::mp11::mp_empty<dynamic_hal_concept::enable_fields_t<Cfg>>::value or
    requires(dynamic_hal_concept::register_datatype_t<Hal, Cfg> value) {
        typename dynamic_hal_concept::register_type_t<Hal, Cfg>;
        {
            Hal::template mask<dynamic_hal_concept::register_type_t<Hal, Cfg>,
                               dynamic_hal_concept::first_enable_field_t<Cfg>>
        } -> stdx::same_as_unqualified<
            dynamic_hal_concept::register_datatype_t<Hal, Cfg>>;
        Hal::template write<dynamic_hal_concept::register_type_t<Hal, Cfg>>(
            value);
    };
} // namespace detail

template <typename Root, /*detail::dynamic_hal_for<Root>*/ typename Hal>
struct dynamic_controller {
  private:
    // keep track of which resources/flows/irqs have been manually
    // enabled/disabled
    using resources_t = detail::collect_t<Root, detail::get_resources_t>;
    CONSTINIT static inline resources_t resource_enables{};

    using flows_t = detail::collect_t<Root, detail::get_flows_t>;
    CONSTINIT static inline flows_t flow_enables{};

    using irq_names_t = detail::collect_t<Root, detail::get_name_list_t>;
    CONSTINIT static inline irq_names_t named_enables{};

    // update bitsets as necessary
    template <typename T, typename... Us>
    static auto enable_one(stdx::type_bitset<Us...> &bs) -> void {
        if constexpr ((... or std::same_as<T, Us>)) {
            bs.template set<T>();
        }
    }
    template <typename T> static auto enable_one() -> void {
        if constexpr (boost::mp11::mp_contains<resources_t, T>::value) {
            enable_one<T>(resource_enables);
        } else if constexpr (boost::mp11::mp_contains<flows_t, T>::value) {
            enable_one<T>(flow_enables);
        } else {
            enable_one<T>(named_enables);
        }
    }

    template <typename T, typename... Us>
    static auto disable_one(stdx::type_bitset<Us...> &bs) -> void {
        if constexpr ((... or std::same_as<T, Us>)) {
            bs.template reset<T>();
        }
    }
    template <typename T> static auto disable_one() -> void {
        if constexpr (boost::mp11::mp_contains<resources_t, T>::value) {
            disable_one<T>(resource_enables);
        } else if constexpr (boost::mp11::mp_contains<flows_t, T>::value) {
            disable_one<T>(flow_enables);
        } else {
            disable_one<T>(named_enables);
        }
    }

    // which resources/flows affect which irqs?
    // compute maps: resource -> list<irq names>, flow -> list<irq names>
    using irqs_t = detail::collect_t<Root, stdx::type_list>;

    // cached values for each enable register
    template <typename Register>
    using register_t = typename Hal::template register_datatype_t<Register>;

    template <typename Register>
    CONSTINIT static inline register_t<Register> cached_enable{};

    // which IRQs are potentially affected by a change?
    template <typename... Ts> CONSTEVAL static auto compute_affected_irqs() {
        using resource_map_t = detail::resource_map_t<irqs_t, resources_t>;
        using flow_map_t = detail::flow_map_t<irqs_t, flows_t>;

        using affected_irq_names_by_resource_t = boost::mp11::mp_append<
            stdx::type_lookup_t<resource_map_t, Ts, stdx::type_list<>>...>;
        using affected_irq_names_by_flow_t = boost::mp11::mp_append<
            stdx::type_lookup_t<flow_map_t, Ts, stdx::type_list<>>...>;

        return boost::mp11::mp_unique<boost::mp11::mp_append<
            affected_irq_names_by_resource_t, affected_irq_names_by_flow_t>>{};
    }

    // compute changing value/mask for register
    template <typename Reg, typename Irq, typename T>
    static auto compute_register(T &new_value, T &mask) -> void {
        constexpr auto field_mask =
            Hal::template mask<Reg, typename Irq::enable_field_t>;
        mask |= field_mask;

        auto const all_resources_on = [] {
            constexpr auto resources_bs =
                resources_t{typename Irq::resources_t{}};
            return (resource_enables & resources_bs) == resources_bs;
        }();
        auto const any_flow_on = [] {
            constexpr auto flows_bs = flows_t{typename Irq::flows_t{}};
            if constexpr (flows_bs.none()) {
                return true;
            }
            return (flow_enables & flows_bs) != flows_t{};
        }();

        if (all_resources_on and any_flow_on and
            named_enables[stdx::type_identity_v<typename Irq::name_t>]) {
            new_value |= field_mask;
        }
    }

    // update cached values and write the changed registers
    template <template <typename...> typename L, typename... Irqs>
    static auto update(L<Irqs...>) -> void {
        constexpr auto by_registers =
            stdx::gather_by<detail::get_register_q<Hal>::template from_irq>(
                stdx::tuple<Irqs...>{});

        stdx::for_each(
            []<typename I, typename... Is>(stdx::tuple<I, Is...>) {
                using reg_t = decltype(Hal::template get_register<
                                       typename I::enable_field_t>());
                auto mask = register_t<reg_t>{};
                auto value = register_t<reg_t>{};
                (compute_register<reg_t, I>(value, mask), ...,
                 compute_register<reg_t, Is>(value, mask));

                auto &cached_value = cached_enable<reg_t>;
                auto new_value = (cached_value & ~mask) | value;
                if (new_value != std::exchange(cached_value, new_value)) {
                    Hal::template write<reg_t>(cached_value);
                }
            },
            by_registers);
    }

    // template <template <typename...> typename L, typename... IrqNames>
    template <typename T> static auto update_affected_irqs(T const &) -> void {
        // using affected_irqs_t =
        //     boost::mp11::mp_append<boost::mp11::mp_copy_if_q<
        //         detail::descendants_t<Root>, detail::has_name<IrqNames>>...>;
        // update(affected_irqs_t{});
    }

    // reset: mark all resources, flows and named irqs enabled
    // and clear all cached state
    template <bool Enable, typename IrqList = stdx::tuple<>>
    static auto reset_internal_state() -> void {
        if constexpr (Enable) {
            resource_enables = resources_t{stdx::all_bits};
            flow_enables = flows_t{stdx::all_bits};
            named_enables = irq_names_t{stdx::all_bits};
        } else {
            resource_enables.reset();
            flow_enables.reset();
            named_enables.reset();
        }

        constexpr auto by_registers =
            stdx::gather_by<detail::get_register_q<Hal>::template from_irq>(
                IrqList{});
        stdx::for_each(
            []<typename I, typename... Is>(stdx::tuple<I, Is...>) {
                using reg_t = decltype(Hal::template get_register<
                                       typename I::enable_field_t>());
                cached_enable<reg_t> = register_t<reg_t>{};
                if constexpr (not Enable) {
                    auto value = register_t<reg_t>{};
                    auto &mask = cached_enable<reg_t>;
                    (compute_register<reg_t, I>(value, mask), ...,
                     compute_register<reg_t, Is>(value, mask));
                }
            },
            by_registers);
    }

  public:
    using hal_t = Hal;

    template <bool Enable = true> static auto init() {
        using enable_irqs_t [[maybe_unused]] =
            boost::mp11::mp_copy_if<detail::descendants_t<Root>,
                                    detail::has_enable_field>;
        reset_internal_state<Enable, enable_irqs_t>();
        update(enable_irqs_t{});
    }

    template <stdx::ct_string... IrqNames> static auto enable() -> void {
        (enable_one<stdx::cts_t<IrqNames>>(), ...);
        update_affected_irqs(stdx::type_list<stdx::cts_t<IrqNames>...>{});
    }
    template <stdx::ct_string... IrqNames> static auto disable() -> void {
        (disable_one<stdx::cts_t<IrqNames>>(), ...);
        update_affected_irqs(stdx::type_list<stdx::cts_t<IrqNames>...>{});
    }

    template <typename... Ts> static auto enable() -> void {
        (enable_one<Ts>(), ...);

        using resource_map_t = detail::resource_map_t<irqs_t, resources_t>;
        using flow_map_t = detail::flow_map_t<irqs_t, flows_t>;

        using affected_irq_names_by_resource_t = boost::mp11::mp_append<
            stdx::type_lookup_t<resource_map_t, Ts, stdx::type_list<>>...>;
        using affected_irq_names_by_flow_t = boost::mp11::mp_append<
            stdx::type_lookup_t<flow_map_t, Ts, stdx::type_list<>>...>;

        using affected_irq_names_t = boost::mp11::mp_unique<
            boost::mp11::mp_append<affected_irq_names_by_resource_t,
                                   affected_irq_names_by_flow_t>>;

        // constexpr auto affected_irq_names = compute_affected_irqs<Ts...>();
        update_affected_irqs(affected_irq_names_t{});
    }
    template <typename... Ts> static auto disable() -> void {
        // (disable_one<Ts>(), ...);

        // using resource_map_t = detail::resource_map_t<irqs_t, resources_t>;
        // using flow_map_t = detail::flow_map_t<irqs_t, flows_t>;

        // using affected_irq_names_by_resource_t = boost::mp11::mp_append<
        //     stdx::type_lookup_t<resource_map_t, Ts, stdx::type_list<>>...>;
        // using affected_irq_names_by_flow_t = boost::mp11::mp_append<
        //     stdx::type_lookup_t<flow_map_t, Ts, stdx::type_list<>>...>;

        // using affected_irq_names_t = boost::mp11::mp_unique<
        //     boost::mp11::mp_append<affected_irq_names_by_resource_t,
        //                            affected_irq_names_by_flow_t>>;

        // // constexpr auto affected_irq_names =
        // compute_affected_irqs<Ts...>();
        // update_affected_irqs(affected_irq_names_t{});
    }
};
} // namespace interrupt
