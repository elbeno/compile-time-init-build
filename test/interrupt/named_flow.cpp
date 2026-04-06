#include "common.hpp"

#include <flow/flow.hpp>
#include <interrupt/config.hpp>
#include <interrupt/dynamic_controller.hpp>
#include <interrupt/manager.hpp>
#include <nexus/config.hpp>
#include <nexus/nexus.hpp>

#include <groov/groov.hpp>
#include <groov/test.hpp>

#include <catch2/catch_test_macros.hpp>

std::string actual = {};

namespace {
using namespace flow::literals;

constexpr auto a = flow::action<"a">([] { actual += "a"; });
struct named_flow : public flow::service<"named"> {};

struct nexus_config {
    constexpr static auto config =
        cib::config(cib::exports<named_flow>, cib::extend<"named">(*a));
};

using G = groov::group<"test", groov::test::bus<"test">>;
} // namespace

TEST_CASE("run flow by name", "[named_flow]") {
    using interrupt::operator""_irq;

    cib::nexus<nexus_config> nexus{};
    nexus.init();

    using int_config = interrupt::root<
        interrupt::irq<"17", 17_irq, 42, interrupt::policies<>, "named">>;

    auto m = interrupt::manager<int_config, test_hal<G>, decltype(nexus)>{};

    actual.clear();
    m.run<17_irq>();
    CHECK(actual.find('a') != std::string::npos);
}
