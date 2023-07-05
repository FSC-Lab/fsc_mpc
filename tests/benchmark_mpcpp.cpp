#include <functional>
#include <random>

#include "benchmark/benchmark.h"
#include "fscore/numbers/numbers.hpp"
#include "mpcpp/acados_mpc.hpp"

static std::random_device dev;
static std::mt19937 rng{dev()};

auto sample = [dist{std::uniform_real_distribution<>{0.0, 10.0}}]() mutable {
  return dist(rng);
};

static void BenchmarkMPC(benchmark::State& state) {
  const long int n_steps = state.range(0);
  const Eigen::VectorXd time_steps = Eigen::VectorXd::LinSpaced(n_steps, 0, 20);
  control::AcadosMPC mpc(time_steps);

  mpc.setConstantParameters(control::AcadosMPC::ParamType{1.0});

  control::AcadosMPC::StateType state_ref;
  state_ref << 10, 0, 0, Eigen::Quaterniond::Identity().coeffs(),
      Eigen::Vector3d::Zero();

  control::AcadosMPC::InputType input_ref{9.81, 0, 0, 0};
  mpc.setReferenceState(state_ref, input_ref);

  control::AcadosMPC::StateType x_op;

  Eigen::Ref<Eigen::Vector3d> position(x_op.head<3>());
  position.setZero();
  Eigen::Map<Eigen::Quaterniond> attitude(x_op.data() + 3);
  attitude.setIdentity();
  Eigen::Ref<Eigen::Vector3d> velocity(x_op.tail<3>());
  velocity.setZero();
  for (auto _ : state) {
    state.PauseTiming();
    position.z() = sample();

    state.ResumeTiming();

    control::AcadosMPC::InputType input = mpc.optimize(x_op);
    benchmark::DoNotOptimize(input);
  }
}

BENCHMARK(BenchmarkMPC)->Arg(5)->Arg(10);

BENCHMARK_MAIN();
