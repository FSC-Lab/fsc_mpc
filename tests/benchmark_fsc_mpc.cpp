// Benchmarking the 'optimize' function, i.e. solving MPC problem
// Copyright © 2023 FSC Lab
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
// OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <functional>
#include <random>

#include "benchmark/benchmark.h"
#include "fsc_mpc/mpc_interface.hpp"

static std::random_device dev;
static std::mt19937 rng{dev()};

static constexpr auto kMaxAlt = 10;
auto sample = [dist{std::uniform_real_distribution<>{0.0, kMaxAlt}}]() mutable {
  return dist(rng);
};

static void BenchmarkMPC(benchmark::State& state) {
  using fsc::control::MPCInterface;
  const auto n_steps = state.range(0);
  const Eigen::VectorXd time_steps =
      Eigen::VectorXd::Constant(n_steps, 1.0 / static_cast<double>(n_steps));
  MPCInterface mpc(time_steps);

  mpc.setConstantParameters(MPCInterface::ParamType{1.0});

  MPCInterface::StateType state_ref;
  state_ref << kMaxAlt, 0, 0, Eigen::Quaterniond::Identity().coeffs(),
      Eigen::Vector3d::Zero();

  constexpr double kGravAccel = 9.81;
  MPCInterface::InputType input_ref{kGravAccel, 0, 0, 0};
  mpc.setReferenceState(state_ref, input_ref);

  MPCInterface::StateType x_op;

  Eigen::Ref<Eigen::Vector3d> position(x_op.head<3>());
  position.setZero();
  Eigen::Map<Eigen::Quaterniond> attitude(x_op.data() + 3);
  attitude.setIdentity();
  Eigen::Ref<Eigen::Vector3d> velocity(x_op.tail<3>());
  velocity.setZero();
  for ([[maybe_unused]] auto _ : state) {
    state.PauseTiming();
    mpc.resetSolver(true);
    position.z() = sample();

    state.ResumeTiming();

    MPCInterface::InputType input = mpc.optimize(x_op);
    benchmark::DoNotOptimize(input);
  }
}

BENCHMARK(BenchmarkMPC)->Arg(5)->Arg(10);  // NOLINT

BENCHMARK_MAIN();
