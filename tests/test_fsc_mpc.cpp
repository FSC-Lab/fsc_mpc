// Tests the MPC against simulation data collected in Python
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

#include <algorithm>
#include <chrono>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "fsc_mpc/mpc_interface.hpp"
#include "fscore/math/math_extras.hpp"
#include "fscore/models/simple_quadrotor.hpp"
#include "fscore/simulation/dynamic_system_simulator.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#define RAPIDJSON_NOEXCEPT_ASSERT(cond) \
  do {                                  \
    if (!static_cast<bool>((cond))) {   \
      std::abort();                     \
    }                                   \
  } while (0)

#define RAPIDJSON_ASSERT(cond)            \
  do {                                    \
    if (!static_cast<bool>((cond)))       \
      throw std::runtime_error(#cond      \
                               " is not " \
                               "met");    \
  } while (0)

#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"

#ifndef TEST_DATA_FILE
#define TEST_DATA_FILE ""
#error THE_TEST_DATA_FILE_MACRO_MUST_BE_DEFINED
#endif

static const Eigen::IOFormat kFmt(Eigen::StreamPrecision, 0, ",", ";\n", "", "",
                                  "[", "]");

MATCHER_P(QuaternionIsClose, expected, ::testing::PrintToString(expected)) {
  const auto ang_dist = arg.angularDistance(expected);
  const bool pass = fsc::IsClose(
      ang_dist, 0.0, {std::numeric_limits<decltype(ang_dist)>::max(), 1e-3});
  if (!pass) {
    *result_listener << "Angular distance is " << ang_dist;
    return false;
  }
  return true;
}

// NOLINTNEXTLINE
void Check3DTrajectories(const Eigen::MatrixXd& a_matrix,
                         const Eigen::MatrixXd& b_matrix, double rtol,
                         double atol) {
  ASSERT_EQ(a_matrix.rows(), 3);
  ASSERT_EQ(b_matrix.rows(), 3);
  ASSERT_EQ(a_matrix.cols(), b_matrix.cols());
  const auto len_traj = a_matrix.cols();

  ASSERT_LE(std::numeric_limits<double>::epsilon(), rtol);
  ASSERT_LT(rtol, 1.0);

  for (int i = 0; i < len_traj; ++i) {
    const Eigen::Vector3d a = a_matrix.col(i);
    const Eigen::Vector3d b = b_matrix.col(i);
    if (a == b) {
      continue;
    }

    const auto diff = (a - b).norm();
    const auto norm =
        std::min((a.norm() + b.norm()), std::numeric_limits<double>::max());

    const auto threshold = std::max(atol, rtol * norm);
    ASSERT_LT(diff, threshold)
        << "Trajectory point mismatch between a: " << a.transpose().format(kFmt)
        << " and b: " << b.transpose().format(kFmt) << " on iteration " << i;
  }
}

std::unordered_map<std::string, Eigen::MatrixXd> ReadMatrixFromJson(
    const rapidjson::Value& value) {
  std::unordered_map<std::string, Eigen::MatrixXd> res;
  for (const auto& it : value.GetObject()) {
    auto size = it.value["size"].GetArray();
    const int rows = size[0].GetInt();
    const int cols = size[1].GetInt();
    const auto& val = it.value["values"].GetArray();

    const auto& key = it.name.GetString();
    res.emplace(key, Eigen::MatrixXd(rows, cols));
    std::transform(val.begin(), val.end(), res[key].data(),
                   std::mem_fn(&rapidjson::Value::GetDouble));
  }
  return res;
}

class TestMPCInterface : public ::testing::Test {
 public:
  using MdlType = fsc::SimpleQuadrotor<double, fsc::conventions::Robotics>;
  using SimType = fsc::DynamicSystemSimulator<MdlType, fsc::ODE4>;
  using MPCInterface = fsc::control::MPCInterface;
  TestMPCInterface() = default;

  void SetUp() override;

  void RunSimulation();
  static constexpr double kRelativeTolerance = 1e-4;
  static constexpr double kAbsoluteTolerance = 1e-4;

  std::unordered_map<std::string, Eigen::MatrixXd> sim_out;
  std::unordered_map<std::string, Eigen::MatrixXd> trajectory;
  Eigen::MatrixXd actual_states;
  Eigen::MatrixXd actual_inputs;
  Eigen::MatrixXd expected_states;
  Eigen::MatrixXd expected_inputs;

  MPCInterface mpc;

 private:
  std::unique_ptr<SimType> sim_;
  double control_period_{-1};
  double sim_period_{-1};
};

// NOLINTNEXTLINE
void TestMPCInterface::SetUp() {
  std::ifstream fp(TEST_DATA_FILE);

  ASSERT_TRUE(fp.is_open());

  rapidjson::IStreamWrapper isw(fp);

  rapidjson::Document doc;
  doc.ParseStream(isw);

  ASSERT_NO_THROW(sim_out = ReadMatrixFromJson(doc["sim_out"].GetObject()));
  expected_states = sim_out["states"];
  expected_inputs = sim_out["inputs"];
  actual_inputs.resizeLike(expected_inputs);
  actual_states.resizeLike(expected_states);
  ASSERT_NO_THROW(trajectory =
                      ReadMatrixFromJson(doc["trajectory"].GetObject()));

  double mass;
  ASSERT_NO_THROW(mass = doc["params"]["quadrotor_mass"].GetDouble());
  ASSERT_NO_THROW(control_period_ =
                      doc["params"]["control_period"].GetDouble());
  ASSERT_NO_THROW(sim_period_ = doc["params"]["sim_period"].GetDouble());
  constexpr double kGravAccel = -9.81;
  MdlType model(mass, kGravAccel);
  sim_ = std::make_unique<SimType>(
      model, sim_period_, trajectory["time"].coeff(0),
      trajectory["states"].col(0), trajectory["inputs"].col(0));

  sim_->setSimulationPostUpdateCallback([](auto&& states,
                                           [[maybe_unused]] auto&& input,
                                           [[maybe_unused]] auto _) {
    ASSERT_TRUE(
        fsc::IsClose(states.template segment<4>(3).norm(), 1.0, {1e-4, 1.0}));
  });
  mpc.setConstantParameters(MPCInterface::ParamType{mass});
}

void TestMPCInterface::RunSimulation() {
  const auto& full_state_ref = trajectory["states"];
  auto state_ref_sz = full_state_ref.cols();

  const auto& full_input_ref = trajectory["inputs"];
  auto input_ref_sz = full_input_ref.cols();

  auto n_mpc_nodes = mpc.num_mpc_nodes();
  for (int i = 0; i < trajectory["time"].size(); ++i) {
    actual_states.col(i) = sim_->state();
    auto n_state_ref =
        i + n_mpc_nodes + 1 > state_ref_sz ? state_ref_sz - i : n_mpc_nodes + 1;
    const MPCInterface::StateTrajectoryType state_ref =
        full_state_ref.middleCols(i, n_state_ref);

    auto n_input_ref =
        i + n_mpc_nodes > input_ref_sz ? input_ref_sz - i : n_mpc_nodes;
    const MPCInterface::InputTrajectoryType input_ref =
        full_input_ref.middleCols(i, n_input_ref);

    mpc.setReferenceTrajectory(state_ref, input_ref);
    const MPCInterface::InputType u_setpoint = mpc.optimize(sim_->state());
    double simulation_time = 0.0;

    actual_inputs.col(i) = u_setpoint;
    while (simulation_time < control_period_) {
      simulation_time += sim_->dt();
      sim_->input() = u_setpoint;
      sim_->simulationUpdate();
    }
  }
}

TEST_F(TestMPCInterface, testBounding) {
  const double neg_bound = -10.0;
  const double pos_bound = 10.0;
  mpc.setBounds(MPCInterface::BoundsType::Constant(neg_bound),
                MPCInterface::BoundsType::Constant(pos_bound));
  RunSimulation();
  ASSERT_TRUE((actual_inputs.array() > (1.0 + 1e-5) * neg_bound).all())
      << actual_inputs.minCoeff();
  ASSERT_TRUE((actual_inputs.array() < (1.0 + 1e-5) * pos_bound).all());
}

TEST_F(TestMPCInterface, testPosition) {
  RunSimulation();
  const Eigen::MatrixXd expected_position = expected_states.topRows(3);
  const Eigen::MatrixXd actual_position = actual_states.topRows(3);
  constexpr auto kPositionRelativeTolerance = 100 * kRelativeTolerance;
  Check3DTrajectories(expected_position, actual_position,
                      kPositionRelativeTolerance, kAbsoluteTolerance);
}

TEST_F(TestMPCInterface, testAttitude) {
  RunSimulation();
  for (int i = 0; i < expected_states.cols(); ++i) {
    const Eigen::Quaterniond expected_quat(
        expected_states.template block<4, 1>(3, i));
    const Eigen::Quaterniond actual_quat(
        actual_states.template block<4, 1>(3, i));
    ASSERT_THAT(expected_quat, QuaternionIsClose(actual_quat));
  }
}

TEST_F(TestMPCInterface, testVelocity) {
  RunSimulation();
  const Eigen::MatrixXd expected_vel = expected_states.bottomRows(3);
  const Eigen::MatrixXd actual_vel = expected_states.bottomRows(3);
  Check3DTrajectories(expected_vel, actual_vel, kRelativeTolerance,
                      kAbsoluteTolerance);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
