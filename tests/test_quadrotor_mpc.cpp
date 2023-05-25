#include <algorithm>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "fscore/models/simple_quadrotor.hpp"
#include "fscore/simulation/dynamic_system_simulator.hpp"
#include "gtest/gtest.h"
#include "quadrotor_mpcpp/quadrotor_mpc.hpp"

#define RAPIDJSON_ASSERT(cond)        \
  if (!static_cast<bool>((cond)))     \
  throw std::runtime_error(#cond      \
                           " is not " \
                           "met")

#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"

#ifndef TEST_DATA_FILE
#define TEST_DATA_FILE ""
#error THE_TEST_DATA_FILE_MACRO_MUST_BE_DEFINED
#endif

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

class TestAcadosMPC : public ::testing::Test {
 public:
  using MdlType = fsc::SimpleQuadrotor<double>;
  using SimType = fsc::DynamicSystemSimulator<MdlType, fsc::ODE4>;
  TestAcadosMPC() = default;

  void SetUp() override;

  void RunSimulation();

  std::unordered_map<std::string, Eigen::MatrixXd> sim_out;
  std::unordered_map<std::string, Eigen::MatrixXd> trajectory;
  Eigen::MatrixXd result_states;
  Eigen::MatrixXd result_inputs;
  Eigen::MatrixXd expected_states;
  Eigen::MatrixXd expected_inputs;

 private:
  control::AcadosMPC mpc_;
  std::unique_ptr<SimType> sim_;
  double control_period_{-1};
  double sim_period_{-1};
};

void TestAcadosMPC::SetUp() {
  std::ifstream fp(TEST_DATA_FILE);

  ASSERT_TRUE(fp.is_open());

  rapidjson::IStreamWrapper isw(fp);

  rapidjson::Document doc;
  doc.ParseStream(isw);

  ASSERT_NO_THROW(sim_out = ReadMatrixFromJson(doc["sim_out"].GetObject()));
  expected_states = sim_out["states"];
  expected_inputs = sim_out["inputs"];
  result_inputs.resizeLike(expected_inputs);
  result_states.resizeLike(expected_states);
  ASSERT_NO_THROW(trajectory =
                      ReadMatrixFromJson(doc["trajectory"].GetObject()));

  double mass;
  ASSERT_NO_THROW(mass = doc["params"]["quadrotor_mass"].GetDouble());
  ASSERT_NO_THROW(control_period_ =
                      doc["params"]["control_period"].GetDouble());
  ASSERT_NO_THROW(sim_period_ = doc["params"]["sim_period"].GetDouble());
  MdlType model(mass, -9.81);
  sim_ = std::make_unique<SimType>(model, 5e-4, trajectory["time"].coeff(0),
                                   trajectory["states"].col(0),
                                   trajectory["inputs"].col(0));
  mpc_.setConstantParameters(control::AcadosMPC::ParamType{mass});

  RunSimulation();
}

void TestAcadosMPC::RunSimulation() {
  const auto& full_state_ref = trajectory["states"];
  auto state_ref_sz = full_state_ref.cols();

  const auto& full_input_ref = trajectory["inputs"];
  auto input_ref_sz = full_input_ref.cols();

  auto n_mpc_nodes = mpc_.num_mpc_nodes();
  for (int i = 0; i < trajectory["time"].size(); ++i) {
    result_states.col(i) = sim_->state();
    auto n_state_ref =
        i + n_mpc_nodes + 1 > state_ref_sz ? state_ref_sz - i : n_mpc_nodes + 1;
    const control::AcadosMPC::StateTrajectoryType state_ref =
        full_state_ref.middleCols(i, n_state_ref);

    auto n_input_ref =
        i + n_mpc_nodes > input_ref_sz ? input_ref_sz - i : n_mpc_nodes;
    const control::AcadosMPC::InputTrajectoryType input_ref =
        full_input_ref.middleCols(i, n_input_ref);

    mpc_.setReferenceTrajectory(state_ref, input_ref);
    const control::AcadosMPC::InputType u_setpoint =
        mpc_.optimize(sim_->state());
    double simulation_time = 0.0;

    result_inputs.col(i) = u_setpoint;
    while (simulation_time < control_period_) {
      simulation_time += sim_->dt();
      sim_->input() = u_setpoint;
      sim_->simulationUpdate();
    }
  }
}

TEST_F(TestAcadosMPC, testOptimize) {
  ASSERT_TRUE(result_states.isApprox(expected_states, 1e-2));
  ASSERT_TRUE(result_inputs.isApprox(expected_inputs, 1e-2));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
