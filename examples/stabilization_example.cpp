
#include <iostream>

#include "fscore/rigid_body.hpp"
#include "quadrotor_mpcpp/quadrotor_mpc.hpp"

int main() {
  fsc::RigidBody<double, fsc::AttitudeDynamicsMode::kWithoutAttitudeDynamics>
      model(1.0);

  control::AcadosMPC controller;
  control::AcadosMPC::StateType target_state;

  const Eigen::Vector3d target_position = 10.0 * Eigen::Vector3d::UnitZ();
  target_state << target_position, Eigen::Quaterniond::Identity().coeffs(),
      Eigen::Vector3d::Zero();

  controller.setReferenceState(target_state);
  constexpr double kControlPeriod = 0.1;
  double time = 0.0;
  controller.setConstantParameters(control::AcadosMPC::ParamType(1.0));
  control::AcadosMPC::StateType state = model.state();
  while (time < 100.0) {
    state = model.state();
    control::AcadosMPC::InputType input = controller.optimize(state);
    decltype(model)::InputType input_6;
    input_6 << input.coeff(0) * Eigen::Vector3d::UnitZ(), input.tail<3>();
    model.modelUpdate(input_6, kControlPeriod);
    if (model.position().isApprox(target_position)) {
      const Eigen::IOFormat fmt(Eigen::StreamPrecision, 0, ",", ";\n", "", "",
                                "[", "]");
      std::cout << "Model reached target position: "
                << model.position().transpose().format(fmt) << " at time "
                << time << "\n";
      break;
    }
    time += kControlPeriod;
  }
}
