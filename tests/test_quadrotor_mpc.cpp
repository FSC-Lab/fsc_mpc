#include <algorithm>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "gtest/gtest.h"
#define RAPIDJSON_ASSERT(cond)        \
  if (!static_cast<bool>((cond)))     \
  throw std::runtime_error(#cond      \
                           " is not " \
                           "met")

#include "quadrotor_mpcpp/quadrotor_mpc.hpp"
#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"

#ifndef TEST_DATA_FILE
#define TEST_DATA_FILE ""
#error THE_TEST_DATA_FILE_MACRO_MUST_BE_DEFINED
#endif

class TestAcadosMPC : public ::testing::Test {
 public:
  TestAcadosMPC() = default;

  void SetUp() override {
    const std::unique_ptr<FILE, decltype(&fclose)> fp{
        fopen(TEST_DATA_FILE, "r"), &fclose};

    char buf[4096];
    rapidjson::FileReadStream ifs(fp.get(), buf, sizeof buf);

    rapidjson::Document doc;
    doc.ParseStream(ifs);

    for (const auto& it : doc.GetObject()) {
      auto size = it.value["size"].GetArray();
      const int rows = size[0].GetInt();
      const int cols = size[1].GetInt();
      const auto& val = it.value["value"].GetArray();

      const auto& key = it.name.GetString();
      test_data.emplace(key, Eigen::MatrixXd(rows, cols));
      std::transform(val.begin(), val.end(), test_data[key].data(),
                     std::mem_fn(&rapidjson::Value::GetDouble));
    }
  }

  std::unordered_map<std::string, Eigen::MatrixXd> test_data;
};

TEST_F(TestAcadosMPC, testOptimize) {
  control::AcadosMPC wrapper;
  control::AcadosMPC::StateTrajectoryType x_reference;
  control::AcadosMPC::InputTrajectoryType u_reference;
  control::AcadosMPC::StateType x_current;
  control::AcadosMPC::InputType u_optimized;

  wrapper.setConstantParameters(Eigen::Matrix<double, 1, 1>(1.0));
  ASSERT_NO_THROW(x_reference = test_data.at("x_reference"));
  ASSERT_NO_THROW(x_current = test_data.at("x_current"));
  ASSERT_NO_THROW(u_reference = test_data.at("u_reference"));
  ASSERT_NO_THROW(u_optimized = test_data.at("u_optimized"));
  ASSERT_NO_THROW(wrapper.setReferenceTrajectory(x_reference, u_reference));
  const control::AcadosMPC::InputType result_optimized =
      wrapper.optimize(x_current);

  ASSERT_TRUE(result_optimized.isApprox(u_optimized));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
