/******************************************************************************
 * Copyright (c) 2025 AISS Group at Harbin Institute of Technology. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/


#include <asnumpy/random/distributions.hpp>
#include <asnumpy/utils/npu_array.hpp>
#include <asnumpy/utils/acl_executor.hpp>
#include <asnumpy/utils/acl_resource.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnnop/aclnn_normal_out.h>
#include <aclnnop/aclnn_normal.h>
#include <aclnnop/aclnn_uniform.h>
#include <aclnnop/aclnn_rsub.h>
#include <aclnnop/aclnn_pow.h>
#include <aclnnop/aclnn_reciprocal.h>
#include <aclnnop/aclnn_log.h>
#include <aclnnop/aclnn_mul.h>
#include <aclnnop/aclnn_sqrt.h>
#include <aclnnop/aclnn_sub.h>
#include <aclnnop/aclnn_tan.h>
#include <aclnnop/aclnn_bernoulli.h>
#include <aclnnop/aclnn_reduce_sum.h>
#include <aclnnop/aclnn_floor.h>
#include <aclnnop/aclnn_cast.h>
#include <aclnnop/aclnn_foreach_sub_scalar.h>
#include <aclnnop/aclnn_foreach_mul_scalar.h>
#include <aclnnop/aclnn_foreach_mul_scalar.h>
#include <aclnnop/aclnn_foreach_div_scalar.h>
#include <aclnnop/aclnn_abs.h>
#include <aclnnop/aclnn_foreach_add_scalar.h>
#include <aclnnop/aclnn_foreach_mul_scalar.h>
#include <aclnnop/aclnn_exp.h> 
#include <aclnnop/aclnn_sign.h>
#include <aclnnop/aclnn_multinomial.h>
#include <aclnnop/aclnn_div.h>
#include <aclnnop/aclnn_add.h>

#include <fmt/core.h>
#include <fmt/format.h>
#include <pybind11/attr.h>
#include <stdexcept>
#include <random>

namespace asnumpy {

NPUArray Generator_Pareto(float a, const std::vector<int64_t>& size) {
    if (a <= 0) throw std::invalid_argument(fmt::format(
        "[distributions.cpp]({}) invalid parameter: a={} <= 0", __func__, a));

    auto uni_temp = NPUArray(size, ACL_FLOAT);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(1, 100);
    uint64_t seed = dist(gen);
    uint64_t offset = 0;
    uint64_t uni_workspaceSize = 0;
    aclOpExecutor* uni_executor;
    LOG_DEBUG("aclnnInplaceUniform start: shape={}", detail::FormatShape(size));
    auto error = aclnnInplaceUniformGetWorkspaceSize(uni_temp.tensorPtr, 0.0, 1.0, seed, offset,
        &uni_workspaceSize, &uni_executor);
    ACLNN_CHECK(error, "aclnnInplaceUniformGetWorkspaceSize");
    AclWorkspace uni_workspace(uni_workspaceSize);
    error = aclnnInplaceUniform(uni_workspace.get(), uni_workspaceSize, uni_executor, nullptr);
    ACLNN_CHECK(error, "aclnnInplaceUniform");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnInplaceUniform completed");

    auto rsubs_temp = NPUArray(size, ACL_FLOAT);
    float scalar1 = 1.0f;
    float scalar2 = 1.0f;
    aclScalar* other = aclCreateScalar(&scalar1, ACL_FLOAT);
    aclScalar* alpha = aclCreateScalar(&scalar2, ACL_FLOAT);
    uint64_t rsubs_workspaceSize = 0;
    aclOpExecutor* rsubs_executor;
    LOG_DEBUG("aclnnRsubs start: shape={}", detail::FormatShape(size));
    error = aclnnRsubsGetWorkspaceSize(uni_temp.tensorPtr, other, alpha, rsubs_temp.tensorPtr,
        &rsubs_workspaceSize, &rsubs_executor);
    ACLNN_CHECK(error, "aclnnRsubsGetWorkspaceSize");
    AclWorkspace rsubs_workspace(rsubs_workspaceSize);
    error = aclnnRsubs(rsubs_workspace.get(), rsubs_workspaceSize, rsubs_executor, nullptr);
    ACLNN_CHECK(error, "aclnnRsubs");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnRsubs completed");

    auto result = NPUArray(size, ACL_FLOAT);
    float scalar3 = 1.0f / a;
    aclScalar* exponent = aclCreateScalar(&scalar3, ACL_FLOAT);
    uint64_t exp_workspaceSize = 0;
    aclOpExecutor* exp_executor;
    LOG_DEBUG("aclnnPowTensorScalar start: shape={}, a={}", detail::FormatShape(size), a);
    error = aclnnPowTensorScalarGetWorkspaceSize(rsubs_temp.tensorPtr, exponent, result.tensorPtr,
        &exp_workspaceSize, &exp_executor);
    ACLNN_CHECK(error, "aclnnPowTensorScalarGetWorkspaceSize");
    AclWorkspace exp_workspace(exp_workspaceSize);
    error = aclnnPowTensorScalar(exp_workspace.get(), exp_workspaceSize, exp_executor, nullptr);
    ACLNN_CHECK(error, "aclnnPowTensorScalar");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnPowTensorScalar completed");

    uint64_t reci_workspaceSize = 0;
    aclOpExecutor* reci_executor;
    LOG_DEBUG("aclnnInplaceReciprocal start: shape={}", detail::FormatShape(size));
    error = aclnnInplaceReciprocalGetWorkspaceSize(result.tensorPtr, &reci_workspaceSize, &reci_executor);
    ACLNN_CHECK(error, "aclnnInplaceReciprocalGetWorkspaceSize");
    AclWorkspace reci_workspace(reci_workspaceSize);
    error = aclnnInplaceReciprocal(reci_workspace.get(), reci_workspaceSize, reci_executor, nullptr);
    ACLNN_CHECK(error, "aclnnInplaceReciprocal");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnInplaceReciprocal completed");

    uint64_t sub_workspaceSize = 0;
    aclOpExecutor* sub_executor;
    LOG_DEBUG("aclnnInplaceSubs start: shape={}", detail::FormatShape(size));
    error = aclnnInplaceSubsGetWorkspaceSize(result.tensorPtr, other, alpha, &sub_workspaceSize, &sub_executor);
    ACLNN_CHECK(error, "aclnnInplaceSubsGetWorkspaceSize");
    AclWorkspace sub_workspace(sub_workspaceSize);
    error = aclnnInplaceSubs(sub_workspace.get(), sub_workspaceSize, sub_executor, nullptr);
    ACLNN_CHECK(error, "aclnnInplaceSubs");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnInplaceSubs completed");
    return result;
}

NPUArray Generator_Rayleigh(float scale, const std::vector<int64_t>& size) {
    auto uni_temp = NPUArray(size, ACL_FLOAT);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(1, 100);
    uint64_t seed = dist(gen);
    uint64_t offset = 0;
    uint64_t uni_workspaceSize = 0;
    aclOpExecutor* uni_executor;
    LOG_DEBUG("aclnnInplaceUniform start: shape={}", detail::FormatShape(size));
    auto error = aclnnInplaceUniformGetWorkspaceSize(uni_temp.tensorPtr, 0.0, 1.0, seed, offset,
        &uni_workspaceSize, &uni_executor);
    ACLNN_CHECK(error, "aclnnInplaceUniformGetWorkspaceSize");
    AclWorkspace uni_workspace(uni_workspaceSize);
    error = aclnnInplaceUniform(uni_workspace.get(), uni_workspaceSize, uni_executor, nullptr);
    ACLNN_CHECK(error, "aclnnInplaceUniform");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnInplaceUniform completed");

    auto result = NPUArray(size, ACL_FLOAT);
    float scalar1 = 1.0f;
    float scalar2 = 1.0f;
    aclScalar* other = aclCreateScalar(&scalar1, ACL_FLOAT);
    aclScalar* alpha = aclCreateScalar(&scalar2, ACL_FLOAT);
    uint64_t rsubs_workspaceSize = 0;
    aclOpExecutor* rsubs_executor;
    LOG_DEBUG("aclnnRsubs start: shape={}", detail::FormatShape(size));
    error = aclnnRsubsGetWorkspaceSize(uni_temp.tensorPtr, other, alpha, result.tensorPtr,
        &rsubs_workspaceSize, &rsubs_executor);
    ACLNN_CHECK(error, "aclnnRsubsGetWorkspaceSize");
    AclWorkspace rsubs_workspace(rsubs_workspaceSize);
    error = aclnnRsubs(rsubs_workspace.get(), rsubs_workspaceSize, rsubs_executor, nullptr);
    ACLNN_CHECK(error, "aclnnRsubs");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnRsubs completed");

    uint64_t log_workspaceSize = 0;
    aclOpExecutor* log_executor;
    LOG_DEBUG("aclnnInplaceLog start: shape={}", detail::FormatShape(size));
    error = aclnnInplaceLogGetWorkspaceSize(result.tensorPtr, &log_workspaceSize, &log_executor);
    ACLNN_CHECK(error, "aclnnInplaceLogGetWorkspaceSize");
    AclWorkspace log_workspace(log_workspaceSize);
    error = aclnnInplaceLog(log_workspace.get(), log_workspaceSize, log_executor, nullptr);
    ACLNN_CHECK(error, "aclnnInplaceLog");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnInplaceLog completed");

    float scalar3 = -2.0f;
    aclScalar* mulnum = aclCreateScalar(&scalar3, ACL_FLOAT);
    uint64_t muls_workspaceSize = 0;
    aclOpExecutor* muls_executor;
    LOG_DEBUG("aclnnInplaceMuls start: shape={}", detail::FormatShape(size));
    error = aclnnInplaceMulsGetWorkspaceSize(result.tensorPtr, mulnum, &muls_workspaceSize, &muls_executor);
    ACLNN_CHECK(error, "aclnnInplaceMulsGetWorkspaceSize");
    AclWorkspace muls_workspace(muls_workspaceSize);
    error = aclnnInplaceMuls(muls_workspace.get(), muls_workspaceSize, muls_executor, nullptr);
    ACLNN_CHECK(error, "aclnnInplaceMuls");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnInplaceMuls completed");

    uint64_t sqrt_workspaceSize = 0;
    aclOpExecutor* sqrt_executor;
    LOG_DEBUG("aclnnInplaceSqrt start: shape={}", detail::FormatShape(size));
    error = aclnnInplaceSqrtGetWorkspaceSize(result.tensorPtr, &sqrt_workspaceSize, &sqrt_executor);
    ACLNN_CHECK(error, "aclnnInplaceSqrtGetWorkspaceSize");
    AclWorkspace sqrt_workspace(sqrt_workspaceSize);
    error = aclnnInplaceSqrt(sqrt_workspace.get(), sqrt_workspaceSize, sqrt_executor, nullptr);
    ACLNN_CHECK(error, "aclnnInplaceSqrt");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnInplaceSqrt completed");

    aclScalar* muls = aclCreateScalar(&scale, ACL_FLOAT);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    LOG_DEBUG("aclnnInplaceMuls start: shape={}, scale={}", detail::FormatShape(size), scale);
    error = aclnnInplaceMulsGetWorkspaceSize(result.tensorPtr, muls, &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnInplaceMulsGetWorkspaceSize");
    AclWorkspace workspace(workspaceSize);
    error = aclnnInplaceMuls(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnInplaceMuls");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnInplaceMuls completed");
    return result;
}

NPUArray Generator_Normal(float loc, float scale, const std::vector<int64_t>& size) {
    LOG_DEBUG("aclnnNormalFloatFloat start: shape={}, loc={}, scale={}", detail::FormatShape(size), loc, scale);
    auto result = NPUArray(size, ACL_DOUBLE);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(1, 100);
    int64_t seed = dist(gen);
    int64_t offset = 0;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnNormalFloatFloatGetWorkspaceSize(loc, scale, seed, offset, result.tensorPtr,
        &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnNormalFloatFloatGetWorkspaceSize");
    AclWorkspace workspace(workspaceSize);
    error = aclnnNormalFloatFloat(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnNormalFloatFloat");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnNormalFloatFloat completed");
    return result;
}

NPUArray Generator_Uniform(double low, double high, const std::vector<int64_t>& size) {
    LOG_DEBUG("aclnnInplaceUniform start: shape={}, low={}, high={}", detail::FormatShape(size), low, high);
    auto result = NPUArray(size, ACL_DOUBLE);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(1, 100);
    uint64_t seed = dist(gen);
    uint64_t offset = 0;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnInplaceUniformGetWorkspaceSize(result.tensorPtr, low, high, seed, offset,
        &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnInplaceUniformGetWorkspaceSize");
    AclWorkspace workspace(workspaceSize);
    error = aclnnInplaceUniform(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnInplaceUniform");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnInplaceUniform completed");
    return result;
}

NPUArray Generator_Standard_normal(const std::vector<int64_t>& size) {
    LOG_DEBUG("aclnnNormalFloatFloat start: shape={}", detail::FormatShape(size));
    float loc = 0.0f;
    float scale = 1.0f;
    auto result = NPUArray(size, ACL_DOUBLE);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(1, 100);
    int64_t seed = dist(gen);
    int64_t offset = 0;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnNormalFloatFloatGetWorkspaceSize(loc, scale, seed, offset, result.tensorPtr,
        &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnNormalFloatFloatGetWorkspaceSize");
    AclWorkspace workspace(workspaceSize);
    error = aclnnNormalFloatFloat(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnNormalFloatFloat");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnNormalFloatFloat completed");
    return result;
}

NPUArray Generator_Standard_cauchy(const std::vector<int64_t>& size) {
    auto result = NPUArray(size, ACL_DOUBLE);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(1, 100);
    uint64_t seed = dist(gen);
    uint64_t offset = 0;
    uint64_t uni_workspaceSize = 0;
    aclOpExecutor* uni_executor;
    LOG_DEBUG("aclnnInplaceUniform start: shape={}", detail::FormatShape(size));
    auto error = aclnnInplaceUniformGetWorkspaceSize(result.tensorPtr, 0.0, 1.0, seed, offset,
        &uni_workspaceSize, &uni_executor);
    ACLNN_CHECK(error, "aclnnInplaceUniformGetWorkspaceSize");
    AclWorkspace uni_workspace(uni_workspaceSize);
    error = aclnnInplaceUniform(uni_workspace.get(), uni_workspaceSize, uni_executor, nullptr);
    ACLNN_CHECK(error, "aclnnInplaceUniform");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnInplaceUniform completed");

    float scalar1 = 0.5f;
    float scalar2 = 1.0f;
    aclScalar* other = aclCreateScalar(&scalar1, ACL_FLOAT);
    aclScalar* alpha = aclCreateScalar(&scalar2, ACL_FLOAT);
    uint64_t subs_workspaceSize = 0;
    aclOpExecutor* subs_executor;
    LOG_DEBUG("aclnnInplaceSubs start: shape={}", detail::FormatShape(size));
    error = aclnnInplaceSubsGetWorkspaceSize(result.tensorPtr, other, alpha, &subs_workspaceSize, &subs_executor);
    ACLNN_CHECK(error, "aclnnInplaceSubsGetWorkspaceSize");
    AclWorkspace subs_workspace(subs_workspaceSize);
    error = aclnnInplaceSubs(subs_workspace.get(), subs_workspaceSize, subs_executor, nullptr);
    ACLNN_CHECK(error, "aclnnInplaceSubs");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnInplaceSubs completed");

    double PI = 3.141592653589793238462643383279502884197169399375105820974944;
    aclScalar* pi = aclCreateScalar(&PI, ACL_DOUBLE);
    uint64_t muls_workspaceSize = 0;
    aclOpExecutor* muls_executor;
    LOG_DEBUG("aclnnInplaceMuls start: shape={}", detail::FormatShape(size));
    error = aclnnInplaceMulsGetWorkspaceSize(result.tensorPtr, pi, &muls_workspaceSize, &muls_executor);
    ACLNN_CHECK(error, "aclnnInplaceMulsGetWorkspaceSize");
    AclWorkspace muls_workspace(muls_workspaceSize);
    error = aclnnInplaceMuls(muls_workspace.get(), muls_workspaceSize, muls_executor, nullptr);
    ACLNN_CHECK(error, "aclnnInplaceMuls");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnInplaceMuls completed");

    uint64_t tan_workspaceSize = 0;
    aclOpExecutor* tan_executor;
    LOG_DEBUG("aclnnInplaceTan start: shape={}", detail::FormatShape(size));
    error = aclnnInplaceTanGetWorkspaceSize(result.tensorPtr, &tan_workspaceSize, &tan_executor);
    ACLNN_CHECK(error, "aclnnInplaceTanGetWorkspaceSize");
    AclWorkspace tan_workspace(tan_workspaceSize);
    error = aclnnInplaceTan(tan_workspace.get(), tan_workspaceSize, tan_executor, nullptr);
    ACLNN_CHECK(error, "aclnnInplaceTan");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnInplaceTan completed");
    return result;
}

NPUArray Generator_Weibull(float a, const std::vector<int64_t>& size) {
    auto uni_temp = NPUArray(size, ACL_FLOAT);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(1, 100);
    uint64_t seed = dist(gen);
    uint64_t offset = 0;
    uint64_t uni_workspaceSize = 0;
    aclOpExecutor* uni_executor;
    LOG_DEBUG("aclnnInplaceUniform start: shape={}, a={}", detail::FormatShape(size), a);
    auto error = aclnnInplaceUniformGetWorkspaceSize(uni_temp.tensorPtr, 0.0, 1.0, seed, offset,
        &uni_workspaceSize, &uni_executor);
    ACLNN_CHECK(error, "aclnnInplaceUniformGetWorkspaceSize");
    AclWorkspace uni_workspace(uni_workspaceSize);
    error = aclnnInplaceUniform(uni_workspace.get(), uni_workspaceSize, uni_executor, nullptr);
    ACLNN_CHECK(error, "aclnnInplaceUniform");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnInplaceUniform completed");

    auto rsubs_temp = NPUArray(size, ACL_FLOAT);
    float scalar1 = 1.0f;
    float scalar2 = 1.0f;
    aclScalar* other1 = aclCreateScalar(&scalar1, ACL_FLOAT);
    aclScalar* alpha = aclCreateScalar(&scalar2, ACL_FLOAT);
    uint64_t rsubs_workspaceSize1 = 0;
    aclOpExecutor* rsubs_executor1;
    LOG_DEBUG("aclnnRsubs start: shape={}", detail::FormatShape(size));
    error = aclnnRsubsGetWorkspaceSize(uni_temp.tensorPtr, other1, alpha, rsubs_temp.tensorPtr,
        &rsubs_workspaceSize1, &rsubs_executor1);
    ACLNN_CHECK(error, "aclnnRsubsGetWorkspaceSize");
    AclWorkspace rsubs_workspace1(rsubs_workspaceSize1);
    error = aclnnRsubs(rsubs_workspace1.get(), rsubs_workspaceSize1, rsubs_executor1, nullptr);
    ACLNN_CHECK(error, "aclnnRsubs");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnRsubs completed");

    uint64_t log_workspaceSize = 0;
    aclOpExecutor* log_executor;
    LOG_DEBUG("aclnnInplaceLog start: shape={}", detail::FormatShape(size));
    error = aclnnInplaceLogGetWorkspaceSize(rsubs_temp.tensorPtr, &log_workspaceSize, &log_executor);
    ACLNN_CHECK(error, "aclnnInplaceLogGetWorkspaceSize");
    AclWorkspace log_workspace(log_workspaceSize);
    error = aclnnInplaceLog(log_workspace.get(), log_workspaceSize, log_executor, nullptr);
    ACLNN_CHECK(error, "aclnnInplaceLog");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnInplaceLog completed");

    auto result = NPUArray(size, ACL_FLOAT);
    float scalar3 = 0.0f;
    aclScalar* other2 = aclCreateScalar(&scalar3, ACL_FLOAT);
    uint64_t rsubs_workspaceSize = 0;
    aclOpExecutor* rsubs_executor;
    LOG_DEBUG("aclnnRsubs start: shape={}", detail::FormatShape(size));
    error = aclnnRsubsGetWorkspaceSize(rsubs_temp.tensorPtr, other2, alpha, result.tensorPtr,
        &rsubs_workspaceSize, &rsubs_executor);
    ACLNN_CHECK(error, "aclnnRsubsGetWorkspaceSize");
    AclWorkspace rsubs_workspace(rsubs_workspaceSize);
    error = aclnnRsubs(rsubs_workspace.get(), rsubs_workspaceSize, rsubs_executor, nullptr);
    ACLNN_CHECK(error, "aclnnRsubs");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnRsubs completed");

    float scalar4 = 1.0f / a;
    aclScalar* exponent = aclCreateScalar(&scalar4, ACL_FLOAT);
    uint64_t exp_workspaceSize = 0;
    aclOpExecutor* exp_executor;
    LOG_DEBUG("aclnnPowTensorScalar start: shape={}, a={}", detail::FormatShape(size), a);
    error = aclnnPowTensorScalarGetWorkspaceSize(result.tensorPtr, exponent, result.tensorPtr, &exp_workspaceSize, &exp_executor);
    ACLNN_CHECK(error, "aclnnPowTensorScalarGetWorkspaceSize");
    AclWorkspace exp_workspace(exp_workspaceSize);
    error = aclnnPowTensorScalar(exp_workspace.get(), exp_workspaceSize, exp_executor, nullptr);
    ACLNN_CHECK(error, "aclnnPowTensorScalar");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnPowTensorScalar completed");
    return result;
}

NPUArray Binomial(int n, float p, const std::vector<int64_t>& size) {
    // 1. 参数校验
    if (n < 0) throw std::invalid_argument(fmt::format(
        "[distributions.cpp]({}) invalid parameter: n={} < 0", __func__, n));
    if (p < 0.0f || p > 1.0f) throw std::invalid_argument(fmt::format(
        "[distributions.cpp]({}) invalid parameter: p={} not in [0,1]", __func__, p));
    if (n == 0) {
        NPUArray result(size, ACL_INT32);
        void* data_ptr = nullptr;
        auto ret = aclGetRawTensorAddr(result.tensorPtr, &data_ptr);
        ACL_RT_CHECK(ret, "aclGetRawTensorAddr");
        if (!data_ptr) {
            throw std::runtime_error(fmt::format(
                "[distributions.cpp]({}) aclGetRawTensorAddr returned null data_ptr", __func__));
        }
        // 用0初始化整个输出张量
        size_t total_elems = std::accumulate(size.begin(), size.end(), int64_t{1}, std::multiplies<int64_t>());
        ret = aclrtMemset(data_ptr, total_elems * sizeof(int32_t), 0, total_elems * sizeof(int32_t));
        ACL_RT_CHECK(ret, "aclrtMemset");
        return result;
    }

    // 2. 构造伯努利张量形状
    std::vector<int64_t> bernoulli_shape = {static_cast<int64_t>(n)};
    bernoulli_shape.insert(bernoulli_shape.end(), size.begin(), size.end());
    NPUArray bernoulli_tensor(bernoulli_shape, ACL_INT32);

    // 3. 构造标量概率张量
    NPUArray prob_tensor({}, ACL_FLOAT);
    void* prob_data_ptr = nullptr;
    auto ret = aclGetRawTensorAddr(prob_tensor.tensorPtr, &prob_data_ptr);
    ACL_RT_CHECK(ret, "aclGetRawTensorAddr");
    if (!prob_data_ptr) {
        throw std::runtime_error(fmt::format(
            "[distributions.cpp]({}) aclGetRawTensorAddr returned null prob_data_ptr", __func__));
    }
    ret = aclrtMemcpy(prob_data_ptr, sizeof(float), &p, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    ACL_RT_CHECK(ret, "aclrtMemcpy");

    // 4. 声明资源并创建流
    uint64_t bernoulli_ws = 0;
    aclOpExecutor* bernoulli_exec = nullptr;
    aclrtStream stream = nullptr;
    ret = aclrtCreateStream(&stream);
    ACL_RT_CHECK(ret, "aclrtCreateStream");
    if (!stream) throw std::runtime_error(fmt::format(
        "[distributions.cpp]({}) aclrtCreateStream returned null", __func__));

    // 5. 生成伯努利张量
    LOG_DEBUG("aclnnBernoulliTensor start: shape={}, n={}, p={}", detail::FormatShape(size), n, p);
    ret = aclnnBernoulliTensorGetWorkspaceSize(
        bernoulli_tensor.tensorPtr, prob_tensor.tensorPtr, 42, 0,
        bernoulli_tensor.tensorPtr, &bernoulli_ws, &bernoulli_exec
    );
    ACLNN_CHECK(ret, "aclnnBernoulliTensorGetWorkspaceSize");
    AclWorkspace bernoulli(bernoulli_ws);
    ret = aclnnBernoulliTensor(bernoulli.get(), bernoulli_ws, bernoulli_exec, stream);
    ACLNN_CHECK(ret, "aclnnBernoulliTensor");
    ret = aclrtSynchronizeStream(stream);
    ACL_RT_CHECK(ret, "aclrtSynchronizeStream");
    LOG_INFO("aclnnBernoulliTensor completed");

    // 6. 归约求和（核心修正部分）
    NPUArray result(size, ACL_INT32);
    uint64_t sum_ws = 0;
    aclOpExecutor* sum_exec = nullptr;
    std::vector<int64_t> reduce_axis = {0};  // 沿第0维求和

    // 6.1 创建aclIntArray类型的归约轴（适配接口要求）
    aclIntArray* dims_array = aclCreateIntArray(reduce_axis.data(), reduce_axis.size());
    if (dims_array == nullptr) {
        aclrtDestroyStream(stream);
        throw std::runtime_error(fmt::format(
            "[distributions.cpp]({}) aclCreateIntArray returned null", __func__));
    }

    // 6.2 调用修正后的归约求和接口（按文档参数顺序）
    LOG_DEBUG("aclnnReduceSum start: shape={}", detail::FormatShape(size));
    ret = aclnnReduceSumGetWorkspaceSize(
        bernoulli_tensor.tensorPtr,  // 输入张量
        dims_array,                  // 归约轴（aclIntArray类型）
        false,                       // 是否保留归约轴
        ACL_INT32,                   // 输出数据类型（匹配result的类型）
        result.tensorPtr,            // 输出张量
        &sum_ws,
        &sum_exec
    );
    ACLNN_CHECK(ret, "aclnnReduceSumGetWorkspaceSize");

    // 6.3 分配求和工作空间并执行
    AclWorkspace sumws(sum_ws);
    ret = aclnnReduceSum(sumws.get(), sum_ws, sum_exec, stream);
    ACLNN_CHECK(ret, "aclnnReduceSum");
    ret = aclrtSynchronizeStream(stream);
    ACL_RT_CHECK(ret, "aclrtSynchronizeStream");

    // 7. 释放所有资源
    aclDestroyIntArray(dims_array);  // 销毁归约轴数组
    aclrtDestroyStream(stream);

    LOG_INFO("aclnnReduceSum completed");
    return result;
}


NPUArray Exponential(float scale, const std::vector<int64_t>& size) {
    // 1. 参数校验
    if (scale <= 0.0f) {
        throw std::invalid_argument(fmt::format(
            "[distributions.cpp]({}) invalid parameter: scale={} <= 0", __func__, scale));
    }

    // 2. 构造均匀分布张量 U
    NPUArray u_tensor(size, ACL_FLOAT);

    uint64_t uniform_ws = 0;
    aclOpExecutor* uniform_exec = nullptr;
    aclrtStream stream = nullptr;
    auto ret = aclrtCreateStream(&stream);
    ACL_RT_CHECK(ret, "aclrtCreateStream");
    if (!stream) {
        throw std::runtime_error(fmt::format(
            "[distributions.cpp]({}) aclrtCreateStream returned null", __func__));
    }

    double low = 0.0;
    double high = 1.0;
    uint64_t seed = 12345;
    uint64_t offset = 0;

    // 均匀分布 in-place 填充 U
    LOG_DEBUG("aclnnInplaceUniform start: shape={}, scale={}", detail::FormatShape(size), scale);
    ret = aclnnInplaceUniformGetWorkspaceSize(
        u_tensor.tensorPtr, low, high, seed, offset,
        &uniform_ws, &uniform_exec
    );
    ACLNN_CHECK(ret, "aclnnInplaceUniformGetWorkspaceSize");
    AclWorkspace uniform(uniform_ws);
    ret = aclnnInplaceUniform(uniform.get(), uniform_ws, uniform_exec, stream);
    ACLNN_CHECK(ret, "aclnnInplaceUniform");
    ret = aclrtSynchronizeStream(stream);
    ACL_RT_CHECK(ret, "aclrtSynchronizeStream");
    LOG_INFO("aclnnInplaceUniform completed");

    // 3. 计算 1 - U
    NPUArray one_tensor({}, ACL_FLOAT);
    void* one_data = nullptr;
    ret = aclGetRawTensorAddr(one_tensor.tensorPtr, &one_data);
    ACL_RT_CHECK(ret, "aclGetRawTensorAddr");
    if (!one_data) {
        aclrtDestroyStream(stream);
        throw std::runtime_error(fmt::format(
            "[distributions.cpp]({}) aclGetRawTensorAddr returned null one_data", __func__));
    }
    float one_val = 1.0f;
    ret = aclrtMemcpy(one_data, sizeof(float), &one_val, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    ACL_RT_CHECK(ret, "aclrtMemcpy");

    NPUArray one_minus_u(size, ACL_FLOAT);
    uint64_t sub_ws = 0;
    aclOpExecutor* sub_exec = nullptr;
    void* sub_ws_addr = nullptr;

    aclScalar* alpha_scalar = aclCreateScalar(&one_val, ACL_FLOAT); // alpha = 1

    LOG_DEBUG("aclnnSub start: shape={}", detail::FormatShape(size));
    ret = aclnnSubGetWorkspaceSize(
        one_tensor.tensorPtr,     // self
        u_tensor.tensorPtr,       // other
        alpha_scalar,             // alpha
        one_minus_u.tensorPtr,    // out
        &sub_ws,
        &sub_exec
    );
    ACLNN_CHECK(ret, "aclnnSubGetWorkspaceSize");
    AclWorkspace subws(sub_ws);
    ret = aclnnSub(subws.get(), sub_ws, sub_exec, stream);
    ACLNN_CHECK(ret, "aclnnSub");
    ret = aclrtSynchronizeStream(stream);
    ACL_RT_CHECK(ret, "aclrtSynchronizeStream");
    aclDestroyScalar(alpha_scalar);
    LOG_INFO("aclnnSub completed");

    // 4. log(1 - U)
    NPUArray log_tensor(size, ACL_FLOAT);
    uint64_t log_ws = 0;
    aclOpExecutor* log_exec = nullptr;
    LOG_DEBUG("aclnnLog start: shape={}", detail::FormatShape(size));
    ret = aclnnLogGetWorkspaceSize(one_minus_u.tensorPtr, log_tensor.tensorPtr, &log_ws, &log_exec);
    ACLNN_CHECK(ret, "aclnnLogGetWorkspaceSize");
    AclWorkspace logws(log_ws);
    ret = aclnnLog(logws.get(), log_ws, log_exec, stream);
    ACLNN_CHECK(ret, "aclnnLog");
    ret = aclrtSynchronizeStream(stream);
    ACL_RT_CHECK(ret, "aclrtSynchronizeStream");
    LOG_INFO("aclnnLog completed");

    // 5. result = -scale * log(1 - U)
    NPUArray scale_tensor({}, ACL_FLOAT);
    void* scale_data = nullptr;
    ret = aclGetRawTensorAddr(scale_tensor.tensorPtr, &scale_data);
    ACL_RT_CHECK(ret, "aclGetRawTensorAddr");
    if (!scale_data) {
        aclrtDestroyStream(stream);
        throw std::runtime_error(fmt::format(
            "[distributions.cpp]({}) aclGetRawTensorAddr returned null scale_data", __func__));
    }
    float neg_scale = -scale;
    ret = aclrtMemcpy(scale_data, sizeof(float), &neg_scale, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    ACL_RT_CHECK(ret, "aclrtMemcpy");

    NPUArray result(size, ACL_FLOAT);
    uint64_t mul_ws = 0;
    aclOpExecutor* mul_exec = nullptr;
    LOG_DEBUG("aclnnMul start: shape={}, scale={}", detail::FormatShape(size), scale);
    ret = aclnnMulGetWorkspaceSize(scale_tensor.tensorPtr, log_tensor.tensorPtr, result.tensorPtr, &mul_ws, &mul_exec);
    ACLNN_CHECK(ret, "aclnnMulGetWorkspaceSize");
    AclWorkspace mulws(mul_ws);
    ret = aclnnMul(mulws.get(), mul_ws, mul_exec, stream);
    ACLNN_CHECK(ret, "aclnnMul");
    ret = aclrtSynchronizeStream(stream);
    ACL_RT_CHECK(ret, "aclrtSynchronizeStream");

    // 6. 清理资源
    aclrtDestroyStream(stream);

    LOG_INFO("aclnnMul completed");
    return result;
}


NPUArray Geometric(float p, const std::vector<int64_t>& size) {
    // 1. 参数校验
    if (p <= 0.0f || p >= 1.0f) {
        throw std::invalid_argument(fmt::format(
            "[distributions.cpp]({}) invalid parameter: p={} not in (0,1)", __func__, p));
    }

    // 2. 构造均匀分布张量 U
    NPUArray u_tensor(size, ACL_FLOAT);

    uint64_t uniform_ws = 0;
    aclOpExecutor* uniform_exec = nullptr;
    aclrtStream stream = nullptr;
    auto ret = aclrtCreateStream(&stream);
    ACL_RT_CHECK(ret, "aclrtCreateStream");
    if (!stream) {
        throw std::runtime_error(fmt::format(
            "[distributions.cpp]({}) aclrtCreateStream returned null", __func__));
    }

    double low = 0.0;
    double high = 1.0;
    uint64_t seed = 12345;
    uint64_t offset = 0;

    LOG_DEBUG("aclnnInplaceUniform start: shape={}, p={}", detail::FormatShape(size), p);
    ret = aclnnInplaceUniformGetWorkspaceSize(
        u_tensor.tensorPtr, low, high, seed, offset,
        &uniform_ws, &uniform_exec
    );
    ACLNN_CHECK(ret, "aclnnInplaceUniformGetWorkspaceSize");
    AclWorkspace uniform(uniform_ws);
    ret = aclnnInplaceUniform(uniform.get(), uniform_ws, uniform_exec, stream);
    ACLNN_CHECK(ret, "aclnnInplaceUniform");
    ret = aclrtSynchronizeStream(stream);
    ACL_RT_CHECK(ret, "aclrtSynchronizeStream");
    LOG_INFO("aclnnInplaceUniform completed");

    // 3. 计算 1 - U
    NPUArray one_tensor({}, ACL_FLOAT);
    void* one_data = nullptr;
    ret = aclGetRawTensorAddr(one_tensor.tensorPtr, &one_data);
    ACL_RT_CHECK(ret, "aclGetRawTensorAddr");
    if (!one_data) {
        aclrtDestroyStream(stream);
        throw std::runtime_error(fmt::format(
            "[distributions.cpp]({}) aclGetRawTensorAddr returned null one_data", __func__));
    }
    float one_val = 1.0f;
    ret = aclrtMemcpy(one_data, sizeof(float), &one_val, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    ACL_RT_CHECK(ret, "aclrtMemcpy");

    NPUArray one_minus_u(size, ACL_FLOAT);
    uint64_t sub_ws = 0;
    aclOpExecutor* sub_exec = nullptr;

    aclScalar* alpha_scalar = aclCreateScalar(&one_val, ACL_FLOAT);

    LOG_DEBUG("aclnnSub start: shape={}", detail::FormatShape(size));
    ret = aclnnSubGetWorkspaceSize(
        one_tensor.tensorPtr,   // self
        u_tensor.tensorPtr,     // other
        alpha_scalar,           // alpha
        one_minus_u.tensorPtr,  // out
        &sub_ws,
        &sub_exec
    );
    ACLNN_CHECK(ret, "aclnnSubGetWorkspaceSize");
    AclWorkspace subws(sub_ws);
    ret = aclnnSub(subws.get(), sub_ws, sub_exec, stream);
    ACLNN_CHECK(ret, "aclnnSub");
    ret = aclrtSynchronizeStream(stream);
    ACL_RT_CHECK(ret, "aclrtSynchronizeStream");
    aclDestroyScalar(alpha_scalar);
    LOG_INFO("aclnnSub completed");

    // 4. log(1 - U)
    NPUArray log_tensor(size, ACL_FLOAT);
    uint64_t log_ws = 0;
    aclOpExecutor* log_exec = nullptr;

    LOG_DEBUG("aclnnLog start: shape={}", detail::FormatShape(size));
    ret = aclnnLogGetWorkspaceSize(one_minus_u.tensorPtr, log_tensor.tensorPtr, &log_ws, &log_exec);
    ACLNN_CHECK(ret, "aclnnLogGetWorkspaceSize");
    AclWorkspace logws(log_ws);
    ret = aclnnLog(logws.get(), log_ws, log_exec, stream);
    ACLNN_CHECK(ret, "aclnnLog");
    ret = aclrtSynchronizeStream(stream);
    ACL_RT_CHECK(ret, "aclrtSynchronizeStream");
    LOG_INFO("aclnnLog completed");

    // 5. 除以 log(1 - p)
    NPUArray denom_tensor({}, ACL_FLOAT);
    void* denom_data = nullptr;
    ret = aclGetRawTensorAddr(denom_tensor.tensorPtr, &denom_data);
    ACL_RT_CHECK(ret, "aclGetRawTensorAddr");
    if (!denom_data) {
        aclrtDestroyStream(stream);
        throw std::runtime_error(fmt::format(
            "[distributions.cpp]({}) aclGetRawTensorAddr returned null denom_data", __func__));
    }
    float denom_val = std::log(1.0f - p);
    ret = aclrtMemcpy(denom_data, sizeof(float), &denom_val, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    ACL_RT_CHECK(ret, "aclrtMemcpy");

    NPUArray div_tensor(size, ACL_FLOAT);
    uint64_t div_ws = 0;
    aclOpExecutor* div_exec = nullptr;
    LOG_DEBUG("aclnnDiv start: shape={}, p={}", detail::FormatShape(size), p);
    ret = aclnnDivGetWorkspaceSize(log_tensor.tensorPtr, denom_tensor.tensorPtr, div_tensor.tensorPtr,
        &div_ws, &div_exec);
    ACLNN_CHECK(ret, "aclnnDivGetWorkspaceSize");
    AclWorkspace divws(div_ws);
    ret = aclnnDiv(divws.get(), div_ws, div_exec, stream);
    ACLNN_CHECK(ret, "aclnnDiv");
    ret = aclrtSynchronizeStream(stream);
    ACL_RT_CHECK(ret, "aclrtSynchronizeStream");
    LOG_INFO("aclnnDiv completed");

    // 6. floor
    NPUArray floor_tensor(size, ACL_FLOAT);
    uint64_t floor_ws = 0;
    aclOpExecutor* floor_exec = nullptr;
    LOG_DEBUG("aclnnFloor start: shape={}", detail::FormatShape(size));
    ret = aclnnFloorGetWorkspaceSize(div_tensor.tensorPtr, floor_tensor.tensorPtr, &floor_ws, &floor_exec);
    ACLNN_CHECK(ret, "aclnnFloorGetWorkspaceSize");
    AclWorkspace floor(floor_ws);
    ret = aclnnFloor(floor.get(), floor_ws, floor_exec, stream);
    ACLNN_CHECK(ret, "aclnnFloor");
    ret = aclrtSynchronizeStream(stream);
    ACL_RT_CHECK(ret, "aclrtSynchronizeStream");
    LOG_INFO("aclnnFloor completed");

    // 7. +1
    NPUArray one_tensor2({}, ACL_FLOAT);
    void* one_data2 = nullptr;
    ret = aclGetRawTensorAddr(one_tensor2.tensorPtr, &one_data2);
    ACL_RT_CHECK(ret, "aclGetRawTensorAddr");
    if (!one_data2) {
        aclrtDestroyStream(stream);
        throw std::runtime_error(fmt::format(
            "[distributions.cpp]({}) aclGetRawTensorAddr returned null one_data2", __func__));
    }
    float one_val2 = 1.0f;
    ret = aclrtMemcpy(one_data2, sizeof(float), &one_val2, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    ACL_RT_CHECK(ret, "aclrtMemcpy");

    NPUArray result(size, ACL_FLOAT);
    uint64_t add_ws = 0;
    aclOpExecutor* add_exec = nullptr;
    aclScalar* alpha_one = aclCreateScalar(&one_val2, ACL_FLOAT);

    LOG_DEBUG("aclnnAdd start: shape={}", detail::FormatShape(size));
    ret = aclnnAddGetWorkspaceSize(
        floor_tensor.tensorPtr,  // self
        one_tensor2.tensorPtr,   // other
        alpha_one,               // alpha
        result.tensorPtr,        // out
        &add_ws,
        &add_exec
    );
    ACLNN_CHECK(ret, "aclnnAddGetWorkspaceSize");
    AclWorkspace addws(add_ws);
    ret = aclnnAdd(addws.get(), add_ws, add_exec, stream);
    ACLNN_CHECK(ret, "aclnnAdd");
    ret = aclrtSynchronizeStream(stream);
    ACL_RT_CHECK(ret, "aclrtSynchronizeStream");
    aclDestroyScalar(alpha_one);

    // 8. 清理
    aclrtDestroyStream(stream);

    LOG_INFO("aclnnAdd completed");
    return result;
}


NPUArray Gumbel(double loc, double scale, const std::vector<int64_t>& size) {
    // 1. 参数校验
    if (scale <= 0.0) {
        throw std::invalid_argument(fmt::format(
            "[distributions.cpp]({}) invalid parameter: scale={} <= 0", __func__, scale));
    }

    // 2. 准备随机流与 U 张量
    NPUArray u_tensor(size, ACL_FLOAT);

    uint64_t uniform_ws = 0;
    aclOpExecutor* uniform_exec = nullptr;
    aclrtStream stream = nullptr;
    auto ret = aclrtCreateStream(&stream);
    ACL_RT_CHECK(ret, "aclrtCreateStream");
    if (!stream) {
        throw std::runtime_error(fmt::format(
            "[distributions.cpp]({}) aclrtCreateStream returned null", __func__));
    }

    double low = 0.0;
    double high = 1.0;
    uint64_t seed = 12345;
    uint64_t offset = 0;

    LOG_DEBUG("aclnnInplaceUniform start: shape={}, loc={}, scale={}", detail::FormatShape(size), loc, scale);
    ret = aclnnInplaceUniformGetWorkspaceSize(
        u_tensor.tensorPtr, low, high, seed, offset,
        &uniform_ws, &uniform_exec
    );
    ACLNN_CHECK(ret, "aclnnInplaceUniformGetWorkspaceSize");
    AclWorkspace uniform(uniform_ws);
    ret = aclnnInplaceUniform(uniform.get(), uniform_ws, uniform_exec, stream);
    ACLNN_CHECK(ret, "aclnnInplaceUniform");
    ret = aclrtSynchronizeStream(stream);
    ACL_RT_CHECK(ret, "aclrtSynchronizeStream");
    LOG_INFO("aclnnInplaceUniform completed");

    // 步骤说明：
    // 3. log_u = log(U)
    // 4. neg_log_u = - log_u  (即 multiply by -1)
    // 5. log_neg_log_u = log(neg_log_u)
    // 6. scaled = scale * log_neg_log_u
    // 7. result = loc - scaled

    // 3. log_u = log(U)
    NPUArray log_u(size, ACL_FLOAT);
    uint64_t log_ws = 0;
    aclOpExecutor* log_exec = nullptr;
    LOG_DEBUG("aclnnLog start: shape={}", detail::FormatShape(size));
    ret = aclnnLogGetWorkspaceSize(u_tensor.tensorPtr, log_u.tensorPtr, &log_ws, &log_exec);
    ACLNN_CHECK(ret, "aclnnLogGetWorkspaceSize");
    AclWorkspace logws(log_ws);
    ret = aclnnLog(logws.get(), log_ws, log_exec, stream);
    ACLNN_CHECK(ret, "aclnnLog");
    ret = aclrtSynchronizeStream(stream);
    ACL_RT_CHECK(ret, "aclrtSynchronizeStream");
    LOG_INFO("aclnnLog completed");

    // 4. neg_log_u = -1.0 * log_u  (构造 -1 标量张量并用 Mul)
    float neg_one_val = -1.0f;
    NPUArray neg_one_tensor({}, ACL_FLOAT);
    void* neg_one_data = nullptr;
    ret = aclGetRawTensorAddr(neg_one_tensor.tensorPtr, &neg_one_data);
    ACL_RT_CHECK(ret, "aclGetRawTensorAddr");
    if (!neg_one_data) {
        aclrtDestroyStream(stream);
        throw std::runtime_error(fmt::format(
            "[distributions.cpp]({}) aclGetRawTensorAddr returned null neg_one_data", __func__));
    }
    ret = aclrtMemcpy(neg_one_data, sizeof(float), &neg_one_val, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    ACL_RT_CHECK(ret, "aclrtMemcpy");

    NPUArray neg_log_u(size, ACL_FLOAT);
    uint64_t mul_ws1 = 0;
    aclOpExecutor* mul_exec1 = nullptr;
    LOG_DEBUG("aclnnMul start: shape={}", detail::FormatShape(size));
    ret = aclnnMulGetWorkspaceSize(neg_one_tensor.tensorPtr, log_u.tensorPtr, neg_log_u.tensorPtr,
        &mul_ws1, &mul_exec1);
    ACLNN_CHECK(ret, "aclnnMulGetWorkspaceSize");
    AclWorkspace mulws1(mul_ws1);
    ret = aclnnMul(mulws1.get(), mul_ws1, mul_exec1, stream);
    ACLNN_CHECK(ret, "aclnnMul");
    ret = aclrtSynchronizeStream(stream);
    ACL_RT_CHECK(ret, "aclrtSynchronizeStream");
    LOG_INFO("aclnnMul completed");

    // 5. log_neg_log_u = log(neg_log_u)
    NPUArray log_neg_log_u(size, ACL_FLOAT);
    uint64_t log2_ws = 0;
    aclOpExecutor* log2_exec = nullptr;
    LOG_DEBUG("aclnnLog start: shape={}", detail::FormatShape(size));
    ret = aclnnLogGetWorkspaceSize(neg_log_u.tensorPtr, log_neg_log_u.tensorPtr, &log2_ws, &log2_exec);
    ACLNN_CHECK(ret, "aclnnLogGetWorkspaceSize");
    AclWorkspace log2ws(log2_ws);
    ret = aclnnLog(log2ws.get(), log2_ws, log2_exec, stream);
    ACLNN_CHECK(ret, "aclnnLog");
    ret = aclrtSynchronizeStream(stream);
    ACL_RT_CHECK(ret, "aclrtSynchronizeStream");
    LOG_INFO("aclnnLog completed");

    // 6. scaled = scale * log_neg_log_u  (构造 scale scalar tensor并用 Mul)
    float scale_f = static_cast<float>(scale);
    NPUArray scale_tensor({}, ACL_FLOAT);
    void* scale_data = nullptr;
    ret = aclGetRawTensorAddr(scale_tensor.tensorPtr, &scale_data);
    ACL_RT_CHECK(ret, "aclGetRawTensorAddr");
    if (!scale_data) {
        aclrtDestroyStream(stream);
        throw std::runtime_error(fmt::format(
            "[distributions.cpp]({}) aclGetRawTensorAddr returned null scale_data", __func__));
    }
    ret = aclrtMemcpy(scale_data, sizeof(float), &scale_f, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    ACL_RT_CHECK(ret, "aclrtMemcpy");

    NPUArray scaled(size, ACL_FLOAT);
    uint64_t mul_ws2 = 0;
    aclOpExecutor* mul_exec2 = nullptr;
    LOG_DEBUG("aclnnMul start: shape={}, scale={}", detail::FormatShape(size), scale);
    ret = aclnnMulGetWorkspaceSize(scale_tensor.tensorPtr, log_neg_log_u.tensorPtr, scaled.tensorPtr,
        &mul_ws2, &mul_exec2);
    ACLNN_CHECK(ret, "aclnnMulGetWorkspaceSize");
    AclWorkspace mulws2(mul_ws2);
    ret = aclnnMul(mulws2.get(), mul_ws2, mul_exec2, stream);
    ACLNN_CHECK(ret, "aclnnMul");
    ret = aclrtSynchronizeStream(stream);
    ACL_RT_CHECK(ret, "aclrtSynchronizeStream");
    LOG_INFO("aclnnMul completed");

    // 7. result = loc - scaled
    //    使用 aclnnSub：self = loc_tensor (scalar), other = scaled (tensor), alpha = 1.0
    float loc_f = static_cast<float>(loc);
    NPUArray loc_tensor({}, ACL_FLOAT);
    void* loc_data = nullptr;
    ret = aclGetRawTensorAddr(loc_tensor.tensorPtr, &loc_data);
    ACL_RT_CHECK(ret, "aclGetRawTensorAddr");
    if (!loc_data) {
        aclrtDestroyStream(stream);
        throw std::runtime_error(fmt::format(
            "[distributions.cpp]({}) aclGetRawTensorAddr returned null loc_data", __func__));
    }
    ret = aclrtMemcpy(loc_data, sizeof(float), &loc_f, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    ACL_RT_CHECK(ret, "aclrtMemcpy");

    NPUArray result(size, ACL_FLOAT);
    uint64_t sub_ws = 0;
    aclOpExecutor* sub_exec = nullptr;
    float alpha_val = 1.0f;
    aclScalar* alpha_scalar = aclCreateScalar(&alpha_val, ACL_FLOAT);
    if (alpha_scalar == nullptr) {
        aclrtDestroyStream(stream);
        throw std::runtime_error(fmt::format(
            "[distributions.cpp]({}) aclCreateScalar returned null", __func__));
    }

    LOG_DEBUG("aclnnSub start: shape={}, loc={}, scale={}", detail::FormatShape(size), loc, scale);
    ret = aclnnSubGetWorkspaceSize(
        loc_tensor.tensorPtr,    // self (scalar)
        scaled.tensorPtr,        // other (tensor)
        alpha_scalar,            // alpha
        result.tensorPtr,        // out
        &sub_ws,
        &sub_exec
    );
    ACLNN_CHECK(ret, "aclnnSubGetWorkspaceSize");
    AclWorkspace subws(sub_ws);
    ret = aclnnSub(subws.get(), sub_ws, sub_exec, stream);
    ACLNN_CHECK(ret, "aclnnSub");
    ret = aclrtSynchronizeStream(stream);
    ACL_RT_CHECK(ret, "aclrtSynchronizeStream");

    // 8. 清理资源
    aclDestroyScalar(alpha_scalar);
    aclrtDestroyStream(stream);

    LOG_INFO("aclnnSub completed");
    return result;
}


NPUArray Laplace(double loc, double scale, const std::vector<int64_t>& size) {
    // 1. 参数校验
    if (scale <= 0.0) {
        throw std::invalid_argument(fmt::format(
            "[distributions.cpp]({}) invalid parameter: scale={} <= 0", __func__, scale));
    }

    // 2. 准备 stream 与 U 张量（Uniform in [-0.5, 0.5)）
    NPUArray u_tensor(size, ACL_FLOAT);

    uint64_t uniform_ws = 0;
    aclOpExecutor* uniform_exec = nullptr;
    aclrtStream stream = nullptr;
    auto ret = aclrtCreateStream(&stream);
    ACL_RT_CHECK(ret, "aclrtCreateStream");
    if (!stream) {
        throw std::runtime_error(fmt::format(
            "[distributions.cpp]({}) aclrtCreateStream returned null", __func__));
    }

    double low = -0.5;
    double high = 0.5;
    uint64_t seed = 12345;
    uint64_t offset = 0;

    LOG_DEBUG("aclnnInplaceUniform start: shape={}, loc={}, scale={}", detail::FormatShape(size), loc, scale);
    ret = aclnnInplaceUniformGetWorkspaceSize(
        u_tensor.tensorPtr, low, high, seed, offset,
        &uniform_ws, &uniform_exec
    );
    ACLNN_CHECK(ret, "aclnnInplaceUniformGetWorkspaceSize");
    AclWorkspace uniform(uniform_ws);
    ret = aclnnInplaceUniform(uniform.get(), uniform_ws, uniform_exec, stream);
    ACLNN_CHECK(ret, "aclnnInplaceUniform");
    ret = aclrtSynchronizeStream(stream);
    ACL_RT_CHECK(ret, "aclrtSynchronizeStream");
    LOG_INFO("aclnnInplaceUniform completed");

    // 3. a = abs(U)
    NPUArray abs_u(size, ACL_FLOAT);
    uint64_t abs_ws = 0;
    aclOpExecutor* abs_exec = nullptr;
    LOG_DEBUG("aclnnAbs start: shape={}", detail::FormatShape(size));
    ret = aclnnAbsGetWorkspaceSize(u_tensor.tensorPtr, abs_u.tensorPtr, &abs_ws, &abs_exec);
    ACLNN_CHECK(ret, "aclnnAbsGetWorkspaceSize");
    AclWorkspace absws(abs_ws);
    ret = aclnnAbs(absws.get(), abs_ws, abs_exec, stream);
    ACLNN_CHECK(ret, "aclnnAbs");
    ret = aclrtSynchronizeStream(stream);
    ACL_RT_CHECK(ret, "aclrtSynchronizeStream");
    LOG_INFO("aclnnAbs completed");

    // 4. t = 1 - 2 * abs_u
    // 4.1 构造 scalar 2.0 (as tensor) 并计算 two_mul_abs = 2 * abs_u (Mul)
    NPUArray two_tensor({}, ACL_FLOAT);
    void* two_data = nullptr;
    float two_val_f = 2.0f;
    ret = aclGetRawTensorAddr(two_tensor.tensorPtr, &two_data);
    ACL_RT_CHECK(ret, "aclGetRawTensorAddr");
    if (!two_data) {
        aclrtDestroyStream(stream);
        throw std::runtime_error(fmt::format(
            "[distributions.cpp]({}) aclGetRawTensorAddr returned null two_data", __func__));
    }
    ret = aclrtMemcpy(two_data, sizeof(float), &two_val_f, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    ACL_RT_CHECK(ret, "aclrtMemcpy");

    NPUArray two_mul_abs(size, ACL_FLOAT);
    uint64_t mul_ws1 = 0;
    aclOpExecutor* mul_exec1 = nullptr;
    LOG_DEBUG("aclnnMul start: shape={}", detail::FormatShape(size));
    ret = aclnnMulGetWorkspaceSize(two_tensor.tensorPtr, abs_u.tensorPtr, two_mul_abs.tensorPtr,
        &mul_ws1, &mul_exec1);
    ACLNN_CHECK(ret, "aclnnMulGetWorkspaceSize");
    AclWorkspace mulws1(mul_ws1);
    ret = aclnnMul(mulws1.get(), mul_ws1, mul_exec1, stream);
    ACLNN_CHECK(ret, "aclnnMul");
    ret = aclrtSynchronizeStream(stream);
    ACL_RT_CHECK(ret, "aclrtSynchronizeStream");
    LOG_INFO("aclnnMul completed");

    // 4.2 one_tensor scalar = 1.0
    NPUArray one_tensor({}, ACL_FLOAT);
    void* one_data = nullptr;
    float one_val_f = 1.0f;
    ret = aclGetRawTensorAddr(one_tensor.tensorPtr, &one_data);
    ACL_RT_CHECK(ret, "aclGetRawTensorAddr");
    if (!one_data) {
        aclrtDestroyStream(stream);
        throw std::runtime_error(fmt::format(
            "[distributions.cpp]({}) aclGetRawTensorAddr returned null one_data", __func__));
    }
    ret = aclrtMemcpy(one_data, sizeof(float), &one_val_f, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    ACL_RT_CHECK(ret, "aclrtMemcpy");

    // 4.3 t = 1 - 2*abs_u  :: use aclnnSub (self=one_tensor, other=two_mul_abs, alpha=1)
    NPUArray t_tensor(size, ACL_FLOAT);
    uint64_t sub_ws1 = 0;
    aclOpExecutor* sub_exec1 = nullptr;
    float alpha_val_f = 1.0f;
    aclScalar* alpha_scalar = aclCreateScalar(&alpha_val_f, ACL_FLOAT);
    if (alpha_scalar == nullptr) {
        aclrtDestroyStream(stream);
        throw std::runtime_error(fmt::format(
            "[distributions.cpp]({}) aclCreateScalar returned null", __func__));
    }
    LOG_DEBUG("aclnnSub start: shape={}", detail::FormatShape(size));
    ret = aclnnSubGetWorkspaceSize(one_tensor.tensorPtr, two_mul_abs.tensorPtr, alpha_scalar, t_tensor.tensorPtr,
        &sub_ws1, &sub_exec1);
    ACLNN_CHECK(ret, "aclnnSubGetWorkspaceSize");
    AclWorkspace subws1(sub_ws1);
    ret = aclnnSub(subws1.get(), sub_ws1, sub_exec1, stream);
    ACLNN_CHECK(ret, "aclnnSub");
    ret = aclrtSynchronizeStream(stream);
    ACL_RT_CHECK(ret, "aclrtSynchronizeStream");
    LOG_INFO("aclnnSub completed");

    // 5. log_t = log(t_tensor)
    NPUArray log_t(size, ACL_FLOAT);
    uint64_t log_ws = 0;
    aclOpExecutor* log_exec = nullptr;
    LOG_DEBUG("aclnnLog start: shape={}", detail::FormatShape(size));
    ret = aclnnLogGetWorkspaceSize(t_tensor.tensorPtr, log_t.tensorPtr, &log_ws, &log_exec);
    ACLNN_CHECK(ret, "aclnnLogGetWorkspaceSize");
    AclWorkspace logws(log_ws);
    ret = aclnnLog(logws.get(), log_ws, log_exec, stream);
    ACLNN_CHECK(ret, "aclnnLog");
    ret = aclrtSynchronizeStream(stream);
    ACL_RT_CHECK(ret, "aclrtSynchronizeStream");
    LOG_INFO("aclnnLog completed");

    // 6. sign_u = U / abs_u  (divide elementwise)
    NPUArray sign_u(size, ACL_FLOAT);
    uint64_t div_ws1 = 0;
    aclOpExecutor* div_exec1 = nullptr;
    LOG_DEBUG("aclnnDiv start: shape={}", detail::FormatShape(size));
    ret = aclnnDivGetWorkspaceSize(u_tensor.tensorPtr, abs_u.tensorPtr, sign_u.tensorPtr, &div_ws1, &div_exec1);
    ACLNN_CHECK(ret, "aclnnDivGetWorkspaceSize");
    AclWorkspace divws1(div_ws1);
    ret = aclnnDiv(divws1.get(), div_ws1, div_exec1, stream);
    ACLNN_CHECK(ret, "aclnnDiv");
    ret = aclrtSynchronizeStream(stream);
    ACL_RT_CHECK(ret, "aclrtSynchronizeStream");
    LOG_INFO("aclnnDiv completed");

    // 7. scaled = scale * log_t  (use scale scalar tensor and Mul)
    float scale_f = static_cast<float>(scale);
    NPUArray scale_tensor({}, ACL_FLOAT);
    void* scale_data = nullptr;
    ret = aclGetRawTensorAddr(scale_tensor.tensorPtr, &scale_data);
    ACL_RT_CHECK(ret, "aclGetRawTensorAddr");
    if (!scale_data) {
        aclDestroyScalar(alpha_scalar);
        aclrtDestroyStream(stream);
        throw std::runtime_error(fmt::format(
            "[distributions.cpp]({}) aclGetRawTensorAddr returned null scale_data", __func__));
    }
    ret = aclrtMemcpy(scale_data, sizeof(float), &scale_f, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    ACL_RT_CHECK(ret, "aclrtMemcpy");

    NPUArray scaled(size, ACL_FLOAT);
    uint64_t mul_ws2 = 0;
    aclOpExecutor* mul_exec2 = nullptr;
    LOG_DEBUG("aclnnMul start: shape={}, scale={}", detail::FormatShape(size), scale);
    ret = aclnnMulGetWorkspaceSize(scale_tensor.tensorPtr, log_t.tensorPtr, scaled.tensorPtr,
        &mul_ws2, &mul_exec2);
    ACLNN_CHECK(ret, "aclnnMulGetWorkspaceSize");
    AclWorkspace mulws2(mul_ws2);
    ret = aclnnMul(mulws2.get(), mul_ws2, mul_exec2, stream);
    ACLNN_CHECK(ret, "aclnnMul");
    ret = aclrtSynchronizeStream(stream);
    ACL_RT_CHECK(ret, "aclrtSynchronizeStream");
    LOG_INFO("aclnnMul completed");

    // 8. tmp = sign_u * scaled  (elementwise mul)
    NPUArray tmp(size, ACL_FLOAT);
    uint64_t mul_ws3 = 0;
    aclOpExecutor* mul_exec3 = nullptr;
    LOG_DEBUG("aclnnMul start: shape={}", detail::FormatShape(size));
    ret = aclnnMulGetWorkspaceSize(sign_u.tensorPtr, scaled.tensorPtr, tmp.tensorPtr, &mul_ws3, &mul_exec3);
    ACLNN_CHECK(ret, "aclnnMulGetWorkspaceSize");
    AclWorkspace mulws3(mul_ws3);
    ret = aclnnMul(mulws3.get(), mul_ws3, mul_exec3, stream);
    ACLNN_CHECK(ret, "aclnnMul");
    ret = aclrtSynchronizeStream(stream);
    ACL_RT_CHECK(ret, "aclrtSynchronizeStream");
    LOG_INFO("aclnnMul completed");

    // 9. result = loc - tmp  (use aclnnSub with self=loc_tensor, other=tmp, alpha=1)
    float loc_f = static_cast<float>(loc);
    NPUArray loc_tensor({}, ACL_FLOAT);
    void* loc_data = nullptr;
    ret = aclGetRawTensorAddr(loc_tensor.tensorPtr, &loc_data);
    ACL_RT_CHECK(ret, "aclGetRawTensorAddr");
    if (!loc_data) {
        aclDestroyScalar(alpha_scalar);
        aclrtDestroyStream(stream);
        throw std::runtime_error(fmt::format(
            "[distributions.cpp]({}) aclGetRawTensorAddr returned null loc_data", __func__));
    }
    ret = aclrtMemcpy(loc_data, sizeof(float), &loc_f, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    ACL_RT_CHECK(ret, "aclrtMemcpy");

    NPUArray result(size, ACL_FLOAT);
    uint64_t sub_ws2 = 0;
    aclOpExecutor* sub_exec2 = nullptr;

    LOG_DEBUG("aclnnSub start: shape={}, loc={}, scale={}", detail::FormatShape(size), loc, scale);
    ret = aclnnSubGetWorkspaceSize(loc_tensor.tensorPtr, tmp.tensorPtr, alpha_scalar, result.tensorPtr,
        &sub_ws2, &sub_exec2);
    ACLNN_CHECK(ret, "aclnnSubGetWorkspaceSize");
    AclWorkspace subws2(sub_ws2);
    ret = aclnnSub(subws2.get(), sub_ws2, sub_exec2, stream);
    ACLNN_CHECK(ret, "aclnnSub");
    ret = aclrtSynchronizeStream(stream);
    ACL_RT_CHECK(ret, "aclrtSynchronizeStream");

    // 10. 清理并返回
    aclDestroyScalar(alpha_scalar);
    aclrtDestroyStream(stream);

    LOG_INFO("aclnnSub completed");
    return result;
}


NPUArray Logistic(double loc, double scale, const std::vector<int64_t>& size) {
    // 1. 参数检查
    if (scale <= 0.0) {
        throw std::invalid_argument(fmt::format(
            "[distributions.cpp]({}) invalid parameter: scale={} <= 0", __func__, scale));
    }

    // 2. 创建均匀分布张量 U ~ Uniform(0,1)
    NPUArray u_tensor(size, ACL_FLOAT);

    uint64_t uniform_ws = 0;
    aclOpExecutor* uniform_exec = nullptr;
    aclrtStream stream = nullptr;
    auto ret = aclrtCreateStream(&stream);
    ACL_RT_CHECK(ret, "aclrtCreateStream");
    if (!stream) {
        throw std::runtime_error(fmt::format(
            "[distributions.cpp]({}) aclrtCreateStream returned null", __func__));
    }

    double low = 0.0;
    double high = 1.0;
    uint64_t seed = 12345;
    uint64_t offset = 0;

    LOG_DEBUG("aclnnInplaceUniform start: shape={}, loc={}, scale={}", detail::FormatShape(size), loc, scale);
    ret = aclnnInplaceUniformGetWorkspaceSize(
        u_tensor.tensorPtr, low, high, seed, offset,
        &uniform_ws, &uniform_exec
    );
    ACLNN_CHECK(ret, "aclnnInplaceUniformGetWorkspaceSize");
    AclWorkspace uniform(uniform_ws);
    ret = aclnnInplaceUniform(uniform.get(), uniform_ws, uniform_exec, stream);
    ACLNN_CHECK(ret, "aclnnInplaceUniform");
    ret = aclrtSynchronizeStream(stream);
    ACL_RT_CHECK(ret, "aclrtSynchronizeStream");
    LOG_INFO("aclnnInplaceUniform completed");

    // 3. 计算 1 - U
    NPUArray one_tensor({}, ACL_FLOAT);
    void* one_data = nullptr;
    ret = aclGetRawTensorAddr(one_tensor.tensorPtr, &one_data);
    ACL_RT_CHECK(ret, "aclGetRawTensorAddr");
    if (!one_data) {
        aclrtDestroyStream(stream);
        throw std::runtime_error(fmt::format(
            "[distributions.cpp]({}) aclGetRawTensorAddr returned null one_data", __func__));
    }
    float one_val = 1.0f;
    ret = aclrtMemcpy(one_data, sizeof(float), &one_val, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    ACL_RT_CHECK(ret, "aclrtMemcpy");

    NPUArray one_minus_u(size, ACL_FLOAT);
    uint64_t sub_ws = 0;
    aclOpExecutor* sub_exec = nullptr;

    aclScalar* alpha_scalar = aclCreateScalar(&one_val, ACL_FLOAT); // alpha = 1
    LOG_DEBUG("aclnnSub start: shape={}", detail::FormatShape(size));
    ret = aclnnSubGetWorkspaceSize(
        one_tensor.tensorPtr, u_tensor.tensorPtr, alpha_scalar,
        one_minus_u.tensorPtr, &sub_ws, &sub_exec
    );
    ACLNN_CHECK(ret, "aclnnSubGetWorkspaceSize");
    AclWorkspace subws(sub_ws);
    ret = aclnnSub(subws.get(), sub_ws, sub_exec, stream);
    ACLNN_CHECK(ret, "aclnnSub");
    ret = aclrtSynchronizeStream(stream);
    ACL_RT_CHECK(ret, "aclrtSynchronizeStream");
    aclDestroyScalar(alpha_scalar);
    LOG_INFO("aclnnSub completed");

    // 4. ratio = U / (1 - U)
    NPUArray ratio(size, ACL_FLOAT);
    uint64_t div_ws = 0;
    aclOpExecutor* div_exec = nullptr;
    LOG_DEBUG("aclnnDiv start: shape={}", detail::FormatShape(size));
    ret = aclnnDivGetWorkspaceSize(u_tensor.tensorPtr, one_minus_u.tensorPtr, ratio.tensorPtr, &div_ws, &div_exec);
    ACLNN_CHECK(ret, "aclnnDivGetWorkspaceSize");
    AclWorkspace divws(div_ws);
    ret = aclnnDiv(divws.get(), div_ws, div_exec, stream);
    ACLNN_CHECK(ret, "aclnnDiv");
    ret = aclrtSynchronizeStream(stream);
    ACL_RT_CHECK(ret, "aclrtSynchronizeStream");
    LOG_INFO("aclnnDiv completed");

    // 5. log(ratio)
    NPUArray log_ratio(size, ACL_FLOAT);
    uint64_t log_ws = 0;
    aclOpExecutor* log_exec = nullptr;
    LOG_DEBUG("aclnnLog start: shape={}", detail::FormatShape(size));
    ret = aclnnLogGetWorkspaceSize(ratio.tensorPtr, log_ratio.tensorPtr, &log_ws, &log_exec);
    ACLNN_CHECK(ret, "aclnnLogGetWorkspaceSize");
    AclWorkspace logws(log_ws);
    ret = aclnnLog(logws.get(), log_ws, log_exec, stream);
    ACLNN_CHECK(ret, "aclnnLog");
    ret = aclrtSynchronizeStream(stream);
    ACL_RT_CHECK(ret, "aclrtSynchronizeStream");
    LOG_INFO("aclnnLog completed");

    // 6. scale * log(ratio)
    NPUArray scale_tensor({}, ACL_FLOAT);
    void* scale_data = nullptr;
    ret = aclGetRawTensorAddr(scale_tensor.tensorPtr, &scale_data);
    ACL_RT_CHECK(ret, "aclGetRawTensorAddr");
    if (!scale_data) {
        aclrtDestroyStream(stream);
        throw std::runtime_error(fmt::format(
            "[distributions.cpp]({}) aclGetRawTensorAddr returned null scale_data", __func__));
    }
    float scale_val = static_cast<float>(scale);
    ret = aclrtMemcpy(scale_data, sizeof(float), &scale_val, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    ACL_RT_CHECK(ret, "aclrtMemcpy");

    NPUArray scaled_log(size, ACL_FLOAT);
    uint64_t mul_ws = 0;
    aclOpExecutor* mul_exec = nullptr;
    LOG_DEBUG("aclnnMul start: shape={}, scale={}", detail::FormatShape(size), scale);
    ret = aclnnMulGetWorkspaceSize(scale_tensor.tensorPtr, log_ratio.tensorPtr, scaled_log.tensorPtr, &mul_ws, &mul_exec);
    ACLNN_CHECK(ret, "aclnnMulGetWorkspaceSize");
    AclWorkspace mulws(mul_ws);
    ret = aclnnMul(mulws.get(), mul_ws, mul_exec, stream);
    ACLNN_CHECK(ret, "aclnnMul");
    ret = aclrtSynchronizeStream(stream);
    ACL_RT_CHECK(ret, "aclrtSynchronizeStream");
    LOG_INFO("aclnnMul completed");

    // 7. loc + (scale * log_ratio)
    NPUArray loc_tensor({}, ACL_FLOAT);
    void* loc_data = nullptr;
    ret = aclGetRawTensorAddr(loc_tensor.tensorPtr, &loc_data);
    ACL_RT_CHECK(ret, "aclGetRawTensorAddr");
    if (!loc_data) {
        aclrtDestroyStream(stream);
        throw std::runtime_error(fmt::format(
            "[distributions.cpp]({}) aclGetRawTensorAddr returned null loc_data", __func__));
    }
    float loc_val = static_cast<float>(loc);
    ret = aclrtMemcpy(loc_data, sizeof(float), &loc_val, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    ACL_RT_CHECK(ret, "aclrtMemcpy");

    NPUArray result(size, ACL_FLOAT);
    uint64_t add_ws = 0;
    aclOpExecutor* add_exec = nullptr;
    aclScalar* alpha_add = aclCreateScalar(&one_val, ACL_FLOAT); // alpha = 1
    LOG_DEBUG("aclnnAdd start: shape={}, loc={}, scale={}", detail::FormatShape(size), loc, scale);
    ret = aclnnAddGetWorkspaceSize(loc_tensor.tensorPtr, scaled_log.tensorPtr, alpha_add, result.tensorPtr,
        &add_ws, &add_exec);
    ACLNN_CHECK(ret, "aclnnAddGetWorkspaceSize");
    AclWorkspace addws(add_ws);
    ret = aclnnAdd(addws.get(), add_ws, add_exec, stream);
    ACLNN_CHECK(ret, "aclnnAdd");
    ret = aclrtSynchronizeStream(stream);
    ACL_RT_CHECK(ret, "aclrtSynchronizeStream");
    aclDestroyScalar(alpha_add);

    // 8. 清理资源
    aclrtDestroyStream(stream);

    LOG_INFO("aclnnAdd completed");
    return result;
}


NPUArray Lognormal(float mean, float sigma, const std::vector<int64_t>& size) {
    // 1. 参数校验
    if (sigma <= 0.0f) {
        throw std::invalid_argument(fmt::format(
            "[distributions.cpp]({}) invalid parameter: sigma={} <= 0", __func__, sigma));
    }

    // 2. 创建正态样本张量 Z (will be filled in-place)
    NPUArray z_tensor(size, ACL_FLOAT);

    // 资源变量（初始化）
    uint64_t normal_ws = 0;
    aclOpExecutor* normal_exec = nullptr;

    uint64_t exp_ws = 0;
    aclOpExecutor* exp_exec = nullptr;

    aclrtStream stream = nullptr;
    auto ret = aclrtCreateStream(&stream);
    ACL_RT_CHECK(ret, "aclrtCreateStream");
    if (!stream) {
        throw std::runtime_error(fmt::format(
            "[distributions.cpp]({}) aclrtCreateStream returned null", __func__));
    }

    // 3. 调用 aclnnInplaceNormal 生成 N(mean, sigma) 到 z_tensor
    int64_t seed = 12345;
    int64_t offset = 0;
    float mean_f = static_cast<float>(mean);
    float sigma_f = static_cast<float>(sigma);

    LOG_DEBUG("aclnnInplaceNormal start: shape={}, mean={}, sigma={}", detail::FormatShape(size), mean, sigma);
    ret = aclnnInplaceNormalGetWorkspaceSize(
        z_tensor.tensorPtr,
        mean_f,
        sigma_f,
        seed,
        offset,
        &normal_ws,
        &normal_exec
    );
    ACLNN_CHECK(ret, "aclnnInplaceNormalGetWorkspaceSize");
    AclWorkspace normal(normal_ws);
    ret = aclnnInplaceNormal(normal.get(), normal_ws, normal_exec, stream);
    ACLNN_CHECK(ret, "aclnnInplaceNormal");
    ret = aclrtSynchronizeStream(stream);
    ACL_RT_CHECK(ret, "aclrtSynchronizeStream");
    LOG_INFO("aclnnInplaceNormal completed");

    // 4. 对正态样本做指数变换 result = exp(z_tensor)
    NPUArray result(size, ACL_FLOAT);

    LOG_DEBUG("aclnnExp start: shape={}, mean={}, sigma={}", detail::FormatShape(size), mean, sigma);
    ret = aclnnExpGetWorkspaceSize(z_tensor.tensorPtr, result.tensorPtr, &exp_ws, &exp_exec);
    ACLNN_CHECK(ret, "aclnnExpGetWorkspaceSize");
    AclWorkspace expws(exp_ws);
    ret = aclnnExp(expws.get(), exp_ws, exp_exec, stream);
    ACLNN_CHECK(ret, "aclnnExp");
    ret = aclrtSynchronizeStream(stream);
    ACL_RT_CHECK(ret, "aclrtSynchronizeStream");

    // 5. 清理 stream 并返回
    aclrtDestroyStream(stream);
    LOG_INFO("aclnnExp completed");
    return result;
}

//该API设计有问题，暂时去除
/**NPUArray Multinomial(int64_t n, const NPUArray& pvals, bool replacement, const std::vector<int64_t>& size, int64_t offset) {
    // 1. 参数校验
    if (n <= 0) {
        throw std::runtime_error(fmt::format("Multinomial: n={} <= 0", n));
    }
    if (pvals.tensorPtr == nullptr) {
        throw std::runtime_error("Multinomial: input pvals is null");
    }

    // 2. 创建输出张量
    NPUArray result(size, ACL_INT64);

    // 资源变量初始化
    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    void* ws_addr = nullptr;

    aclrtStream stream = nullptr;
    auto ret = aclrtCreateStream(&stream);
    if (ret != ACL_SUCCESS || !stream) {
        throw std::runtime_error(fmt::format("Multinomial: create stream failed, error={}", ret));
    }

    // 3. 获取 workspace
    int64_t seed = 12345;  // 固定随机种子，必要时可传参

    ret = aclnnMultinomialGetWorkspaceSize(
        pvals.tensorPtr,
        n,
        replacement,
        seed,
        offset,
        result.tensorPtr,
        &ws_size,
        &executor
    );
    if (ret != ACL_SUCCESS) {
        aclrtDestroyStream(stream);
        throw std::runtime_error(fmt::format("Multinomial: get ws failed, error={}", ret));
    }

    if (ws_size > 0) {
        ret = aclrtMalloc(&ws_addr, ws_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            aclrtDestroyStream(stream);
            throw std::runtime_error(fmt::format("Multinomial: malloc ws failed, error={}", ret));
        }
    }

    // 4. 执行 Multinomial
    ret = aclnnMultinomial(ws_addr, ws_size, executor, stream);
    if (ret != ACL_SUCCESS) {
        if (ws_addr) aclrtFree(ws_addr);
        aclrtDestroyStream(stream);
        throw std::runtime_error(fmt::format("Multinomial: compute failed, error={}", ret));
    }

    // 5. 同步
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        if (ws_addr) aclrtFree(ws_addr);
        aclrtDestroyStream(stream);
        throw std::runtime_error(fmt::format("Multinomial: sync failed, error={}", ret));
    }

    // 6. 释放资源
    if (ws_addr) {
        aclrtFree(ws_addr);
        ws_addr = nullptr;
    }
    aclrtDestroyStream(stream);

    return result;
}*/

}