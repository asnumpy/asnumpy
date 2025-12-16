# *****************************************************************************
# Copyright (c) 2025 AISS Group at Harbin Institute of Technology. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# *****************************************************************************

import numpy as np
from scipy import stats
import asnumpy as ap


def main():
    np.random.seed(123)
    #测试pareto分布
    ap_pareto_data = ap.random.pareto(3.0, (10000, ))
    pareto_data = ap_pareto_data.to_numpy()
    ks_stat, p_value = stats.kstest(pareto_data, 'lomax', args=(3.0, 0, 1))
    print(f"pareto分布 K-S检验 P值: {p_value:.4f} → {'通过' if p_value > 0.05 else '失败'}")

    #测试rayleigh分布
    ap_rayleigh_data = ap.random.rayleigh(2.0, (10000, ))
    rayleigh_data = ap_rayleigh_data.to_numpy()
    ks_stat, p_value = stats.kstest(rayleigh_data, 'rayleigh', args=(0, 2.0))
    print(f"rayleigh分布 K-S检验 P值: {p_value:.4f} → {'通过' if p_value > 0.05 else '失败'}")

    #测试正态分布
    ap_normal_data = ap.random.normal(5.0, 2.0, (10000, ))
    normal_data = ap_normal_data.to_numpy()
    ks_stat, p_value = stats.kstest(normal_data, 'norm', args=(5.0, 2.0))
    print(f"正态分布 K-S检验 P值: {p_value:.4f} → {'通过' if p_value > 0.05 else '失败'}")
    
    # 测试均匀分布
    ap_uniform_data = ap.random.uniform(0.0, 1.0, (10000, ))
    uniform_data = ap_uniform_data.to_numpy()
    ks_stat, p_value = stats.kstest(uniform_data, 'uniform', args=(0.0, 1.0))
    print(f"均匀分布 K-S检验 P值: {p_value:.4f} → {'通过' if p_value > 0.05 else '失败'}")

    #测试标准正态分布
    ap_standard_normal_data = ap.random.standard_normal((10000, ))
    standard_normal_data = ap_standard_normal_data.to_numpy()
    ks_stat, p_value = stats.kstest(standard_normal_data, 'norm', args=(0.0, 1.0))
    print(f"标准正态分布 K-S检验 P值: {p_value:.4f} → {'通过' if p_value > 0.05 else '失败'}")

    #测试标准柯西分布
    ap_standard_cauchy_data = ap.random.standard_cauchy((10000, ))
    standard_cauchy_data = ap_standard_cauchy_data.to_numpy()
    ks_stat, p_value = stats.kstest(standard_cauchy_data, 'cauchy', args=(0.0, 1.0))
    print(f"标准柯西分布 K-S检验 P值: {p_value:.4f} → {'通过' if p_value > 0.05 else '失败'}")

    # 测试weibull分布
    ap_weibull_data = ap.random.weibull(1.5, (10000, ))
    weibull_data = ap_weibull_data.to_numpy()
    ks_stat, p_value = stats.kstest(weibull_data, 'weibull_min', args=(1.5, 0, 1))
    print(f"weibull分布 K-S检验 P值: {p_value:.4f} → {'通过' if p_value > 0.05 else '失败'}")

    # 测试二项分布
    ap_binomial_data = ap.random.binomial(10, 0.5, (10000, ))
    binomial_data = ap_binomial_data.to_numpy()
    observed = np.bincount(binomial_data, minlength=11)
    expected = stats.binom.pmf(range(11), 10, 0.5) * len(binomial_data)
    chi2_stat, p_value = stats.chisquare(observed, expected)
    print(f"二项分布 卡方检验 P值: {p_value:.4f} → {'通过' if p_value > 0.05 else '失败'}")

    # 测试指数分布
    ap_exp_data = ap.random.exponential(3.0, (10000, ))
    exp_data = ap_exp_data.to_numpy()
    ks_stat, p_value = stats.kstest(exp_data, 'expon', args=(0, 3.0))
    print(f"指数分布 K-S检验 P值: {p_value:.4f} → {'通过' if p_value > 0.05 else '失败'}")

    # 测试几何分布
    ap_geometric_data = ap.random.geometric(0.3, (10000, ))
    geometric_data = ap_geometric_data.to_numpy()
    max_k = geometric_data.max()
    # 设定一个阈值，将大于 threshold 的值合并为一个组
    threshold = 10
    bins = list(range(1, threshold + 1))
    bins.append(np.inf)  # 最后一个区间为 [threshold, ∞)，即"threshold及以上"
    # 计算前 threshold-1 个区间的理论概率 (每个区间就是一个整数k)
    expected_probs = stats.geom.pmf(bins[:-1], 0.3)  # 注意：scipy.stats.geom的定义域从1开始，PMF为(1-p)^(k-1)*p
    # 最后一个区间的概率是 1 减去前面所有概率之和
    expected_probs = np.append(expected_probs, 1 - stats.geom.cdf(threshold - 1, 0.3))
    # 将概率转换为期望频数
    expected_freq = expected_probs * 10000
    observed_freq, _ = np.histogram(geometric_data, bins=bins)
    # np.histogram 的 bins 如 [1,2,3,...,inf]，所以区间是 [1,2), [2,3), ... [threshold, inf]
    observed = observed[expected >= 5]
    expected = expected[expected >= 5]
    chi2_stat, p_value = stats.chisquare(observed, expected)
    print(f"几何分布 卡方检验 P值: {p_value:.4f} → {'通过' if p_value > 0.05 else '失败'}")

    # 测试gumbel分布
    ap_gumbel_data = ap.random.gumbel(0.0, 1.0, (10000, ))
    gumbel_data = ap_gumbel_data.to_numpy()
    ks_stat, p_value = stats.kstest(gumbel_data, 'gumbel_r', args=(0.0, 1.0))
    print(f"gumbel分布 K-S检验 P值: {p_value:.4f} → {'通过' if p_value > 0.05 else '失败'}")

    # 测试laplace分布
    ap_laplace_data = ap.random.laplace(0.0, 1.0, (10000, ))
    laplace_data = ap_laplace_data.to_numpy()
    ks_stat, p_value = stats.kstest(laplace_data, 'laplace', args=(0.0, 1.0))
    print(f"laplace分布 K-S检验 P值: {p_value:.4f} → {'通过' if p_value > 0.05 else '失败'}")

    # 测试逻辑分布
    ap_logistic_data = ap.random.logistic(0.0, 1.0, (10000, ))
    logistic_data = ap_logistic_data.to_numpy()
    ks_stat, p_value = stats.kstest(logistic_data, 'logistic', args=(0.0, 1.0))
    print(f"逻辑分布 K-S检验 P值: {p_value:.4f} → {'通过' if p_value > 0.05 else '失败'}")

    # 测试对数正态分布
    ap_lognormal_data = ap.random.lognormal(0.0, 1.0, (10000, ))
    lognormal_data = ap_lognormal_data.to_numpy()
    ks_stat, p_value = stats.kstest(lognormal_data, 'lognorm', args=(1.0, 0, np.exp(0.0)))
    print(f"对数正态分布 K-S检验 P值: {p_value:.4f} → {'通过' if p_value > 0.05 else '失败'}")


if __name__ == "__main__":
    main()