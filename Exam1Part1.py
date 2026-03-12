'''
program should solicit input from the user (with suggested default values) μ, σ, Dmax and Dmin. It
should then produce 11 samples of N=100 rocks randomly selected from the truncated log-normal
distribution and report to the user through the Command Line Interfaces (cli) the sample mean (D̅) and
variance (S2) of each sample as well as the mean and variance of the sampling mean.

Assumptions:
1. While the actual gavel is not spherical, we will assume that the rocks are spherical.
2. Prior to sieving, the gravel follows a log-normal distribution (i.e., loge(D) is N(μ,σ)), where D
is the rock diameter, μ=mean of ln(D) and σ= standard deviation of ln(D).
3. After sieving, the log-normal distribution is now truncated to have a maximum (Dmax) and
minimum size (Dmin) imposed by the aperture size of the screens.
'''

import math
import scipy.stats as stats
import importlib
hw4 = importlib.import_module("Reworked Problem 1")

#region functions
def perform_welchs_t_test(mean_1, var_1, n_1, mean_2, var_2, n_2, alpha=0.05):
    """
    1-sided t-test.
    H0: mu_2 >= mu_1
    H1: mu_2 < mu_1
    
    :param mean_1: Mean of sample 1
    :param var_1: Variance of sample 1
    :param n_1: Size of sample 1
    :param mean_2: Mean of sample 2
    :param var_2: Variance of sample 2
    :param n_2: Size of sample 2
    :param alpha: Significance level
    :return: t_stat, df, p_value, t_crit
    """
    # Calculating the t-statistic
    numerator = mean_2 - mean_1
    denominator = math.sqrt((var_1 / n_1) + (var_2 / n_2))
    t_stat = numerator / denominator
    
    # Calculating the degrees of freedom
    df_num = ((var_1 / n_1) + (var_2 / n_2))**2
    df_den = (((var_1 / n_1)**2) / (n_1 - 1)) + (((var_2 / n_2)**2) / (n_2 - 1))
    df = df_num / df_den
    
    # Calculating the p-value and critical t-value (1-sided)
    p_value = stats.t.cdf(t_stat, df)
    t_crit = stats.t.ppf(alpha, df)
    
    return t_stat, df, p_value, t_crit

def simulate_supplier(name, mean_ln, sig_ln, D_Min, D_Max, N_samples, N_sampleSize):
    """
    Simulates the gravel generation for a specific supplier using hw4 functions.
    
    :return: overall_mean, overall_variance
    """
    print(f"\nGetting data for Supplier {name}")
    print(f"D_max = {D_Max:.3f}, D_min = {D_Min:.3f}")
    
    # Calculating integration boundaries
    F_DMin, F_DMax = hw4.getFDMaxFDMin(mean_ln, sig_ln, D_Min, D_Max)
    
    Samples, Means = hw4.makeSamples((mean_ln, sig_ln, D_Min, D_Max, F_DMax, F_DMin, N_sampleSize, N_samples, False))
    
    # Get the statistics of the sampling mean
    mean_of_means, var_of_means = hw4.sampleStats(Means)
    
    print(f"Supplier {name} mean of means: {mean_of_means:.4f}")
    print(f"Supplier {name} variance of means: {var_of_means:.6f}")
    
    return mean_of_means, var_of_means

def main():

    mean_ln = math.log(2)
    sig_ln = 1.0
    N_samples = 11
    N_sampleSize = 100
    D_Min = 3.0 / 8.0  # Both suppliers use the same small aperture
    alpha = 0.05
    
    print("Starting gravel test...")
    
    # Step 1 & 2: Simulate Supplier 1 (D_max = 1.0)
    D_Max_1 = 1.0
    mean_1, var_1 = simulate_supplier("1", mean_ln, sig_ln, D_Min, D_Max_1, N_samples, N_sampleSize)
    
    # Step 3 & 4: Simulate Supplier 2 (D_max = 7/8)
    D_Max_2 = 7.0 / 8.0
    mean_2, var_2 = simulate_supplier("2", mean_ln, sig_ln, D_Min, D_Max_2, N_samples, N_sampleSize)
    
    # 1-sided statistical t-test
    print("\nRunning t-test")
    print("H0: mu_2 >= mu_1 (Supplier 2 is not smaller)")
    print("H1: mu_2 < mu_1  (Supplier 2 is smaller)")
    print(f"alpha = {alpha}")
    
    t_stat, df, p_value, t_crit = perform_welchs_t_test(mean_1, var_1, N_samples, mean_2, var_2, N_samples, alpha)
    
    print(f"\nStats:")
    print(f"df: {df:.2f}")
    print(f"t-stat: {t_stat:.4f}")
    print(f"t-crit: {t_crit:.4f}")
    print(f"p-value: {p_value:.4e}")
    
    print("\nResult:")
    if t_stat <= t_crit:
        print("Reject H0.")
        print("Supplier 2 has smaller gravel.")
    else:
        print("Fail to reject H0.")
        print("Not enough proof that Supplier 2's gravel is smaller.")

if __name__ == '__main__':
    main()