import numpy as np
import scipy.stats
import json
import math
import time

#This function returns a vector, which is f(t) for each t in times
def eval_f(A_vals, lambda_vals, times, m):
    l1 = np.reshape(lambda_vals, (m, 1))
    c = l1 * times
    c = np.exp(-1 * c)
    r = np.dot(A_vals, c)
    return r

def eval_likelihood(A_vals, lambda_vals, times, data_points, m, sigma):
    r = eval_f(A_vals, lambda_vals, times, m)
    val = ((r - data_points)*(r - data_points)) / (2 * sigma )
    val = np.exp(-1 * val)
    t = 1.0
    for item in val:
        t = t * item
    return t

def eval_log_likelihood(A_vals, lambda_vals, times, data_points, m, sigma):
    r = eval_f(A_vals, lambda_vals, times, m)
    val = ((r - data_points)*(r - data_points)) / (2 * sigma * sigma )
    val = np.exp(-1 * val)
    norm = math.sqrt(2 * math.pi * sigma * sigma)
    val = val / norm
    val = np.log(val)
    t = 0.0
    for item in val:
        t = t + item
    return t

def eval_prior_prob_log_sum(A_vals, lambda_vals, sigma):
    return np.log(eval_prior_prob(A_vals, lambda_vals, sigma))

def eval_prior_prob(A_vals, lambda_vals, sigma):
    if sigma < 0 or sigma > 1:
        return 0
    for i in range(0,len(A_vals)):
        if A_vals[i] < 0 or A_vals[i] > 1:
            return 0
        if lambda_vals[i] < 0 or lambda_vals[i] > 1:
            return 0
    return 1.0 * eval_prior_prob_sigma(sigma)

def eval_prior_prob_sigma(sigma):
    param_file = open('params.json')
    params = json.load(param_file)
    mean = params["sigma"]
    return scipy.stats.norm(mean, 0.1).pdf(sigma)

def propose_new_A(A, m):
    A_new = np.random.multivariate_normal(A, 0.1 * np.identity(m))
    return A_new

def propose_new_lambda(lambda_old, m):
    lambda_new = np.random.multivariate_normal(lambda_old, 0.1 * np.identity(m))
    return lambda_new

def propose_new_sigma(sigma_old):
    return np.random.normal(sigma_old, 0.05)

def MCMC_MH(Y, times, m):
    A = 0.5 * np.ones(m)
    lambdas = 0.5 * np.ones(m)
    A = np.array([0.3, 0.7])
    lambdas = np.array([0.3, 0.7])

    param_file = open('params.json')
    params = json.load(param_file)
    sigma = params["sigma"]
    num_iter = params["num_iter"]

    i = 0
    A_history = []
    L_history = []
    sigma_history = []
    likelihood_history = []


    t_end = time.time() + 60 * num_iter
    while time.time() < t_end:
        A_new = propose_new_A(A, m)
        lambda_new = propose_new_lambda(lambdas, m)
        sigma_new = propose_new_sigma(sigma)

        prior_old = eval_prior_prob_log_sum(A, lambdas, sigma)
        likelihood_old = eval_log_likelihood(A, lambdas, times, Y, m, sigma)

        prior_new = eval_prior_prob_log_sum(A_new, lambda_new, sigma_new)
        likelihood_new = eval_log_likelihood(A_new, lambda_new, times, Y, m, sigma_new)

        logp = likelihood_new + prior_new - (likelihood_old + prior_old)
        if logp > 0:
            A = A_new
            lambdas = lambda_new
            sigma = sigma_new
            likelihood_history.append(likelihood_new)
            print ("found new sample ", i)
        else:
            prob = np.exp(logp)
            sample = np.random.uniform(0,1)
            if sample < prob:
                A = A_new
                lambdas = lambda_new
                sigma = sigma_new
                #i += 1
                print ("found new sample with prob", i)
                print (sample, prob)
                likelihood_history.append(likelihood_new)
            else:
                likelihood_history.append(likelihood_old)


        A_history.append(A)
        L_history.append(lambdas)
        sigma_history.append(sigma)
        i += 1

    print("NumIter is ", i)
    np.savetxt("L_result", np.array(L_history))
    np.savetxt("A_result", np.array(A_history))
    np.savetxt("sigma_result", np.array(sigma_history))
    np.savetxt("likelihood_result", np.array(likelihood_history))


if __name__ == "__main__":
    param_file = open('params.json')
    params = json.load(param_file)
    Y = np.loadtxt(params["Y_file"])
    m = params["m"]
    times = np.loadtxt(params["times_file"])
    MCMC_MH(Y, times, m)