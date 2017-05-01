import numpy as np
import json

def create_fake_data(use_default=True):
    param_file = open('params.json')
    params = json.load(param_file)

    m = params["m"]
    sigma = params["sigma"]
    N = params["N"]

    A_vals = []
    lambda_vals = []
    if use_default==False:
        #need to sample m A's and m lambdas
        A_vals = np.random.uniform(0,1,m)
        #lambda_vals = np.random.randint(1, N/2, size=m)
        lambda_vals = np.random.uniform(0, 1, size=m)

        np.savetxt(params["A_file"], A_vals )
        np.savetxt(params["lambda_file"], lambda_vals)
    else:
        A_vals = np.loadtxt(params["A_file"])
        lambda_vals = np.loadtxt(params["lambda_file"])

    data = []
    #times = np.arange(0, 1, 1.0/ N)
    times = np.arange(0, 10.0, 10.0 / N)
    #times = np.random.uniform(low=0.0, high=1.0, size=N)

    for t in times:
        l2 = lambda_vals * t * -1
        e = np.exp(l2)
        datapoint = np.dot(A_vals,e)
        data.append(datapoint)

    data = np.array(data)
    data = data + sigma * np.random.normal(0,1,N)

    np.savetxt(params["Y_file"], data)
    np.savetxt(params["times_file"], times)


if __name__ == "__main__":
    create_fake_data()
