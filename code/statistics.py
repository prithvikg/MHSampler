import numpy as np
import json
import matplotlib.pyplot as plt

def getmean(series):
    return np.mean(series, axis=0)

def autocorrelation_fn(series,mean):
    n = len(series)
    T = int(np.ceil(n/2))
    cov = np.correlate(series-mean, series-mean, mode = 'same')
    C = cov[-T:]
    C /= n -1 - np.arange(T)
    rho = np.array(C)
    for t in range(T):
        rho[t] = C[t]/C[0]
    return rho,C

def autcorrelation_time(rho, window=5):
    tau = 1
    t = 1
    while(window*tau > t and t <len(rho)):
        tau += rho[t]
        t += 1
    return tau


def calculateStats():
    param_file = open('params.json')
    params = json.load(param_file)
    A = np.loadtxt("A_result")
    L = np.loadtxt("L_result")
    m = params["m"]
    acts = []
    print ("Here1")
    for i in range(0,m):
        print ("Here2")
        rhoA,CA = autocorrelation_fn(A[:,i], getmean(A[:,i]))
        act = autcorrelation_time(rhoA)
        acts.append(act)
        plt.plot(range(0,len(rhoA)), rhoA)
        plt.savefig("Aauto" + str(i))
        plt.clf()
    for i in range(0,m):
        print ("Here3")
        rhoL,CL = autocorrelation_fn(L[:,i], getmean(L[:,i]))
        act = autcorrelation_time(rhoL)
        acts.append(act)
        plt.plot(range(0, len(rhoL)), rhoL)
        plt.savefig("Lauto" + str(i))
        plt.clf()

    thefile = open('acts', 'w')
    for item in acts:
        thefile.write("%s\n" % item)
    print (acts)

if __name__ == "__main__":
    calculateStats()
