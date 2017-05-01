import numpy as np
import json
import hw2_funcs as funcs
import graph_plot as graphs
import statistics as stats

if __name__ == "__main__":
    param_file = open('params.json')
    params = json.load(param_file)
    Y = np.loadtxt(params["Y_file"])
    m = params["m"]
    times = np.loadtxt(params["times_file"])
    funcs.MCMC_MH(Y, times, m)
    #print ("plotting graphs")
    #graphs.plotGraphs()
    #print ("Calculating stats")
    #stats.calculateStats()