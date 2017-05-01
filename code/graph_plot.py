import numpy as np
import json
import matplotlib.pyplot as plt

def plotGraph(x, name):
    n, bins, patches = plt.hist(x, normed=1, bins=50)
    plt.ylabel('Count')
    plt.grid(True)
    plt.savefig(name)
    plt.clf()

def plotGraphs():
    Y = np.loadtxt("A_result")
    val = Y[:, 0:1]
    val = val.tolist()
    plotGraph(val)

def plotGraphsA():
    param_file = open('params.json')
    params = json.load(param_file)
    Y = np.loadtxt("A_result")
    m = params["m"]
    for i in range(0,m):
        val = Y[1000:,i:i+1]
        name = "A" + str(i)
        plotGraph(val, name)

def plotGraphsL():
    param_file = open('params.json')
    params = json.load(param_file)
    Y = np.loadtxt("L_result")
    m = params["m"]
    for i in range(0,m):
        val = Y[1000:,i:i+1]
        name = "L" + str(i)
        plotGraph(val, name)

def plotGraphsSigma():
    param_file = open('params.json')
    params = json.load(param_file)
    Y = np.loadtxt("sigma_result")
    m = params["m"]
    name = "Sigma"
    plotGraph(Y, name)

def plotSeq():
    param_file = open('params.json')
    params = json.load(param_file)
    m = params["m"]
    for i in range(0,m):
        Y = np.loadtxt("A_result")
        plt.plot(range(len(Y)-1000), Y[1000:, i])
        plt.savefig("A_seq" + str(i))
        plt.clf()
    for i in range(0, m):
        Y = np.loadtxt("L_result")
        plt.plot(range(len(Y)-1000), Y[1000:, i])
        plt.savefig("L_seq" + str(i))
        plt.clf()

def plotScatterA():
    #lt.figure(figsize=(10, 8))
    param_file = open('params.json')
    params = json.load(param_file)
    Y = np.loadtxt("A_result")
    m = params["m"]
    A = np.loadtxt(params["A_file"])
    val1 = Y[5000:, 0:0 + 1]
    val2 = Y[5000:, 1:1 + 1]

    plt.scatter(val1, val2, marker='x', color='b', alpha=0.005,
                s=124,label='Sampled Points')
    plt.scatter([A[0], A[1]], [A[1], A[0]], marker='o', color='g', alpha=0.7,
                s=124, label='Actual Points')
    # Chart title
    plt.title('Scatter Plot of A values')
    plt.ylabel('A1')
    plt.xlabel('A2')
    plt.legend(loc='upper right')
    plt.savefig("AScatter")
    plt.clf()


def plotScatterLambda():
    #lt.figure(figsize=(10, 8))
    param_file = open('params.json')
    params = json.load(param_file)
    Y = np.loadtxt("L_result")
    m = params["m"]
    A = np.loadtxt(params["lambda_file"])
    val1 = Y[5000:, 0:0 + 1]
    val2 = Y[5000:, 1:1 + 1]

    plt.scatter(val1, val2, marker='x', color='b', alpha=0.005,
                s=124,label='Sampled Points')
    plt.scatter([A[0], A[1]], [A[1], A[0]], marker='o', color='g', alpha=0.7,
                s=124, label='Actual Points')


    # Chart title
    plt.title('Scatter Plot of Lambda values')
    plt.ylabel('L1')
    plt.xlabel('L2')
    plt.legend(loc='upper right')
    plt.savefig("LScatter")
    plt.clf()


def plotGraphs():
    plotGraphsA()
    plt.clf()

    plotGraphsL()
    plt.clf()

    plotGraphsSigma()
    plt.clf()

    plotSeq()
    plt.clf()

    plotScatterA()
    plotScatterLambda()


if __name__ == "__main__":
    plotGraphs()