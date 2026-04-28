# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 11:11:58 2026

@author: Mira
"""
from Final_GLA import GLA
from Bit_calculations import X_bits,C_9bit,indencies_bits
from Corrosense_9bits_random import Training_matrix
from Initializations_updated import Kmeans_plus_plus
import numpy as np

X=Training_matrix
K=450
Titel='K-means++'
Initialization=Kmeans_plus_plus(X, K)
Epsilon=0.05
epsilon=0.05
split_epsilon=10
max_iterations=10000

def run_GLA(X,K,Initialization,Epsilon,max_iterations,Titel):
    C,D,_,d,iteration = GLA(X, Initialization, Epsilon, max_iterations)
    Bits_X=X_bits(X)
    Bits_after=C_9bit(C)+indencies_bits(D, K)+128+128
    Procent=((Bits_after-Bits_X)/Bits_X)*100
    """
    print(Titel)
    print('Number of codevectors:',K)
    print('Average distortion:',d)
    print('Iterations:',iteration)
    print('Compression:')
    print('-'*20)
    #print('Bits for training matrix X:',Bits_X)
    print('Total bits after compression:',Bits_after)
    print('Percentage change:',Procent)
    """
    return d, iteration, Bits_X, Bits_after, Procent
#run_GLA(X, K, Initialization, Epsilon, max_iterations, Titel)



def mean_run_GLA(number_of_runs):
    distortions = []
    iterations = []
    bits_after_list = []
    procent_list = []

    for _ in range(number_of_runs):
        d, iteration, Bits_X, Bits_after, Procent = run_GLA(
            X, K, Initialization, Epsilon, max_iterations, "Run")

        distortions.append(d)
        iterations.append(iteration)
        bits_after_list.append(Bits_after)
        procent_list.append(Procent)
    print(Titel)
    print('Number of codevectors:',K)
    print("Average over", number_of_runs, "runs")
    print("Average distortion:", np.mean(distortions))
    print("Average iterations:", np.mean(iterations))
    print("Average bits after:", np.mean(bits_after_list))
    print("Average percentage change:", np.mean(procent_list))

mean_run_GLA(100)

