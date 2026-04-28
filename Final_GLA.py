# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 11:01:39 2026

@author: Mira
"""

import numpy as np


def Distance_matrix(X, C):
    n=X.shape[1]
    Diag_XTX_T = np.sum(X * X, axis=0, keepdims=True)
    Diag_CTC = np.sum(C * C, axis=0, keepdims=True).T
    CTX=np.dot(C.T,X)
    D=np.dot(np.ones((C.shape[1],1)),Diag_XTX_T)-2*(CTX)+np.dot(Diag_CTC,np.ones((1,n)))
    return(D)

def Assignment_matrix(X,D,C):
    n=X.shape[1]
    Encoder_gamma = np.argmin(D,axis=0)
    A=np.zeros((C.shape[1],n),dtype=int)
    A[Encoder_gamma,np.arange(n)]=1
    np.set_printoptions(threshold=np.inf)
    return A

def Reconstruction_matrix(C,A):
    X_hat = np.dot(C,A)
    return X_hat
    
def Average_distortion(X,X_hat):
    n=X.shape[1]
    XminusX_hat=X-X_hat
    d=1/n*(np.sum(XminusX_hat**2))
    return d

def Update_codebook(X,A):
    AAT_inv=np.linalg.pinv(np.dot(A,A.T))
    XAT=np.dot(X,A.T)
    C_new=np.dot(XAT,AAT_inv)
    return C_new


def GLA(X,Initialization,Epsilon,max_iterations):
    C=Initialization
    prev_d = np.inf
    Converged=False
    for iteration in range(max_iterations):
        D=Distance_matrix(X, C)
        A=Assignment_matrix(X,D,C)
        X_hat=Reconstruction_matrix(C, A)
        d=Average_distortion(X, X_hat)
        if abs(prev_d - d) < Epsilon:
            Converged=True
            return C,D,A,d,iteration+1
            break
        C=Update_codebook(X, A)
        prev_d=d
    if not Converged:
        print('Reached max iterations')
        return C,D,A,d,iteration+1

