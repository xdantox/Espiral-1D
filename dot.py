import numpy as np
def dot(a,b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2] 
def Mdot(M,b):
    return np.array([dot(M[0],b),dot(M[1],b),dot(M[2],b)])