# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 16:13:15 2021

@author: Andre
"""

import math
import matplotlib.pyplot as plt

def binomial(n, k, p):
    if (k <= n):
        return (math.factorial(n) / math.factorial(n-k) / math.factorial(k) * p**k * (1-p)**(n-k))
    else:
        return 0

def deriv(f, x, h):
    return (f(N, K, x + h) - f(N, K, x)) / h
    

    
    
#on veut maximiser la proba binomiale pour n = 100 et k = 80
N = 100
K = 80

proba = []
probaBinomiale = []
probaBinomialeD = []

for P in range(0, 101):
    probaBinomiale.append(binomial(N, K, P/100))
    probaBinomialeD.append(deriv(binomial, P/100, 0.0000001))
    proba.append(P/100)

plt.plot(proba, probaBinomiale)
plt.plot(proba, probaBinomialeD)