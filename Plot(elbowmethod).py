# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 08:39:17 2026

@author: Mira
"""

import matplotlib.pyplot as plt

K=[10,50,100,150,200,250,300,350,400,450,512]
Ad=[1113196,719297,596777,535582,490812,461171,430358,410327,393153,376907,359121]

K_optimal = 200
Ad_optimal = 490812

plt.plot(K, Ad, marker='o')
plt.xlabel("K")
plt.ylabel("Average distortion")

plt.scatter(K_optimal, Ad_optimal,color='red', s=120)
plt.axvline(x=K_optimal,color='red', linestyle='--')
plt.axhline(y=Ad_optimal,color='red', linestyle='--')

# Tekst annotation
plt.text(K_optimal+10, Ad_optimal+20000, 'Knee point (K ≈ 200)')
plt.grid()
plt.suptitle("Average Distortion vs. Number of Code Vectors (K)", fontsize=14)
plt.title("K-means++ Initialization", fontsize=10)
plt.show()