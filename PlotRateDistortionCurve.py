# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 10:26:21 2026

@author: Mira
"""

import matplotlib.pyplot as plt

K=[10,50,100,150,200,250,300,350,400,450,512]
Ad=[1113196,719297,596777,535582,490812,461171,430358,410327,393153,376907,359121]
Bitrate=[0.033219280948873623478,0.056438561897747246957,0.066438561897747246956,0.072288186904958808771,0.076438561897747246958,0.079657842846620870436,0.082288186904958808771,0.084512111118323288031,0.086438561897747246956,0.088137811912170370586,0.090000000000000000000] 

K_optimal = 200
Ad_optimal = 490812

plt.plot(Bitrate, Ad, marker='o')
plt.xlabel("Bitrate")
plt.ylabel("Average distortion")


# Tekst annotation

plt.grid()
plt.suptitle("Rate-distortion curve", fontsize=14)
plt.title("K-means++ Initialization", fontsize=10)
plt.show()