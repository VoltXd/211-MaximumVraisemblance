# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 17:00:17 2021

@author: Andre
"""

import sys
import matplotlib.pyplot as plt

def plot3d(x, y, z, theta, phi):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    n = 100
    
    ax.scatter(x, y, z)
    
    ax.set_xlabel('deltaPoids')
    ax.set_ylabel('deltaAge')
    ax.set_zlabel('score')
    ax.view_init(theta, phi)
    
    plt.show()

try:
    f = open('DonneesRugbyPoidsAgeScore.CSV', 'r')
except FileNotFoundError:
    sys.exit(-1)
    

dPoidsIndex = 0
dAgeIndex = 0
scoreIndex = 0    

for line in f:
    line.strip()
    liste = line.split(';')
    for i, element in enumerate(liste):
        if (element == "DeltaPoids"):
            dPoidsIndex = i
        elif (element == "DeltaAge"):
            dAgeIndex = i
        elif (element == "Score\n"):
            scoreIndex = i
    break

dPoids = []
dAge = []
score = []

for line in f:
    line.strip()
    liste = line.split(';')
    if liste[dPoidsIndex] != "":
        dPoids.append(int(liste[dPoidsIndex]))
    if liste[dAgeIndex] != "":
        dAge.append(int(liste[dAgeIndex]))
    if liste[scoreIndex] != "" and liste[scoreIndex] != "\n" :
        score.append(int((liste[scoreIndex])[:-1]))

plot3d(dPoids, dAge, score, 0, 0)
plot3d(dPoids, dAge, score, 0, -90)
plot3d(dPoids, dAge, score, 20, -80)    

f.close()