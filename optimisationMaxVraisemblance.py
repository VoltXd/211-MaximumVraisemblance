# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 17:00:17 2021

@author: Andre
"""

import sys
import matplotlib.pyplot as plt
import numpy as np
import colorama


###################################################################################
#                                                                                 #
#                             !!!!!!! ATTENTION !!!!!!!                           #
#          Il faut bien penser à ne pas se tromper dans les indices, le           #
#                             bon ordre est le suivant :                          #
#                                                                                 #
#   p: vecteur 4 lignes                                                           #
#       p[0]: Coef de dilatation d'arctan                                         #
#       p[1]: Coef de décalage d'arctan                                           #
#       p[2]: Coef de dilatation de la gaussienne                                 #
#       p[3]: Coef de décalage de la gaussienne                                   #
#                                                                                 #
#   u: matrice n lignes, 2 colonnes                                               #
#       u[0]: Différence d'age                                                    #
#       u[1]: Différence de poids                                                 #
#                                                                                 #
###################################################################################

#Il est nécessaire d'avoir le fichier "DonneesRugbyPoidsAgeScore.CSV" dans le même dossier que ce script

def progressBar(progress, total, color=colorama.Fore.YELLOW):
    percent = 100 * progress / float(total)
    bar = '█' * int(percent) + '-' * (100 - int(percent))
    if int(percent) == 100:
        print(colorama.Fore.GREEN + f'\r|{bar}| {percent:.2f}%')
        print(colorama.Fore.RESET)
    else:
        print(color + f'\r|{bar}| {percent:.2f}%', end='\r')
    
#Affichage des échantillons sous un certain angle
def plot3d(x, y, z, theta, phi):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    ax.scatter(x, y, z)
    
    ax.view_init(theta, phi)
    
    
    ax.set_xlabel('deltaPoids')
    ax.set_ylabel('deltaAge')
    ax.set_zlabel('score')
    plt.show()
    
#Affiche une fonction (la surface de probabilité) et les échantillons sur un même graphique
def plotProba(f, x, dA, dP, s):
    # Make data.
    X = np.arange(-40, 80, 0.1)
    Y = np.arange(-2, 3, 0.01)
    X, Y = np.meshgrid(X, Y)
    Z = f([Y, X], x)
    
    plt.contourf(X, Y, Z, 30, cmap="magma")
    plt.colorbar()
    plt.xlabel('deltaPoids')
    plt.ylabel('deltaAge')
    plt.show()
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(dP, dA, s)
    ax.contour3D(X, Y, Z, 50)
    ax.view_init(20, -45)
    ax.set_xlabel('deltaPoids')
    ax.set_ylabel('deltaAge')
    ax.set_zlabel('score')
    
    plt.show()
    
#Affiche une fonction (utilisé pour le critère) sois en affichant la surface, sois avec des niveaux de couleurs
def plotCritere(f, x, u, y):
    X = np.arange(0, 5, 0.001)
    Y = np.arange(0, 0.05, 0.001)
    X, Y = np.meshgrid(X, Y)
    
    Z = f(y, u, [X, x[1], Y, x[3]])
    
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z, 50)
    ax.view_init(20, -45)
    ax.set_xlabel('DilatAtan')
    ax.set_ylabel('DilatExp')
    ax.set_zlabel('critere')
    
    plt.show()
    
    plt.contourf(X, Y, Z, 30, cmap="magma")
    plt.plot(x[0], x[2], 'rx')
    plt.xlabel('DilatAtan')
    plt.ylabel('DilatExp')
    plt.colorbar()
    plt.show()

    

#Définition de la probabilité(p: parametres, u: entrees)
def proba(u, p):
    return ((np.arctan(p[0]*(u[0] - p[1]))/np.pi + 0.5) * np.exp(-(p[2]*(u[1] - p[3]))**2))
    
#Définition de la fonction de vraisemblance[y: sorties (1 ou 0), u: toute les entrees, p: parametres]
#!!!! On renvoie la vraisemblance * (-1) afin de faire une descente de gradient
def vraisemblance(y, u, p):
    pi = 1
    n = len(y)
    for i in range(n):
        pi *= proba(u[i,:], p)**y[i] * (1 - proba(u[i,:], p))**(1 - y[i])
    return -pi

#Ancienne définition de la vraisemblance (abondonnée à cause de log(0.0000...)=NaN)
"""def vraisemblance(y, u, p):
    logPi = 0
    n = len(y)
    for i in range(n):
        logPi += y[i] * np.log(proba(u[i,:], p)) + (1 - y[i]) * np.log(1 - proba(u[i,:], p))
    return -logPi
"""

#Définition du gradient d'une fonction (la fonction gradient est adapté au format de la fonction de vraisemblance (f(y, u, x)))
def gradF(f, x, h, y, u):
    grad = []
    fx = f(y, u, x)
    for i in range(len(x)):
        hGrad = np.zeros(len(x))
        hGrad[i] = h
        xh = x + hGrad
        grad.append((f(y, u, xh) - fx)/h)
        
    return grad

#Optimisation du pas selon le critère d'Armijo
def pasArmijo(a0, f, y, u, p, direction, normGrad, beta, tau):
    alpha = [a0]
    while (f(y, u, p + alpha[-1] * direction) >= (f(y, u, p) - alpha[-1] * beta * normGrad**2)):
        alpha.append(tau * alpha[-1])
    return alpha[-1]

#Optimisation du pas selon le critère d'Armijo-Goldstein
def pasGoldstein(a0, f, y, u, p, direction, normGrad, beta1, beta2, tau):
    alphaMin = pasArmijo(a0, f, y, u, p, direction, normGrad, beta1, tau)
    if (f(y, u, p + alphaMin * direction) <= (f(y, u, p) - alphaMin * beta1 * normGrad**2)) and (f(y, u, p + alphaMin * direction) >= (f(y, u, p) - alphaMin * beta2 * normGrad**2)):
        #Le pas d'armijo respecte les deux critères, on le retourne
        return alphaMin
    else:
        alphaMax = alphaMin
        #Tant qu'alphaMax ne respecte pas Goldstein, on le double
        while not((f(y, u, p + alphaMax * direction) >= (f(y, u, p) - alphaMax * beta2 * normGrad**2))):
            alphaMax *= 2
        #On fait une dichotomie sur alpha jusqu'à ce qu'il respecte les deux critères
        alpha = (alphaMax + alphaMin) / 2
        while not((f(y, u, p + alpha * direction) <= (f(y, u, p) - alpha * beta1 * normGrad**2)) and (f(y, u, p + alpha * direction) >= (f(y, u, p) - alpha * beta2 * normGrad**2))):
            if not((f(y, u, p + alpha * direction) <= (f(y, u, p) - alpha * beta1 * normGrad**2))):
                alphaMax = alpha
            else:
                alphaMin = alpha
            alpha = (alphaMax + alphaMin) / 2
        return alpha
    

#1. Récupération des données des matchs
try:
    print('Opening "DonneesRugbyPoidsAgeScore.CSV" : ')
    f = open('DonneesRugbyPoidsAgeScore.CSV', 'r')
    print("\tDone!\n")
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

print("Extracting datas : ")
for line in f:
    line.strip()
    liste = line.split(';')
    if liste[dPoidsIndex] != "":
        dPoids.append(int(liste[dPoidsIndex]))
    if liste[dAgeIndex] != "":
        dAge.append(int(liste[dAgeIndex]))
    if liste[scoreIndex] != "" and liste[scoreIndex] != "\n" :
        score.append(int((liste[scoreIndex])[:-1]))
f.close()
print("\tDone! \n\tFile closed!\n")

#Test de modélisation pour la moitié des échantillons
"""dPoids = dPoids[0:8]
dAge = dAge[0:8]
score = score[0:8]"""

#Affichage des échantillons des matchs   
plot3DNumber = 3

print("Plotting datas : ")
progressBar(0, plot3DNumber)
plot3d(dPoids, dAge, score, 0, 0)
progressBar(1, plot3DNumber)
plot3d(dPoids, dAge, score, 0, -90)
progressBar(2, plot3DNumber)
plot3d(dPoids, dAge, score, 20, -80)
progressBar(3, plot3DNumber)
print("\tDone!\n")    

#2. Optimisation du critère (Max de vraisemblance)
#Initialisation des parametres, de la matrice des entrées et des parametres du gradient
p0 = [1, -1/2, 0.02, 30]
entrees = np.array([dAge, dPoids]).T 

nMax = 10**4
normGradMin = 10**-6
varMin = 10**-8

n = 0
normGrad = [1]
var = [1]
criterePrec = [1]

pas = []
pasDérivé = 0.000001

#On affiche le critère pour les paramètres initiaux (décalages fixes)
print("Ploting cost function :")
plotCritere(vraisemblance, p0, entrees, score)
print('\tDone!\n')

#Test de la fonction de vraisemblance 
print("Vraisemblance initiale : {}\n".format(-vraisemblance(score, entrees, p0)))

parametres = [p0]
tableauVraisemblances = []

#Affichage de la probabilité avant optimisation
print("Ploting initial probability surface :")
plotProba(proba, parametres[-1], dAge, dPoids, score)
print("\tDone!\n")

print("Performing gradient descent:")
progressBar(n, nMax)
#Itérations de la méthode du gradient (Criteres d'arréts: nMax d'itérations, norme du gradient, variation relative du critère)
while nMax > n and normGradMin < normGrad[-1] and varMin * criterePrec[-1] < var[-1] :
    tableauVraisemblances.append(-vraisemblance(score, entrees, parametres[-1]))
    
    #Calcul du gradient (direction de descente)
    directionMin = -np.array(gradF(vraisemblance, parametres[-1], pasDérivé, score, entrees))
    normGrad.append(np.linalg.norm(directionMin))
    
    #Optimisation du pas par algorithme de Backtracking-Armijo
    #pas.append(pasArmijo(100, vraisemblance, score, entrees, parametres[-1], directionMin, normGrad[-1], 0.001, 0.5))
    
    #Optimisation du pas par condition de Goldstein-Armijo
    pas.append(pasGoldstein(100, vraisemblance, score, entrees, parametres[-1], directionMin, normGrad[-1], 0.001, 0.6, 0.25))
    
    prochainPoint = parametres[-1] + pas[-1]*directionMin
    parametres.append(prochainPoint)
    var.append(np.linalg.norm(vraisemblance(score, entrees, parametres[-1]) - vraisemblance(score, entrees, parametres[-2])))
    criterePrec.append(np.linalg.norm(vraisemblance(score, entrees, parametres[-2])))
    n += 1
    progressBar(n, nMax)
progressBar(nMax, nMax)
print("\tDone!\nDescent finished!\n")
tableauVraisemblances.append(-vraisemblance(score, entrees, parametres[-1]))

#On force p[2] à être dans R+ (plus pratique pour l'affichage)
if (parametres[-1][2] < 0):
    parametres[-1][2] = -parametres[-1][2]

#Affichage de la probabilité après optimisation
plotProba(proba, parametres[-1], dAge, dPoids, score)

#Indique quel a était le critère d'arrêt satisfait
if nMax <= n:
    print("Le critère d'arrêt est le dépassement du nombre d'itérations max")
elif normGradMin > normGrad[-1] :
    print("Le critère d'arrêt est la norme min du gradient")
elif varMin * criterePrec[-1] > var[-1]:
    print("Le critère d'arrêt est la variation du point")

#Affichage de la vraisemblance après optimisation
print('La valeur de la vraisemblance apres optimisation est : {}'.format(tableauVraisemblances[-1]))

#Graphiques divers
print("Ploting maximum likelyhood through iterations :")
plt.plot(tableauVraisemblances)
plt.xlabel('Itération')
plt.ylabel('Vraisemblance')
plt.show()
plt.plot(pas, 'x')
plt.xlabel('Itération')
plt.ylabel('pas')
plt.yscale('log')
plt.show()
print("\tDone!\n")

print("Ploting criterion relative variation :")
varRelat = [i / j for i, j in zip(var[1:], criterePrec[1:])]
plt.plot(varRelat)
plt.xlabel('Itération')
plt.ylabel('Variation relative du critère')
plt.yscale('log')
plt.show()
plt.plot(normGrad[1:])
plt.xlabel('Itération')
plt.ylabel('Norme du gradient')
plt.yscale('log')
plt.show()
print('\tDone!\n')

print("Ploting final cost function :")
plotCritere(vraisemblance, parametres[-1], entrees, score)
print('\tDone!\n')

print("Affichage des probabilités associés à chaque échantillon : ")
for i in range(len(score)):
    print("{}, {:2f}".format(score[i], proba(entrees[i], parametres[-1])))
        
