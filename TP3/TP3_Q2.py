# -*- coding: utf-8 -*-
"""
INF8215 - TP3 - Q2
2. Data preprocessing (8 points)
Created on Tue Nov 27 14:18:54 2018

@author: L.G
"""

import pandas as pd


#Chargement de l'ensemble d'entraînement et de l'ensemble de test
PATH = "data/"
X_train = pd.read_csv(PATH + "train.csv")
X_test = pd.read_csv(PATH + "test.csv")

#Suppression de colonnes inutiles
X_train = X_train.drop(columns = ["OutcomeSubtype","AnimalID"])
X_test = X_test.drop(columns = ["ID"])

X_train, y_train = X_train.drop(columns = ["OutcomeType"]),X_train["OutcomeType"]

#5 premiers exemples de l'ensemble d'entraînement
X_train.head()

#5 premiers exemples de l'attribut à prédire
y_train.head()

"""
Travail demandé
"""

# La partie déjà prétraitée du dataset est chargée dans X_train1 et X_test1
X_train1 = pd.read_csv("data/train_preprocessed.csv")
X_test1 = pd.read_csv("data/test_preprocessed.csv")

X_train1.head()

#Le reste du dataset que vous devez traiter est:
X_train = X_train.drop(columns = ["Color","Name","DateTime"])
X_test = X_test.drop(columns = ["Color","Name","DateTime"])
X_train.head()


