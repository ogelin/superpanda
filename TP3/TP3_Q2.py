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

#Question 11: AgeuponOutcome (1 point)
# transform all in number of weeks and have only one column (no onehot)

#Question 12: AnimalType (1 point)
# binary  category in one column no onehot

#Question 13: SexuponOutcome (1 point)
# separate into 2 words
# combine the neutered and spayed vs the intact for a binary category

#Question 14: Breed (1 point)
# 1 separate this column into 2: first breed / second breed (if known, if not known put MIX)
# 2 separate each column into categories
# 3 transform each column into a onehot

""" Pipeline
Question 15: Complétez pipeline ci-dessous (4 points)"""

from preprocessing import TransformationWrapper
from preprocessing import LabelEncoderP
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer

# pipeline_color = Pipeline([
#     ("name", Transformer()),
# ])


# full_pipeline = ColumnTransformer([
#         ("color", pipeline_color, ["Color"]),
        

#     ])

#Lancez le pipeline

# column_names = []
# X_train_prepared = pd.DataFrame(full_pipeline.fit_transform(X_train),columns = columns)
# X_test_prepared = pd.DataFrame(full_pipeline.fit_transform(X_test),columns = columns)



#Concaténation des deux parties du dataset:

# X_train = pd.concat([X_train1,X_train_prepared], axis = 1)
# X_test = pd.concat([X_test1,X_test_prepared], axis = 1)
