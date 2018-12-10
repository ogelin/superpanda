import pandas as pd
import numpy as np
import math
from preprocessing import TransformationWrapper
from preprocessing import LabelEncoderP
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

##################################################################
# Partie des colonnes (Name,DateTime,color) déjà traitées.
#################################################################
PATH = "data/"
X_train = pd.read_csv(PATH + "train.csv")
X_test = pd.read_csv(PATH + "test.csv")

X_train = X_train.drop(columns = ["OutcomeSubtype","AnimalID"])
X_test = X_test.drop(columns = ["ID"])

X_train, y_train = X_train.drop(columns = ["OutcomeType"]),X_train["OutcomeType"]

print((X_train["AgeuponOutcome"].value_counts()/len(X_train)))
print(X_test.head())
print(y_train.head())

X_train1 = pd.read_csv("data/train_preprocessed.csv")
X_test1 = pd.read_csv("data/test_preprocessed.csv")

X_train1.head()

##################################################################
# TP4 - Partie 2
#################################################################

X_train = X_train.drop(columns = ["Color","Name","DateTime"])
X_test = X_test.drop(columns = ["Color","Name","DateTime"])

X_train.head()
# Pour voir les valeurs de la colonne
print((X_train["SexuponOutcome"].value_counts()/len(X_train))[:10])


###################### AnimalType ###############################
pipeline_AnimalType = Pipeline([
    ("encode", LabelEncoderP()),
])

###################### SexuponOutcome ###############################
def parse_sterilize(text):
    sterilize, _ = text.split(" ")
    if (sterilize == "Neutered") or (sterilize == "Spayed"):
        sterilize = "Sterilize"
    return sterilize

pipeline_sterilize = Pipeline([
    ('sterilize', TransformationWrapper(transformation=parse_sterilize)),
    ("encode", OneHotEncoder(categories='auto', sparse=False))
])

def parse_gender(text):
    _, gender = text.split(" ")
    return gender

pipeline_gender = Pipeline([
    ('gender', TransformationWrapper(transformation=parse_gender)),
    ("encode", OneHotEncoder(categories='auto', sparse=False))
])

def parse_unknown(text):
    if text == "Unknown":
        res = "unknown unknown"
    else :
        res = text
    return res

pipeline_SexuponOutcome_u = Pipeline([
    ('SexuponOutcome1', TransformationWrapper(transformation = parse_unknown)),
    ("SexuponOutcome2", SimpleImputer(strategy='constant', fill_value='unknown unknown')),
    ('feats', FeatureUnion([
        ('sterilize', pipeline_sterilize),
        ('gender', pipeline_gender)
    ])),
])

###################### AgeUponOutcome ###############################

def parse_days(text):

    if isinstance(text, float) and math.isnan(text):
        return -1

    quantity, unit = text.split(" ")

    if "day" in unit:
        return quantity
    elif "week" in unit:
        return quantity * 7
    elif "month" in unit:
        return quantity * 30
    elif "year" in unit:
        return quantity * 365

pipeline_age = Pipeline([
    ('age', TransformationWrapper(transformation=parse_days))
])


###################### full_pipeline ###############################
full_pipeline = ColumnTransformer([
        ("AnimalType", pipeline_AnimalType, ["AnimalType"]),
        ("SexuponOutcome", pipeline_SexuponOutcome_u, ["SexuponOutcome"]),
    ("AgeuponOutcome", pipeline_age, ["AgeuponOutcome"])
    ])

columns = ["AnimalType", "Not-Sterilize", "Sterilize", "Unknown-Sterilize", "Female", "Male", "Unknown-Sex", "Age in days"]
X_train = pd.DataFrame(full_pipeline.fit_transform(X_train), columns=columns)
X_test = pd.DataFrame(full_pipeline.transform(X_test), columns=columns)
