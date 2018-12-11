import pandas as pd
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

print(X_train.head())
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
###################### full_pipeline ###############################
full_pipeline = ColumnTransformer([
        ("AnimalType", pipeline_AnimalType, ["AnimalType"]),
        ("SexuponOutcome", pipeline_SexuponOutcome_u, ["SexuponOutcome"]),
    ])

columns = ["AnimalType", "Not-Sterilize", "Sterilize", "Unknown-Sterilize", "Female", "Male", "Unknown-Sex"]
X_train = pd.DataFrame(full_pipeline.fit_transform(X_train),columns= columns)
X_test = pd.DataFrame(full_pipeline.transform(toy_test),columns= columns)

# pipeline_color = Pipeline([
#     ("name", Transformer()),
# ])


# full_pipeline = ColumnTransformer([
#         ("color", pipeline_color, ["Color"]),


#     ])

# column_names = []
# X_train_prepared = pd.DataFrame(full_pipeline.fit_transform(X_train),columns = columns)
# X_test_prepared = pd.DataFrame(full_pipeline.fit_transform(X_test),columns = columns)

# X_train = pd.concat([X_train1,X_train_prepared], axis = 1)
# X_test = pd.concat([X_test1,X_test_prepared], axis = 1)

'''
# textual
pipeline_certain = Pipeline([
    ('certain', TransformationWrapper(transformation=parse_certain)),
    ("encode", LabelEncoderP()),
])

pipeline_positif = Pipeline([
    ('positif', TransformationWrapper(transformation=parse_positif)),
    ("encode", LabelEncoderP()),

])

pipeline_textual_u = Pipeline([
    ("textual", SimpleImputer(strategy='constant', fill_value='maybe yes')),

    ('feats', FeatureUnion([
        ('certain', pipeline_certain),
        ('positif', pipeline_positif)
    ])),

])

# categorical
pipeline_categorical = Pipeline([
    ("fillna", SimpleImputer(strategy='constant', fill_value=1.0)),
    ("encode", OneHotEncoder(categories='auto', sparse=False))
])

# numerical
pipeline_numerical = Pipeline([
    ("fillna", SimpleImputer(strategy='mean')),
    ("scaler", StandardScaler())
])

full_pipeline = ColumnTransformer([
    ("textual", pipeline_textual_u, ["textual"]),
    ("categorical", pipeline_categorical, ["categorical"]),
    ("numerical small", pipeline_numerical, ["numerical_small"]),
    ("numerical high", pipeline_numerical, ["numerical_high"]),
])
'''