import pandas as pd
import numpy as np
import math
from preprocessing import TransformationWrapper
from preprocessing import LabelEncoderP
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
import os

exported_train_file = "X_train.csv"
exported_test_file = "X_test.csv"

class TransformerToArray(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, X):
        return np.asarray(X.todense())

# Returns a column from the dataframe named df as a numpy array of type string.
class TextExtractor(BaseEstimator, TransformerMixin):
    """Adapted from code by @zacstewart
       https://github.com/zacstewart/kaggle_seeclickfix/blob/master/estimator.py
       Also see Zac Stewart's excellent blogpost on pipelines:
       http://zacstewart.com/2014/08/05/pipelines-of-featureunions-of-pipelines.html
       """
    def __init__(self, column_name):
        self.column_name = column_name
    def transform(self, df):
        # Select the relevant column and return it as a numpy array.
        # Set the array type to be string.
        return np.asarray(df[self.column_name]).astype(str)
    def fit(self, *_):
        return self



##################################################################
# Partie des colonnes (Name,DateTime,color) déjà traitées.
#################################################################
PATH = "data/"
X_train = pd.read_csv(PATH + "train.csv")
X_test = pd.read_csv(PATH + "test.csv")

X_train = X_train.drop(columns = ["OutcomeSubtype","AnimalID"])
X_test = X_test.drop(columns = ["ID"])

X_train, y_train = X_train.drop(columns = ["OutcomeType"]),X_train["OutcomeType"]

#print(X_train.Breed.str.contains('0', regex=False).unique())


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

    quantity, unit = text.split(" ")

    if "day" in unit:
        return int(quantity)
    elif "week" in unit:
        return int(quantity) * 7
    elif "month" in unit:
        return int(quantity) * 30
    elif "year" in unit:
        return int(quantity) * 365


pipeline_age = Pipeline([
    ('most_frequent', SimpleImputer(strategy="most_frequent")),
    ('age', TransformationWrapper(transformation=parse_days)),
    ('scaler', StandardScaler())
])

###################### breed ###############################

vect = CountVectorizer()

def parse_breed(text):
    if "/" in text:
        if "Mix" not in text:
            text += " Mix "
    return text

#CountVectorizer
pipeline_breed = Pipeline([

    ('name_extractor', TextExtractor('Breed')),
    ('breed', vect),
    ('array', TransformerToArray()),
])

def get_breed_names(data_frame_serie):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data_frame_serie)
    return vectorizer.get_feature_names()


###################### full_pipeline ###############################
full_pipeline = ColumnTransformer([
        ("AnimalType", pipeline_AnimalType, ["AnimalType"]),
        ("SexuponOutcome", pipeline_SexuponOutcome_u, ["SexuponOutcome"]),
        ("AgeuponOutcome", pipeline_age, ["AgeuponOutcome"]),
        ("Breed", pipeline_breed, ["Breed"])
    ])

###################### utilities ###############################
def is_already_exported(filename):
    file_path = "./" + filename
    return os.path.exists(file_path)

def get_formated_data():
    if is_already_exported(filename=exported_train_file) and is_already_exported(filename=exported_test_file) :
        train = pd.read_csv(exported_train_file, delimiter=',', header=None).values
        test = pd.read_csv(exported_test_file, delimiter=',', header=None).values
        return train, test
    else:
        return data_processing()

###################### data processing ###############################

def data_processing():
    col = ["AnimalType", "Not-Sterilize", "Sterilize", "Unknown-Sterilize", "Female", "Male", "Unknown-Sex", "Age in days"] + get_breed_names(X_train.Breed)

    X_train_prepared = pd.DataFrame(full_pipeline.fit_transform(X_train), columns=col)
    X_test_prepared = pd.DataFrame(full_pipeline.transform(X_test), columns=col)

    train = pd.concat([X_train1,X_train_prepared], axis=1)
    test = pd.concat([X_test1,X_test_prepared], axis=1)

    # Export data
    n = train.shape[1] - 13
    np.savetxt(exported_train_file, train, fmt=','.join(['%1.8f'] + ['%i'] * 11 + ['%1.8f'] + ['%i'] * n))
    np.savetxt(exported_test_file, test, fmt=','.join(['%1.8f'] + ['%i'] * 11 + ['%1.8f'] + ['%i'] * n))

    return train, test

################### MAIN ######################################

# --- PARTIE 2 ---
X_train, X_test = get_formated_data()

# --- PARTIE 3 ---
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support

target_label = LabelEncoder()
y_train_label = target_label.fit_transform(y_train)
print(target_label.classes_)


# BONUS 2 : En observant la distribution des classes de l'attribut cible (à l'aide des fonctions de visualisation de
# pandas), justifiez l'utilisation de l'objet StratifiedKFold de sklearn pour la division de l'ensemble d'entraînement
# lors de cross-validation en comparaison à une méthode pûrement aléatoire
from sklearn.model_selection import StratifiedKFold
'''
RÉPONSE: 
Voici l'observation des classes de l'attribut cible : 
   print((y_train.value_counts()/len(y_train))[:5])
   Adoption           0.402896
   Transfer           0.352501
   Return_to_owner    0.179056
   Euthanasia         0.058177
   Died               0.007370
On voit qu'il y a 3 classes qui sont vraiment plus présentes que les 2 autres dans nos données. Avec une approche
aléatoire, on ne pourrait pas s'assurer hors de tous doute d'avoir la présence de ces classes plus rares dans notre
set de test. Pour sa part, StratifiedKFold offre de garder la présence de chaque classe dans chaque fold et de facon
proportionnel à l'ensemble de données complet. Comme cela, on s'assure de valider aussi avec les classes les moins
présentes.
'''

def compare(models, X, y, nb_runs):

    #Init tables that will contain metrics values
    losses = np.zeros(shape=(len(models),nb_runs))
    precision = np.zeros(shape=(len(models),nb_runs))
    recall = np.zeros(shape=(len(models),nb_runs))
    fscore = np.zeros(shape=(len(models),nb_runs))

    skf = StratifiedKFold(n_splits=nb_runs)

    #Start the cross-validation
    run_i = 0
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # For each model
        model_i = 0
        for model in models:
            train_p = model.fit_predict(X_train, y_train)
            test_p = model.predict(X_test)
            print("train : " + str(precision_recall_fscore_support(y_train, train_p, average="macro")))
            print("test : " + str(precision_recall_fscore_support(y_test, test_p, average="macro")))

            # On enregistre les métriques
            precision[model_i][run_i], recall[model_i][run_i], fscore[model_i][run_i], _ = precision_recall_fscore_support(y_test, test_p, average="macro")
            losses[model_i][run_i] = model.losses_[-1]
            print("loss = " + str(model.losses_[-1]))
            model_i = model_i + 1

        run_i = run_i + 1

    # Moyennes par model
    losses_mean = np.mean(losses, axis=1)
    precision_mean = np.mean(precision, axis=1)
    recall_mean = np.mean(recall, axis=1)
    fscore_mean = np.mean(fscore, axis=1)

    # Écart type par model
    losses_std = np.std(losses, axis=1)
    precision_std = np.std(precision, axis=1)
    recall_std = np.std(recall, axis=1)
    fscore_std = np.std(fscore, axis=1)

    losses_mean_std = np.concatenate((losses_mean, losses_std), axis=0)
    precision_mean_std = np.concatenate((precision_mean, precision_std), axis=0)
    recall_mean_std = np.concatenate((recall_mean, recall_std), axis=0)
    fscore_mean_std = np.concatenate((fscore_mean, fscore_std), axis=0)
    return losses_mean_std, precision_mean_std, recall_mean_std, fscore_mean_std


from SoftmaxClassifier import SoftmaxClassifier
nb_run = 2
models = [
    SoftmaxClassifier(),
]
scoring = ['neg_log_loss', 'precision_macro','recall_macro','f1_macro']
compare(models,X_train,y_train_label,nb_run) # ,scoring)  mettre 1
i = 0
