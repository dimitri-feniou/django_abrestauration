from io import StringIO
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import math
import datetime

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def clean_df_main_data (df: pd.DataFrame):
    #
    # managing Date errors and formats

    df = df.replace({'Date': {'11 Aout 2019': '2019-08-11', '15 Aout 2019': '2019-08-15'}})
    df['Dates'] = pd.to_datetime(df['Date'])
    df = df.drop(["Date"], axis=1)

    # replace "nan" by 0 when the stock is 0
    list_1 = ["Controle", "Reste"]
    for i in list_1:
        df[i] = df.apply(lambda x: 0 if (math.isnan(x[i]) == True) and (x["Stock"] == 0.0) else x[i], axis=1)

    # create feature "article first word"
    df["articles first word"] = df["Articles"].str.split(" ").str[0]
    # remove article "gobelet" because not prediction for this article
    to_del = df[df["articles first word"].isin(["Gobelets"])].index.tolist()
    df = df.drop(to_del)

    # create feature "vente_reel" coresponding "controle" - "reste"
    df["vente_reel"] = df["Controle"] - df["Reste"]

    # drop columns not usefull
    column = ['Stock', 'Controle', 'Reste', 'articles first word', 'Perte/sale', 'Responsable',
              'Stade']
    df.drop(columns=column, inplace=True, axis=1)

    # clean Spelling mistake
    df = df.replace({'Articles': {'Cheddar (88 tranches)': 'Cheddar (84 tranches)',
                                  'Knack rose': 'Knack Rose', "Biere speciale": "Biere Speciale",
                                  "Fisher Doreleï": "Fischer Doreleï", "Powerrade": "Powerade",
                                  "Bière Spéciale": "Biere Speciale"}})

    # remove articles not relevant
    i = df[(df.Articles == '13,24€ tickets moyen')].index
    j = df[(df.Articles == 'Total Sandwich')].index
    df = df.drop(i)
    df = df.drop(j)

    # remove all missing values
    df = df.dropna()

    # drop negative values for feature "vente_reel"
    df = df.drop(df[df.vente_reel < 0].index)

    return (df)


def clean_df_events (df: pd.DataFrame):
    #

    # managing Date errors and formats
    df = df.replace({'Date': {'11 Aout 2019': '2019-08-11', '15 Aout 2019': '2019-08-15'}})
    df['Dates'] = pd.to_datetime(df['Date'])
    df = df.drop(["Date"], axis=1)

    return (df)

def clean_merge_main_events_before_encoding (df: pd.DataFrame):
    #

    # drop columns not usefull
    column = ["NoShow", "Adversaire", "Visiteurs", "Distance", "Stade"]
    df.drop(columns=column, inplace=True, axis=1)

    # rename columns remplacing space by "_"
    df.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)

    return (df)

# input articles to predict?




#if all point de vente
liste_features_categoriel_all = ["Jour", "Horaire","Point_de_vente"]

# for all point de vente
list_not_usefull_and_string_all =["Articles", "Dates"]


def encode_and_bind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    resp = pd.concat([original_dataframe, dummies], axis=1)
    return (resp)

# encodeur and remove features not usefull
def encode_remove_feature_and_feature_not_usefull (df, list_features_categoriel, list_not_usefull):

    for i in list_features_categoriel:
        df_encode = encode_and_bind(df, i)
        df_encode.drop(columns=list_features_categoriel, inplace=True, axis=1)
        df_encode.drop(columns=list_not_usefull, inplace=True, axis=1)

    return (df_encode)


def define_features_target_and_normalize(df: pd.DataFrame):
    #

    # define target and features
    X = df.drop(axis=1, columns="vente_reel")  # Features
    y = df["vente_reel"]  # Target variable

    # create a scaler object
    std_scaler = StandardScaler()
    # fit and transform the data
    X_normalize = pd.DataFrame(std_scaler.fit_transform(X), columns=X.columns)

    return (X_normalize, y)



def modele_fitted(X_train, y_train):
    # Fit regression model
    param = {'kernel': ('linear', 'poly', 'rbf', 'sigmoid'), 'C': [1, 5, 10], 'degree': [3, 8],
             'coef0': [0.01, 10, 0.5], 'gamma': ('auto', 'scale')}

    model_SVR = SVR()
    #### parametre à changer ou searchCV avec best estimator
    grids = GridSearchCV(model_SVR, param, cv=5, n_jobs=-1, verbose=2)
    # training
    grids.fit(X_train, y_train)
    return (grids)




def prediction(model_fitted, X_event):
    #

    y_predict = model_fitted.predict(X_event)
    return (y_predict)




