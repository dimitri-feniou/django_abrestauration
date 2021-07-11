import numpy as np
import pandas as pd
from pandas_ods_reader import read_ods

# importation of linked python file
import fonctions_machine_learning as ml
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

pd.set_option('display.max_columns', None)

#read csv main data
sheet_name = "BDD"
df = read_ods('imad_ExportABResto.ods', sheet_name)

#read csv events data
sheet_name = "Events"
df_events = read_ods('ExportABResto_v1.1.ods', sheet_name)

#
index_input = len(df)+1


#
df = ml.clean_df_main_data(df)

#
df_events = ml.clean_df_events(df_events)



#
df_merge = pd.merge(df,df_events, how='inner', on='Dates')



#
df_merge = ml.clean_merge_main_events_before_encoding(df_merge)


#if all point de vente
liste_features_categoriel_all = ["Jour", "Horaire"]

# if all point de vente
list_not_usefull_and_string_all =["Articles", "Dates", "Point_de_vente"]

#
list_of_product_to_pred = df_merge.Articles.unique().tolist()


#input of event to predict
jour_event = "Samedi"
horaire_event = "14h00"
distance_event = "+ de 200 km"
no_show_event = "Super"
spectateur_event = 23000
#temperature à recuperer via l'api (pour linstant input pour test)
temperature_event = 5

# creer list d'input a ajouter au df_merge
list_input = [np.NAN, np.NAN, np.NAN, np.NAN, jour_event, horaire_event, spectateur_event, temperature_event]

#
df_outpout = pd.DataFrame(columns = ["article", "y_pred", "score_train", "score_test"])


for i in ["Coca"]:

    #
    df_produit = df_merge[df_merge["Articles"]==i]
    if len(df_produit) < 2:
        continue
    

    # ajouter les input en fin du dataframe du datafraSme df_produit
    df_produit = df_produit.append({'Jour': jour_event, "Horaire": horaire_event, 'Spectateurs': spectateur_event , "Temperature": temperature_event}, ignore_index=True)

    #
    df_merge_encoded = ml.encode_remove_feature_and_feature_not_usefull(df_produit, liste_features_categoriel_all, list_not_usefull_and_string_all)
    #

    X_encoded_normalize, y = ml.define_features_target_and_normalize(df_merge_encoded)

    # on garde la derniere ligne  X_encoded_normalize
    X_event_encoded_normalized = X_encoded_normalize.tail(1)

    # on supprime la derniere ligne de X_encoded_normalize et de y
    X_encoded_normalize.drop(X_encoded_normalize.tail(1).index, inplace=True)
    y.drop(y.tail(1).index, inplace=True)

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X_encoded_normalize, y, test_size=0.3,
                                                        random_state=1)  # 70% training and 30% test
    #
    model = ml.modele_fitted(X_train, y_train)

    # score train
    y_predict_train = model.predict(X_train)
    score_train = r2_score(y_train, y_predict_train)

    #score test
    y_predict_test = model.predict(X_test)
    score_test = r2_score(y_test, y_predict_test)

    #
    y_pred = ml.prediction(model, X_event_encoded_normalized)[0]

    # remplissage du dataframe df_output
    df_outpout = df_outpout.append({'article': i, 'y_pred': y_pred, 'score_train': score_train , "score_test": score_test},ignore_index=True)


print(df_outpout)



