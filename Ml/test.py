import pandas as pd
from pandas_ods_reader import read_ods

# importation of linked python file
import fonctions_machine_learning as ml
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


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

print(df.head())
#
df_events = ml.clean_df_events(df_events)

print(df_events.head())

#
df_merge = pd.merge(df,df_events, how='inner', on='Dates')

print(df_merge.head())

#
df_merge = ml.clean_merge_main_events_before_encoding(df_merge)

print(df_merge.head())

#if all point de vente
liste_features_categoriel_all = ["Jour", "Horaire"]

# if all point de vente
list_not_usefull_and_string_all =["Articles", "Dates", "Point_de_vente"]

#
list_of_product_to_pred = df_merge.Articles.unique()


#input of event to predict
#jour_event = input("jour?")
#horaire_event = input("horaire?")
#distance_event = input("distance?")
#no_show_event = input("noshow?")
#spectateur_event = int(input("nombre spectateur?"))
#temperature à recuperer via l'api (pour linstant input pour test)
#temperature_event = int(input("temperature?"))

# creer list d'input a ajouter au df_merge
#list_input = ["/", "/", jour_event, horaire_event, spectateur_event, temperature_event]
list_input = ["/", "/", "/", "/", "Mardi", "14h00", 23000, 15]


print("ca charge........")
#
df_outpout = pd.DataFrame(columns = ["article", "y_pred", "score_train", "score_test"])

i = "Coca"

#
df_produit = df_merge[df_merge["Articles"]==i]

# ajouter les input en fin du dataframe du dataframe df_produit
df_produit.loc[index_input]=list_input

#
df_merge_encoded = ml.encode_remove_feature_and_feature_not_usefull(df_produit, liste_features_categoriel_all, list_not_usefull_and_string_all)
#
X_encoded_normalize, y = ml.define_features_target_and_normalize(df_merge_encoded)

X_event_encoded_normalized = X_encoded_normalize.tail(1)

X_encoded_normalize.drop(X_encoded_normalize.tail(1).index, inplace=True)
y.drop(y.tail(1).index, inplace=True)


print(type(X_encoded_normalize))
print("1111111")
print(X_encoded_normalize)
print(X_event_encoded_normalized)

X_train, X_test, y_train, y_test = train_test_split(X_encoded_normalize, y, test_size=0.3, random_state=1)

model = ml.modele_fitted(X_train, y_train)

y_predict_train = model.predict(X_train)
score_train = r2_score(y_train, y_predict_train)

y_predict_test = model.predict(X_test)
score_test = r2_score(y_test, y_predict_test)

print(score_train)
print("222222222")
print(score_test)

y_pred = ml.prediction(model, X_event_encoded_normalized)

df_outpout = df_outpout.append({'article': i, 'y_pred': y_pred, 'score_train': score_train , "score_test": score_test},ignore_index=True)
print(y_pred)
print(df_outpout)