# Librerias EDA

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import f_classif, SelectKBest
import numpy as np
import json
from pickle import dump

# Librerias ML

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV

#Recopilamos datos
data_url = "https://raw.githubusercontent.com/4GeeksAcademy/decision-tree-project-tutorial/main/diabetes.csv"
sep = (",")

def DataCompiler(url, sep):
    data = pd.read_csv(url, sep = sep)

    #Guardamos el csv en local
    data.to_csv("../data/raw/raw_data.csv", sep=";")

    return data

data = DataCompiler(data_url, sep)

#Obtenemos informacion sobre el dataset

def DataInfo(dataset):
    print(f"Dataset dimensions: {dataset.shape}")
    print(f"\nDataset information:\n{dataset.info()}")
    print(f"\nDataset nan-values: {dataset.isna().sum().sort_values(ascending=False)}")
    

DataInfo (data)

#Funcion para eliminar duplicados

#Columna identificadora del Dataset.

def EraseDuplicates(dataset, current_id = ""):
    older_shape = dataset.shape
    id = current_id
    
    if (id != ""):
        dataset.drop(id , axis = 1, inplace = True)
                     
    if (dataset.duplicated().sum()):
        print(f"Total number of duplicates {dataset.duplicated().sum()}")
        print ("Erase duplicates...")
        dataset.drop_duplicates(inplace = True)
    else:
        print ("No coincidences.")
        pass
    
    print (f"The older dimension of dataset is {older_shape}, and the new dimension is {dataset.shape}.")
    
    return dataset

data = EraseDuplicates(data)

#Analisis sobre variables categoricas

categoric_predictors = ["Pregnancies", "Outcome"]

def CategoricGraf(dataset, lst):
    #Creamos la figura
    fig, axis = plt.subplots(1, 2, figsize=(15,5))

    #Creamos las graficas necesarias
    for i in range(len(lst)):
        sns.histplot( ax = axis[i], data = dataset, x = lst[i])
   

    #Mostramos el grafico.
    plt.tight_layout()
    plt.show()

CategoricGraf(data, categoric_predictors)

# Analisis sobre variables numericas

continuous_lst = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "Age", "DiabetesPedigreeFunction"]
vector_ax = [4,4]

def NumericalGraf(dataset, lst, ax_dimension):

    x_pos = 0
    y_pos = 0
    #Creamos la figura
    fig, axis = plt.subplots(ax_dimension[0], ax_dimension[1], figsize=(15,8), gridspec_kw={"height_ratios" : [6,1,6,1]})

    #Creamos las graficas necesarias
    for i in range(len(lst)):
       sns.histplot( ax = axis[x_pos,y_pos], data = dataset, x = lst[i], kde = True).set(xlabel = None)
       sns.boxplot( ax = axis[x_pos + 1,y_pos], data = dataset, x = lst[i])
       y_pos += 1
       if (y_pos == ax_dimension[1]):
           x_pos += 2
           y_pos = 0
    
    plt.tight_layout()
    plt.show()

NumericalGraf(data, continuous_lst, vector_ax)

#Analisis Categorico/categorico
target = "Outcome"
categoric_predictors = ["Pregnancies"]

def CatCatAnalysi(dataset, target, lst):
    #Creamos la figura
    fig, axis = plt.subplots(1, 2, figsize=(15,5))

    #Creamos las graficas.
    for i in range(len(lst)):
        sns.countplot(ax = axis[i], data = dataset, x = lst[i], hue = target)


    plt.tight_layout()
    plt.show()

CatCatAnalysi(data, target, categoric_predictors)

continuous_y = "DiabetesPedigreeFunction"
continuous_lst = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "Age"]
vector_ax = [4,3]

def NumNumAnalysi(dataset, y, x_list, ax_dimension):
    #Creamos la figura
    fig, axis = plt.subplots(ax_dimension[0], ax_dimension[1], figsize=(15,10))

    x_pos = 0
    y_pos = 0

    for i in range(len(x_list)):
        if (y_pos == 0):
            sns.regplot( ax = axis[x_pos, y_pos], data = dataset, x = x_list[i], y = y)
        else:
            sns.regplot( ax = axis[x_pos, y_pos], data = dataset, x = x_list[i], y = y).set(ylabel = None)

        if (y_pos < ax_dimension[1] - 1):
            sns.heatmap( data[[y,x_list[i]]].corr(), annot=True, fmt=".2f", ax = axis[x_pos + 1, y_pos], cbar=False, xticklabels = False)
        else:
            sns.heatmap( data[[y,x_list[i]]].corr(), annot = True, fmt = ".2f", ax = axis[x_pos + 1, y_pos], xticklabels = False) 
        
        y_pos = y_pos + 1
        if (y_pos == ax_dimension[1]):
            y_pos = 0
            x_pos = x_pos + 2
    
    plt.tight_layout()
    plt.show()

NumNumAnalysi(data, continuous_y, continuous_lst, vector_ax)

vector_ax = [2,4]
continuous_lst = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "Age", "DiabetesPedigreeFunction"]

def CombTargPred(dataset, target_class, lst, ax_dimension):
    
    y_pos = 0
    x_pos = 0

    fig, axis = plt.subplots(ax_dimension[0], ax_dimension[1], figsize = (15, 8))

    for i in range(len(lst)):
        
        if (y_pos == 0):
            sns.regplot(ax = axis[x_pos,y_pos], data = dataset, x = lst[i], y = target_class)
        elif (y_pos != 0):
            sns.regplot(ax = axis[x_pos, y_pos], data = dataset, x = lst[i], y = target_class).set(ylabel=None)

        y_pos += 1

        if (y_pos == ax_dimension[1]):
            x_pos += 1
            y_pos = 0

    

    plt.tight_layout()
    plt.show()

CombTargPred(data, target, continuous_lst, vector_ax)

#Tabla de correlaciones

def CorrelationPlot(dataset):
    fig, axis = plt.subplots(figsize=(18,10))

    sns.heatmap(dataset.corr(), annot=True, fmt=".2f")

    plt.tight_layout()
    plt.show()

CorrelationPlot(data)

sns.pairplot(data)

# Comprobamos las metricas de la tabla.

data.describe()

#Creamos una funcion para transformar los outliers.

def SplitOutliers(dataset, target):
    
    dataset_with_outliers = dataset.copy()
    
    #Establecemos los limites.
    for i in dataset.columns:
        if (i == target):
            print(f"Target detected: {target}")
            pass
        
        #Esta parte la tengo que mejorar para poder clasificar los campos categoricos
        elif (i == "Urban_rural_code"):
            print(f"Categorical predictor: Urban_rural_code")
            pass

        else:
            stats = dataset[i].describe()
            iqr = stats["75%"] - stats["25%"]
            upper_limit = float(stats["75%"] + (2 * iqr))
            lower_limit = float(stats["25%"] - (2 * iqr))
            if (lower_limit < 0):
                lower_limit = 0

            #Ajustamos el outlier por encima.
            dataset[i] = dataset[i].apply(lambda x : upper_limit if (x > upper_limit) else x)

            #Ajustamos el outlier por debajo.
            dataset[i] = dataset[i].apply(lambda x : lower_limit if (x < lower_limit) else x)

            #Guardamos los límites en un json.
            with open (f"../data/interim/outerliers_{i}.json", "w") as j:
                json.dump({"upper_limit" : upper_limit, "lower_limit" : lower_limit}, j)
    
    return dataset_with_outliers, dataset

data_with_outliers, data_without_outliers = SplitOutliers (data, target)

#Comprobamos si existen valores faltantes.

data_with_outliers.isna().sum().sort_values()

data_without_outliers.isna().sum().sort_values()

# Primero dividimos los dataframes entre test y train

def SplitData(dataset, target):
    #Aislamos el target
    features = []
    for i in dataset.columns:
        if (i == target):
            pass
        else:
            features.append(i)
    
    x = dataset.drop(target, axis = 1)[features]
    y = dataset[target].squeeze()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42)

    return x_train, x_test, y_train, y_test

x_train_with_outliers, x_test_with_outliers, y_train, y_test = SplitData(data_with_outliers, target)
x_train_without_outliers, x_test_without_outliers, _, _ = SplitData(data_without_outliers, target)

y_train.to_csv("../data/processed/y_train.csv", index_label = False)
y_test.to_csv("../data/processed/y_test.csv", index_label = False)

#Tenemos que escalar los dataset con Normalizacion y con Escala mM (min-Max)

#Normalizacion
def StandardScaleData(dataset):
    #Aislamos el target
    features = []
    for i in dataset.columns:
        if (i == target):
            pass
        else:
            features.append(i)

    scaler = StandardScaler()
    scaler.fit(dataset)

    x_scaler = scaler.transform(dataset)
    x_scaler = pd.DataFrame(dataset, index = dataset.index, columns = features)
    
    if(dataset is x_train_with_outliers):
        dump(scaler, open("../data/interim/standar_scale_with_outliers.sav", "wb"))

    elif(dataset is x_train_without_outliers):
        dump(scaler, open("../data/interim/standar_scale_without_outliers.sav", "wb"))

    return x_scaler

x_train_with_outliers_standarscale = StandardScaleData(x_train_with_outliers)
x_train_without_outliers_standarscale = StandardScaleData(x_train_without_outliers)
x_test_with_outliers_standscale = StandardScaleData(x_test_with_outliers)
x_test_without_outliers_standscale = StandardScaleData(x_test_without_outliers)

#Escala mM
def MinMaxScaleData(dataset):
    #Aislamos el target
    features = []
    for i in dataset.columns:
        if (i == target):
            pass
        else:
            features.append(i)

    scaler = MinMaxScaler()
    scaler.fit(dataset)

    x_scaler = scaler.transform(dataset)
    x_scaler = pd.DataFrame(dataset, index = dataset.index, columns = features)

    if(dataset is x_train_with_outliers):
        dump(scaler, open("../data/interim/min-Max_Scale_with_outliers.sav", "wb"))

    elif(dataset is x_train_without_outliers):
        dump(scaler, open("../data/interim/min-Max_Scale_without_outliers.sav", "wb"))

    return x_scaler

x_train_with_outliers_mMScale = MinMaxScaleData(x_train_with_outliers)
x_train_without_outliers_mMScale = MinMaxScaleData(x_train_without_outliers)
x_test_with_outliers_mMScale = MinMaxScaleData(x_test_with_outliers)
x_test_without_outliers_mMScale = MinMaxScaleData(x_test_without_outliers)

#Seleccion de caracteristicas

def SelectFeaturesTrain(dataset, y, filename, k = 5):
    sel_model = SelectKBest(f_classif, k=k)
    sel_model.fit(dataset, y)
    col_name = sel_model.get_support()
    x_sel = pd.DataFrame(sel_model.transform(dataset), columns = dataset.columns.values[col_name])
    dump(sel_model, open(f"../data/interim/{filename}.sav", "wb"))
    train_cols = x_sel.columns
    return x_sel, train_cols

def SelectFeaturesTest(dataset, y, filename, train_cols, k = 5):
    dataset = pd.DataFrame(dataset[train_cols])
    sel_model = SelectKBest(f_classif, k=k)
    sel_model.fit(dataset, y)
    col_name = sel_model.get_support()
    x_sel = pd.DataFrame(sel_model.transform(dataset), columns = dataset.columns.values[col_name])
    dump(sel_model, open(f"../data/interim/{filename}.sav", "wb"))
    return x_sel

#Dataset sin normalizacion
x_train_sel_with_outliers, cols = SelectFeaturesTrain(x_train_with_outliers, y_train, "x_train_with_outliers")
x_test_sel_with_outliers = SelectFeaturesTest(x_test_with_outliers, y_test, "x_test_sel_with_outliers", cols)
x_train_sel_without_outliers, cols = SelectFeaturesTrain(x_train_without_outliers, y_train, "x_train_without_outliers")
x_test_sel_without_outliers = SelectFeaturesTest(x_test_with_outliers, y_test, "x_test_sel_without_outliers", cols)

#Dataset Normalizado
x_train_sel_with_outliers_standarscale, cols = SelectFeaturesTrain(x_train_with_outliers_standarscale, y_train, "x_train_with_outliers_standarscale")
x_test_sel_with_outliers_standarscale = SelectFeaturesTest(x_test_with_outliers, y_test, "x_test_sel_with_outliers_standarscale", cols)
x_train_sel_without_outliers_standarscale, cols = SelectFeaturesTrain(x_train_without_outliers_standarscale, y_train, "x_train_sel_without_outliers_standarscale")
x_test_sel_without_outliers_standarscale = SelectFeaturesTest(x_test_with_outliers, y_test, "x_test_sel_without_outliers_standarscale", cols)

#Train dataset Escalado min-Max
x_train_sel_with_outliers_mMScale, cols = SelectFeaturesTrain(x_train_with_outliers_mMScale, y_train, "x_test_with_outliers_mMScaler")
x_test_sel_with_outliers_mMScale = SelectFeaturesTest(x_test_with_outliers, y_test, "x_test_sel_with_outliers_mMScale", cols)
x_train_sel_without_outliers_mMScale, cols = SelectFeaturesTrain(x_train_without_outliers_mMScale, y_train, "x_train_without_outliers_mMScaler")
x_test_sel_without_outliers_mMScale = SelectFeaturesTest(x_test_with_outliers, y_test, "x_test_sel_without_outliers_mMScale", cols)

#Para acabar nos guardamos los datasets en un csv

def DataToCsv(dataset, filename):
    return dataset.to_csv(f"../data/processed/{filename}.csv", index_label = False)

DataToCsv(x_train_sel_with_outliers, "x_train_sel_with_outliers")
DataToCsv(x_test_sel_with_outliers, "x_test_sel_with_outliers")
DataToCsv(x_train_sel_without_outliers, "x_train_sel_without_outliers")
DataToCsv(x_test_sel_without_outliers, "x_test_sel_without_outliers")
DataToCsv(x_train_sel_with_outliers_standarscale, "x_train_sel_with_outliers_standarscale")
DataToCsv(x_test_sel_with_outliers_standarscale, "x_test_sel_with_outliers_standarscale")
DataToCsv(x_train_sel_without_outliers_standarscale, "x_train_sel_without_outliers_standarscale")
DataToCsv(x_test_sel_without_outliers_standarscale, "x_test_sel_without_outliers_standarscale")
DataToCsv(x_train_sel_with_outliers_mMScale, "x_train_sel_with_outliers_mMScale")
DataToCsv(x_test_sel_with_outliers_mMScale, "x_test_sel_with_outliers_mMScale")
DataToCsv(x_train_sel_without_outliers_mMScale, "x_train_sel_without_outliers_mMScale")
DataToCsv(x_test_sel_without_outliers_mMScale, "x_test_sel_without_outliers_mMScale")

######## MACHINE LEARNING ######

traindfs = [ x_train_sel_with_outliers, x_train_sel_without_outliers, x_train_sel_with_outliers_standarscale]
testdfs = [ x_test_sel_with_outliers, x_test_sel_without_outliers, x_test_sel_with_outliers_standarscale]

def DecisionTree (traindataset, testdataset):
    results = []
    models = []

    for i in range(len(traindataset)):
        model = DecisionTreeClassifier(random_state=42)
        traindf = traindataset[i]

        model.fit(traindf, y_train)
        y_train_predict = model.predict(traindf)
        y_test_predict = model.predict(testdataset[i])

        models.append(model)
        result = {"index" : i, "train_score" : accuracy_score(y_train, y_train_predict), "test_score" : accuracy_score(y_test, y_test_predict)}
        results.append(result)

        plt.figure(figsize=(18,5))
        tree.plot_tree(model, feature_names = traindf.columns, filled = True)

        plt.tight_layout()
        plt.show()

    with open ("../data/processed/accuracy.json", "w") as j:
        json.dump( results, j)

    return results, models

pre_results, pre_models = DecisionTree(traindfs, testdfs)

#Creamos el diccionario de hyperparametros

hyperparameters = {"criterion" : ["gini", "entropy", "log_loss"], "splitter" : ["best", "random"], "max_depth" : np.random.randint(1, 5, size=3), "min_samples_split" : np.random.randint(5, size=3), "min_samples_leaf" : np.random.randint(1, 5, size=3), "random_state" : np.random.randint(1, 100, size=3)}

#Pasamos el modelo preentrenado con los hiperparametros (En este caso paso un grid pero podría pasar un RandomSearchCV)
grid = GridSearchCV(pre_models[0], hyperparameters, scoring="accuracy")

grid.fit(x_train_sel_without_outliers_standarscale, y_train)

grid.best_params_

#Guardamos el modelo entrenado

clf = grid.best_estimator_

dump(clf, open(f"../models/decision_tree.sav", "wb"))

y_test_predict = clf.predict(x_test_sel_with_outliers)

score = accuracy_score(y_test, y_test_predict)

print(score)

#Creamos la matriz de confusion para ver donde estaban los errores

confmatrx = pd.DataFrame(confusion_matrix(y_test, y_test_predict))

plt.figure(figsize=(5,5))

sns.heatmap(data = confmatrx, annot = True, fmt="d", cbar = False)

plt.show()