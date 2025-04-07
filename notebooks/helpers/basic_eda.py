import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
def eda(df, name, id:str = None):
    print('='*10)
    print('.:STARTING EDA:.')
    print('='*10)
    print('DATA SET:', name)
    print(df.head())
    print(' 1. Data set dimensions: ', df.shape)
    # Check general information
    print('Información general')
    print(df.info())
    # Check duplicates
    g_duplicates = df.duplicated()
    print(f'Número de duplicados generales: {sum(g_duplicates)}')
    if id is not None:
        fuzzyDuplicates = df.duplicated(id)
        print(f'Número de duplicados engañosos: {sum(fuzzyDuplicates)}')
    #Check for nan values existence.
    print('Valores faltantes por cada columna')
    print(df.isna().sum())
# test function

def statistical_information(df: pd.DataFrame, 
                            columns:list = None ):
    if columns is None:
        columns = df.columns
    for i in columns:
        df[i].describe()
        print('='*10)
        print(f'Información de la la columna {i}')
        print(df[i].describe())
        print('='*10)

def separate_target(df: pd.DataFrame,
                    target_column: str):
    characteristics = df.drop(target_column, axis=1)
    target = df[target_column]
    return characteristics, target

def evaluate_model(**kwargs):
    model = kwargs.get('model', None)
    if model is None:
        raise ValueError("Model not provided")
    print(f'.:EVALUATING MODEL: {kwargs["name"]}:.')
    print('='*10)
    print('Predicciones')
    print('='*10)
    y_pred = model.predict(kwargs['X_validation'])    
    ypred_unique = set(y_pred)
    for i in ypred_unique:
        print(f'Predicción clase {i}: {sum(y_pred == i)}')
    print('Métricas')
    print('='*10)
    print(f'F1 Score: {f1_score(kwargs["Y_validation"], y_pred, average="weighted")}')
    print(f'F1 Score (macro): {f1_score(kwargs["Y_validation"], y_pred, average="macro")}')
    print(f'Precision: {precision_score(kwargs["Y_validation"], y_pred, average="weighted", zero_division=0)}')
    print(f'Precision (macro): {precision_score(kwargs["Y_validation"], y_pred, average="macro", zero_division=0)}')
    print(f'Recall: {recall_score(kwargs["Y_validation"], y_pred, average="weighted")}')
    print(f'Recall (macro): {recall_score(kwargs["Y_validation"], y_pred, average="macro")}')
    print(f'Score: {model.score(kwargs["X_validation"], kwargs["Y_validation"])}')
    return y_pred
if __name__ == '__main__':
    
    raw_patients = pd.read_csv('C:/Users/danie/Desktop/master/Data_Science_fundamentals/heart_disease_proyect/data/raw/raw_medical_records.csv')
    eda(raw_patients, 'raw_patients')