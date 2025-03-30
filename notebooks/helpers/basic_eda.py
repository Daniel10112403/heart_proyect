import pandas as pd
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
if __name__ == '__main__':
    
    raw_patients = pd.read_csv('C:/Users/danie/Desktop/master/Data_Science_fundamentals/heart_disease_proyect/data/raw/raw_medical_records.csv')
    eda(raw_patients, 'raw_patients')