import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
from sklearn.utils import shuffle
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

def calculate_metrics(**kwargs):
    # recover model
    model = kwargs.get('model', None)
    if model is None:
        raise ValueError("Model not provided")
    # recover data
    y_true = kwargs.get('Y_validation', None)
    x_true = kwargs.get('X_validation', None)
    y_pred = model.predict(x_true)
    
    f1_score_weighted = f1_score(y_true, y_pred, average='weighted') 
    f1_score_macro = f1_score(y_true, y_pred, average='macro')
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    presicion_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted')
    recall_macro = recall_score(y_true, y_pred, average='macro')
    score = model.score(x_true, y_true)
    #store metrics in a dictionary
    dict_metrics = {
        'f1_score_weighted': f1_score_weighted,
        'f1_score_macro': f1_score_macro,
        'precision_weighted': precision_weighted,
        'precision_macro': presicion_macro,
        'recall_weighted': recall_weighted,
        'recall_macro': recall_macro,
        'score': score
    }
    return dict_metrics, y_pred
    

def evaluate_model(**kwargs):
    name = kwargs.pop('name', None)
    dict_metrics, y_pred = calculate_metrics(**kwargs)
    print(f'.:EVALUATING MODEL: {name}:.')
    print('='*10)
    print('Predicciones')
    print('='*10)
    
    ypred_unique = set(y_pred)
    for i in ypred_unique:
        print(f'Predicción clase {i}: {sum(y_pred == i)}')
    print('='*10)
    print('Métricas')
    print('='*10)
    print(f'F1 Score: {dict_metrics["f1_score_weighted"]}')
    print(f'F1 Score (macro): {dict_metrics["f1_score_macro"]}')
    print(f'Precision: {dict_metrics["precision_weighted"]}')
    print(f'Precision (macro): {dict_metrics["precision_macro"]}')
    print(f'Recall: {dict_metrics["recall_weighted"]}')
    print(f'Recall (macro): {dict_metrics["recall_macro"]}')
    print(f'Score: {dict_metrics["score"]}')
    return y_pred

def oversample(df, growth_rate: int = None):
    """
    Oversample the minority class in a dataframe.
    :param df: DataFrame to oversample
    :param growth_rate: Growth rate for the minority class
    :return: Oversampled DataFrame
    """
    
    # Get the counts of each class
    counts = df['target'].value_counts()
    majority_class = counts.idxmax()
    minority_class = counts.idxmin()
    if growth_rate is None:
            growth_rate = counts[majority_class] /counts[minority_class] 
            growth_rate = np.floor(growth_rate)
    
    
    # Calculate the number of samples needed for the minority class
    num_samples_needed = int(counts[minority_class] * growth_rate)

    # Oversample the minority class
    minority_class_samples = df[df['target'] == minority_class]
    oversampled_minority_class = minority_class_samples.sample(num_samples_needed - counts[minority_class], replace=True)

    # Combine the original DataFrame with the oversampled minority class
    oversampled_df = pd.concat([df, oversampled_minority_class], ignore_index=True)
    # Shuffle the DataFrame
    oversampled_df = shuffle(oversampled_df, random_state=42)
        
    return oversampled_df

if __name__ == '__main__':
    np.random.seed(42)  # For reproducibility
    data = {
    'feature_1': np.random.randn(100),
    'feature_2': np.random.randn(100),
    'target': np.random.choice([0, 1, 2], size=100, p=[0.7, 0.2, 0.1])  # Unbalanced classes
}

    df = pd.DataFrame(data)

    # Check class distribution before oversampling
    print("Class distribution before oversampling:")
    print(df['target'].value_counts())
    oversampled_df = oversample(df)

    # Check class distribution after oversampling
    print("\nClass distribution after oversampling:")
    print(oversampled_df['target'].value_counts())