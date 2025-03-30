def eda(df, name, id='id'):
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
    duplicates = df.duplicates()
    fuzzyDuplicates = df.duplicates(id)
    print(f'Número de duplicados generales: {duplicates.sum()}')
    print(f'Número de duplicados engañosos: {fuzzyDuplicates.sum()}')
    #Check for nan values existence.
    print('Valores faltantes por cada columna')
    print(df.isna().sum())
