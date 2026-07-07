from Pesquisa_principal.silver.otimizadores.dataset_balanceado import dataset

if __name__ == "__main__":

    X_train, X_test, y_train, y_test = dataset()['X_train'], dataset()['X_test'], dataset()['y_train'], dataset()['y_test']

    print(X_train.head())
    print(X_test.head())
    print(y_train.head())
    print(y_test.head())