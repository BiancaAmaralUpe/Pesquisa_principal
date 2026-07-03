# ======================================================================================
# encoding.py
# ======================================================================================
# Responsabilidade:
# - Identificar variáveis categóricas
# - Aplicar One-Hot Encoding nas colunas categóricas
# - Manter colunas numéricas e booleanas, se existirem
# - Salvar o encoder ajustado na base de treino
# ======================================================================================

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def identificar_colunas_categoricas(
    dataframe: pd.DataFrame,
) -> list[str]:
    """
    Identifica colunas categóricas da base.
    """

    colunas_categoricas = dataframe.select_dtypes(
        include=[
            "object",
            "string",
            "category",
        ]
    ).columns.tolist()

    print("\n" + "=" * 80)
    print("IDENTIFICAÇÃO DE COLUNAS CATEGÓRICAS")
    print("=" * 80)

    print(f"Quantidade de colunas categóricas: {len(colunas_categoricas)}")

    for coluna in colunas_categoricas:
        print(f"- {coluna}")

    return colunas_categoricas

def identificar_colunas_numericas_booleanas(
    dataframe: pd.DataFrame,
    colunas_categoricas: list[str],
) -> list[str]:
    """
    Identifica colunas que não precisam de One-Hot Encoding.
    """

    colunas_numericas_booleanas = [
        coluna
        for coluna in dataframe.columns
        if coluna not in colunas_categoricas
    ]

    print("\n" + "=" * 80)
    print("IDENTIFICAÇÃO DE COLUNAS NUMÉRICAS/BOOLEANAS")
    print("=" * 80)

    print(
        "Quantidade de colunas numéricas/booleanas: "
        f"{len(colunas_numericas_booleanas)}"
    )

    for coluna in colunas_numericas_booleanas:
        print(f"- {coluna}")

    return colunas_numericas_booleanas

def criar_encoder(
    colunas_categoricas: list[str],
    colunas_numericas_booleanas: list[str],
) -> ColumnTransformer:
    """
    Cria o transformador de encoding.

    O OneHotEncoder transforma categorias em colunas binárias.
    O handle_unknown='ignore' evita erro caso apareça categoria nova no teste.
    """

    transformadores = []

    if colunas_categoricas:
        transformadores.append(
            (
                "one_hot_encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=np.int8,
                ),
                colunas_categoricas,
            )
        )

    if colunas_numericas_booleanas:
        transformadores.append(
            (
                "manter_numericas_booleanas",
                "passthrough",
                colunas_numericas_booleanas,
            )
        )

    encoder = ColumnTransformer(
        transformers=transformadores,
        verbose_feature_names_out=False,
    )

    return encoder

def aplicar_encoding_treino_teste(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    caminho_encoder: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, ColumnTransformer]:
    """
    Aplica One-Hot Encoding em X_train e X_test.

    O encoder é ajustado apenas no treino.
    Depois, o mesmo encoder transforma treino e teste.
    """

    print("\n" + "=" * 80)
    print("APLICAÇÃO DO ONE-HOT ENCODING")
    print("=" * 80)

    colunas_categoricas = identificar_colunas_categoricas(X_train)

    colunas_numericas_booleanas = identificar_colunas_numericas_booleanas(
        dataframe=X_train,
        colunas_categoricas=colunas_categoricas,
    )

    encoder = criar_encoder(
        colunas_categoricas=colunas_categoricas,
        colunas_numericas_booleanas=colunas_numericas_booleanas,
    )

    print("\nAjustando encoder no X_train...")
    X_train_encoded_array = encoder.fit_transform(X_train)

    print("Aplicando encoder no X_test...")
    X_test_encoded_array = encoder.transform(X_test)

    nomes_colunas_encoded = encoder.get_feature_names_out()

    X_train_encoded = pd.DataFrame(
        X_train_encoded_array,
        columns=nomes_colunas_encoded,
        index=X_train.index,
    )

    X_test_encoded = pd.DataFrame(
        X_test_encoded_array,
        columns=nomes_colunas_encoded,
        index=X_test.index,
    )

    caminho_encoder.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoder, caminho_encoder)

    print("\nEncoding finalizado com sucesso.")
    print(f"Formato original X_train: {X_train.shape}")
    print(f"Formato original X_test: {X_test.shape}")
    print(f"Formato codificado X_train: {X_train_encoded.shape}")
    print(f"Formato codificado X_test: {X_test_encoded.shape}")
    print(f"Encoder salvo em: {caminho_encoder}")

    print("\nPrimeiras colunas geradas pelo encoding:")
    for coluna in X_train_encoded.columns[:20]:
        print(f"- {coluna}")

    return X_train_encoded, X_test_encoded, encoder