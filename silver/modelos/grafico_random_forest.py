# ======================================================================================
# grafico_random_forest.py
# ======================================================================================
# Responsabilidade:
# - Avaliar possível overfitting da Random Forest
# - Testar diferentes valores de max_depth
# - Gerar gráficos de Accuracy e Macro F1-score
# - Registrar os resultados em arquivo .txt
# ======================================================================================

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


def registrar_analise_overfitting_random_forest(
    caminho_log: Path,
    dataframe_resultados: pd.DataFrame,
    n_estimators: int,
    min_samples_leaf: int,
    max_features: str,
    random_state: int,
    n_jobs: int,
) -> None:
    """
    Registra no arquivo .txt os resultados da análise de overfitting
    da Random Forest para diferentes valores de max_depth.
    """

    caminho_log.parent.mkdir(parents=True, exist_ok=True)

    data_execucao = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(caminho_log, "a", encoding="utf-8") as arquivo:
        arquivo.write("\n")
        arquivo.write("=" * 80)
        arquivo.write("\n")
        arquivo.write("ANÁLISE DE OVERFITTING - RANDOM FOREST\n")
        arquivo.write("=" * 80)
        arquivo.write("\n")

        arquivo.write(f"Data/hora da execução: {data_execucao}\n\n")

        arquivo.write("Parâmetros fixos utilizados:\n")
        arquivo.write(f"- N_ESTIMATORS_RANDOM_FOREST: {n_estimators}\n")
        arquivo.write(f"- MIN_SAMPLES_LEAF_RANDOM_FOREST: {min_samples_leaf}\n")
        arquivo.write(f"- MAX_FEATURES_RANDOM_FOREST: {max_features}\n")
        arquivo.write(f"- RANDOM_STATE: {random_state}\n")
        arquivo.write(f"- N_JOBS_RANDOM_FOREST: {n_jobs}\n")
        arquivo.write("- criterion: gini\n")
        arquivo.write("- class_weight: balanced_subsample\n\n")

        arquivo.write("Resultados por profundidade testada:\n\n")

        for _, linha in dataframe_resultados.iterrows():
            arquivo.write("-" * 80)
            arquivo.write("\n")

            arquivo.write(
                f"MAX_DEPTH_RANDOM_FOREST: {int(linha['max_depth'])}\n"
            )

            arquivo.write(
                f"MIN_SAMPLES_LEAF_RANDOM_FOREST: {min_samples_leaf}\n"
            )

            arquivo.write("\nMétricas obtidas:\n")
            arquivo.write(
                f"- Accuracy treino: {linha['accuracy_treino']:.4f}\n"
            )
            arquivo.write(
                f"- Accuracy teste: {linha['accuracy_teste']:.4f}\n"
            )
            arquivo.write(
                f"- Macro F1-score treino: {linha['macro_f1_treino']:.4f}\n"
            )
            arquivo.write(
                f"- Macro F1-score teste: {linha['macro_f1_teste']:.4f}\n"
            )

            arquivo.write("\n")

        arquivo.write("=" * 80)
        arquivo.write("\n")


def gerar_grafico_overfitting_random_forest(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    caminho_saida: str,
    caminho_log: Path | None = None,
    n_estimators: int = 100,
    min_samples_leaf: int = 100,
    max_features: str = "sqrt",
    random_state: int = 42,
    n_jobs: int = -1,
) -> None:
    """
    Gera gráficos para analisar possível overfitting da Random Forest
    comparando desempenho de treino e teste em diferentes profundidades.
    """

    Path(caminho_saida).mkdir(parents=True, exist_ok=True)

    profundidades = [
        5,
        10,
        15,
        20,
        25,
        30,
    ]

    resultados = {
        "max_depth": [],
        "accuracy_treino": [],
        "accuracy_teste": [],
        "macro_f1_treino": [],
        "macro_f1_teste": [],
    }

    print("\n" + "=" * 80)
    print("ANÁLISE DE OVERFITTING - RANDOM FOREST")
    print("=" * 80)

    for profundidade in profundidades:
        modelo = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion="gini",
            max_depth=profundidade,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=n_jobs,
        )

        modelo.fit(X_train, y_train)

        y_pred_train = modelo.predict(X_train)
        y_pred_test = modelo.predict(X_test)

        accuracy_treino = accuracy_score(y_train, y_pred_train)
        accuracy_teste = accuracy_score(y_test, y_pred_test)

        macro_f1_treino = f1_score(
            y_train,
            y_pred_train,
            average="macro",
            zero_division=0,
        )

        macro_f1_teste = f1_score(
            y_test,
            y_pred_test,
            average="macro",
            zero_division=0,
        )

        resultados["max_depth"].append(profundidade)
        resultados["accuracy_treino"].append(accuracy_treino)
        resultados["accuracy_teste"].append(accuracy_teste)
        resultados["macro_f1_treino"].append(macro_f1_treino)
        resultados["macro_f1_teste"].append(macro_f1_teste)

        print(
            f"max_depth={profundidade} | "
            f"acc_treino={accuracy_treino:.4f} | "
            f"acc_teste={accuracy_teste:.4f} | "
            f"f1_treino={macro_f1_treino:.4f} | "
            f"f1_teste={macro_f1_teste:.4f}"
        )

    dataframe_resultados = pd.DataFrame(resultados)

    caminho_accuracy = (
        Path(caminho_saida) / "grafico_overfitting_random_forest_accuracy.png"
    )

    caminho_macro_f1 = (
        Path(caminho_saida) / "grafico_overfitting_random_forest_macro_f1.png"
    )

    plt.figure(figsize=(10, 6))
    plt.plot(
        dataframe_resultados["max_depth"],
        dataframe_resultados["accuracy_treino"],
        marker="o",
        label="Treino",
    )
    plt.plot(
        dataframe_resultados["max_depth"],
        dataframe_resultados["accuracy_teste"],
        marker="o",
        label="Teste",
    )
    plt.title("Overfitting - Random Forest (Accuracy)")
    plt.xlabel("max_depth")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        caminho_accuracy,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(
        dataframe_resultados["max_depth"],
        dataframe_resultados["macro_f1_treino"],
        marker="o",
        label="Treino",
    )
    plt.plot(
        dataframe_resultados["max_depth"],
        dataframe_resultados["macro_f1_teste"],
        marker="o",
        label="Teste",
    )
    plt.title("Overfitting - Random Forest (Macro F1)")
    plt.xlabel("max_depth")
    plt.ylabel("Macro F1")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        caminho_macro_f1,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    if caminho_log is not None:
        registrar_analise_overfitting_random_forest(
            caminho_log=caminho_log,
            dataframe_resultados=dataframe_resultados,
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    print("\nGráficos salvos com sucesso em:")
    print(f"- {caminho_accuracy}")
    print(f"- {caminho_macro_f1}")