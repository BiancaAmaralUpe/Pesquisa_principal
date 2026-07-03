from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier

# ======================================================================
# Registro no arquivo .txt
# ======================================================================
def registrar_analise_overfitting_arvore_decisao(
    caminho_log: Path,
    dataframe_resultados: pd.DataFrame,
    min_samples_leaf: int,
    random_state: int,
) -> None:
    """
    Registra no arquivo .txt os resultados da análise de overfitting
    da Árvore de Decisão para diferentes valores de max_depth.
    """

    caminho_log.parent.mkdir(parents=True, exist_ok=True)

    data_execucao = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(caminho_log, "a", encoding="utf-8") as arquivo:
        arquivo.write("\n")
        arquivo.write("=" * 80)
        arquivo.write("\n")
        arquivo.write("ANÁLISE DE OVERFITTING - ÁRVORE DE DECISÃO\n")
        arquivo.write("=" * 80)
        arquivo.write("\n")

        arquivo.write(f"Data/hora da execução: {data_execucao}\n\n")

        arquivo.write("Parâmetros fixos utilizados:\n")
        arquivo.write(f"- MIN_SAMPLES_LEAF_ARVORE_DECISAO: {min_samples_leaf}\n")
        arquivo.write(f"- RANDOM_STATE: {random_state}\n")
        arquivo.write("- criterion: gini\n")
        arquivo.write("- class_weight: balanced\n\n")

        arquivo.write("Resultados por profundidade testada:\n\n")

        for _, linha in dataframe_resultados.iterrows():
            arquivo.write("-" * 80)
            arquivo.write("\n")

            arquivo.write(
                f"MAX_DEPTH_ARVORE_DECISAO: {int(linha['max_depth'])}\n"
            )

            arquivo.write(
                f"MIN_SAMPLES_LEAF_ARVORE_DECISAO: {min_samples_leaf}\n"
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

# ======================================================================
# Geração dos gráficos
# ======================================================================
def gerar_grafico_overfitting_arvore_decisao(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    caminho_saida: str,
    caminho_log: Path | None = None,
    min_samples_leaf: int = 100,
    random_state: int = 42,
) -> None:
    """
    Gera gráfico para analisar overfitting da Árvore de Decisão
    comparando desempenho de treino e teste em diferentes profundidades.
    """

    Path(caminho_saida).mkdir(parents=True, exist_ok=True)

    profundidades = [2, 4, 6, 8, 10, 12, 15, 20, 25, 30]

    resultados = {
        "max_depth": [],
        "accuracy_treino": [],
        "accuracy_teste": [],
        "macro_f1_treino": [],
        "macro_f1_teste": [],
    }

    print("\n" + "=" * 80)
    print("ANÁLISE DE OVERFITTING - ÁRVORE DE DECISÃO")
    print("=" * 80)

    for profundidade in profundidades:
        # ======================================================================
        # Treinamento do modelo
        # Aqui tive que deixar os valores que eram fixos, para valores variaveis, 
        # pois é para ficar registrando no .txt. Assim tenho controle dos parametros que eu estou usando.
        # ======================================================================
        modelo = DecisionTreeClassifier(
            max_depth=profundidade,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced",
            random_state=random_state,
        )

        modelo.fit(X_train, y_train)

        pred_train = modelo.predict(X_train)
        pred_test = modelo.predict(X_test)

        acc_train = accuracy_score(y_train, pred_train)
        acc_test = accuracy_score(y_test, pred_test)

        f1_train = f1_score(y_train, pred_train, average="macro")
        f1_test = f1_score(y_test, pred_test, average="macro")

        resultados["max_depth"].append(profundidade)
        resultados["accuracy_treino"].append(acc_train)
        resultados["accuracy_teste"].append(acc_test)
        resultados["macro_f1_treino"].append(f1_train)
        resultados["macro_f1_teste"].append(f1_test)

        print(
            f"max_depth={profundidade} | "
            f"acc_treino={acc_train:.4f} | acc_teste={acc_test:.4f} | "
            f"f1_treino={f1_train:.4f} | f1_teste={f1_test:.4f}"
        )

    df_resultados = pd.DataFrame(resultados)

    # ======================================================================
    # Registro no arquivo .txt
    # ======================================================================
    if caminho_log is not None:
        registrar_analise_overfitting_arvore_decisao(
            caminho_log=caminho_log,
            dataframe_resultados=df_resultados,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )

    # ======================================================================
    # Geração dos gráficos
    # ======================================================================
    plt.figure(figsize=(10, 6))
    plt.plot(
        df_resultados["max_depth"],
        df_resultados["accuracy_treino"],
        marker="o",
        label="Treino",
    )
    plt.plot(
        df_resultados["max_depth"],
        df_resultados["accuracy_teste"],
        marker="o",
        label="Teste",
    )
    plt.title("Overfitting - Árvore de Decisão (Accuracy)")
    plt.xlabel("max_depth")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        Path(caminho_saida) / "grafico_overfitting_arvore_accuracy.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # ======================================================================
    # Geração do gráfico Macro F1
    # ======================================================================
    plt.figure(figsize=(10, 6))
    plt.plot(
        df_resultados["max_depth"],
        df_resultados["macro_f1_treino"],
        marker="o",
        label="Treino",
    )
    plt.plot(
        df_resultados["max_depth"],
        df_resultados["macro_f1_teste"],
        marker="o",
        label="Teste",
    )
    plt.title("Overfitting - Árvore de Decisão (Macro F1)")
    plt.xlabel("max_depth")
    plt.ylabel("Macro F1")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        Path(caminho_saida) / "grafico_overfitting_arvore_macro_f1.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print("\nGráficos salvos com sucesso em:")
    print(f"- {Path(caminho_saida) / 'grafico_overfitting_arvore_accuracy.png'}")
    print(f"- {Path(caminho_saida) / 'grafico_overfitting_arvore_macro_f1.png'}")