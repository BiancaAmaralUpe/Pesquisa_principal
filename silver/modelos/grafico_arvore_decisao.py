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
    parada_antecipada_ativada: bool,
    melhor_max_depth: int | None,
    melhor_macro_f1_teste: float,
    quantidade_sem_melhoria: int,
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

        if parada_antecipada_ativada:
            arquivo.write("\n")
            arquivo.write("Parada antecipada:\n")
            arquivo.write("- Status: ativada\n")
            arquivo.write(f"- Melhor max_depth encontrado: {melhor_max_depth}\n")
            arquivo.write(
                f"- Melhor Macro F1-score teste: {melhor_macro_f1_teste:.4f}\n"
            )
            arquivo.write(
                f"- Quantidade final sem melhoria relevante: "
                f"{quantidade_sem_melhoria}\n"
            )

        arquivo.write("=" * 80)
        arquivo.write("\n")

# ======================================================================
# Geração dos gráficos
# ======================================================================
def gerar_grafico_overfitting_arvore_decisao(
    X_train,
    X_test,
    y_train,
    y_test,
    caminho_saida: str,
    caminho_log,
    min_samples_leaf: int,
    random_state: int,
    ativar_parada_antecipada: bool,
    tolerancia_melhoria: float,
    paciencia: int,
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
    melhor_macro_f1_teste = -1
    quantidade_sem_melhoria = 0
    melhor_max_depth = None

    print("\n" + "=" * 80)
    print("ANÁLISE DE OVERFITTING - ÁRVORE DE DECISÃO")
    print("=" * 80)
    
    for profundidade in profundidades:
        modelo = DecisionTreeClassifier(
            criterion="gini",
            max_depth=profundidade,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            class_weight="balanced",
        )

        modelo.fit(
            X_train,
            y_train,
        )

        pred_train = modelo.predict(X_train)
        pred_test = modelo.predict(X_test)

        acc_train = accuracy_score(
            y_train,
            pred_train,
        )

        acc_test = accuracy_score(
            y_test,
            pred_test,
        )

        f1_train = f1_score(
            y_train,
            pred_train,
            average="macro",
            zero_division=0,
        )

        f1_test = f1_score(
            y_test,
            pred_test,
            average="macro",
            zero_division=0,
        )

        resultados["max_depth"].append(profundidade)
        resultados["accuracy_treino"].append(acc_train)
        resultados["accuracy_teste"].append(acc_test)
        resultados["macro_f1_treino"].append(f1_train)
        resultados["macro_f1_teste"].append(f1_test)

        print(
            f"max_depth={profundidade} | "
            f"acc_treino={acc_train:.4f} | "
            f"acc_teste={acc_test:.4f} | "
            f"f1_treino={f1_train:.4f} | "
            f"f1_teste={f1_test:.4f}"
        )

        melhoria = f1_test - melhor_macro_f1_teste

        if melhoria > tolerancia_melhoria:
            melhor_macro_f1_teste = f1_test
            melhor_max_depth = profundidade
            quantidade_sem_melhoria = 0
        else:
            quantidade_sem_melhoria += 1

        if ativar_parada_antecipada and quantidade_sem_melhoria >= paciencia:
            print("\nParada antecipada ativada na análise de profundidade.")
            print(f"Melhor max_depth encontrado: {melhor_max_depth}")
            print(f"Melhor Macro F1-score teste: {melhor_macro_f1_teste:.4f}")
            print(
                f"Sem melhoria relevante nas últimas "
                f"{quantidade_sem_melhoria} profundidades testadas."
            )
            break

    df_resultados = pd.DataFrame(resultados)

    if caminho_log is not None:
        registrar_analise_overfitting_arvore_decisao(
            caminho_log=caminho_log,
            dataframe_resultados=df_resultados,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            parada_antecipada_ativada=ativar_parada_antecipada,
            melhor_max_depth=melhor_max_depth,
            melhor_macro_f1_teste=melhor_macro_f1_teste,
            quantidade_sem_melhoria=quantidade_sem_melhoria,
        )

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