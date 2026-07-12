# ======================================================================================
# grafico_mlp.py
# ======================================================================================
# Responsabilidade:
# - Avaliar possível overfitting do modelo MLP
# - Testar diferentes arquiteturas de camadas ocultas
# - Gerar gráficos de Accuracy e Macro F1-score
# - Registrar os resultados em arquivo .txt
# ======================================================================================

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier


def converter_arquitetura_para_texto(
    arquitetura: tuple[int, ...],
) -> str:
    """
    Converte a arquitetura da MLP para um texto utilizado nos gráficos.

    Exemplos:
    (32,) -> 32
    (64, 32) -> 64x32
    (128, 64, 32) -> 128x64x32
    """

    return "x".join(
        str(quantidade_neuronios)
        for quantidade_neuronios in arquitetura
    )


def registrar_analise_overfitting_mlp(
    caminho_log: Path,
    dataframe_resultados: pd.DataFrame,
    activation: str,
    solver: str,
    alpha: float,
    batch_size: int,
    learning_rate_init: float,
    max_iter: int,
    early_stopping: bool,
    validation_fraction: float,
    n_iter_no_change: int,
    tol: float,
    random_state: int,
) -> None:
    """
    Registra no arquivo .txt os resultados da análise de overfitting
    da MLP para diferentes arquiteturas.
    """
    
    caminho_log.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data_execucao = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    with open(
        caminho_log,
        "a",
        encoding="utf-8",
    ) as arquivo:
        arquivo.write("\n")
        arquivo.write("=" * 80)
        arquivo.write("\n")
        arquivo.write("ANÁLISE DE OVERFITTING - MLP\n")
        arquivo.write("=" * 80)
        arquivo.write("\n")

        arquivo.write(
            f"Data/hora da execução: {data_execucao}\n\n"
        )

        arquivo.write("Parâmetros fixos utilizados:\n")
        arquivo.write(f"- activation: {activation}\n")
        arquivo.write(f"- solver: {solver}\n")
        arquivo.write(f"- alpha: {alpha}\n")
        arquivo.write(f"- batch_size: {batch_size}\n")
        arquivo.write(
            f"- learning_rate_init: {learning_rate_init}\n"
        )
        arquivo.write(f"- max_iter: {max_iter}\n")
        arquivo.write(
            f"- early_stopping: {early_stopping}\n"
        )
        arquivo.write(
            f"- validation_fraction: {validation_fraction}\n"
        )
        arquivo.write(
            f"- n_iter_no_change: {n_iter_no_change}\n"
        )
        arquivo.write(f"- random_state: {random_state}\n\n")

        arquivo.write(
            "Resultados por arquitetura testada:\n\n"
        )

        for _, linha in dataframe_resultados.iterrows():
            arquivo.write("-" * 80)
            arquivo.write("\n")

            arquivo.write(
                f"Arquitetura: {linha['arquitetura']}\n"
            )

            arquivo.write(
                f"Quantidade total de neurônios ocultos: "
                f"{int(linha['total_neuronios'])}\n"
            )

            arquivo.write(
                f"Quantidade de iterações executadas: "
                f"{int(linha['n_iteracoes'])}\n"
            )

            arquivo.write(
                f"Loss final: {linha['loss_final']:.6f}\n"
            )

            arquivo.write("\nMétricas obtidas:\n")

            arquivo.write(
                f"- Accuracy treino: "
                f"{linha['accuracy_treino']:.4f}\n"
            )

            arquivo.write(
                f"- Accuracy teste: "
                f"{linha['accuracy_teste']:.4f}\n"
            )

            arquivo.write(
                f"- GAP Accuracy: "
                f"{linha['gap_accuracy']:.4f}\n"
            )

            arquivo.write(
                f"- Macro F1-score treino: "
                f"{linha['macro_f1_treino']:.4f}\n"
            )

            arquivo.write(
                f"- Macro F1-score teste: "
                f"{linha['macro_f1_teste']:.4f}\n"
            )

            arquivo.write(
                f"- GAP Macro F1-score: "
                f"{linha['gap_macro_f1']:.4f}\n"
            )

            arquivo.write("\n")

        arquivo.write("=" * 80)
        arquivo.write("\n")


def gerar_grafico_overfitting_mlp(
    X_train_modelo: pd.DataFrame,
    X_train_avaliacao: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train_modelo: pd.Series,
    y_train_avaliacao: pd.Series,
    y_test: pd.Series,
    caminho_saida: str,
    caminho_log: Path | None = None,
    arquiteturas: list[tuple[int, ...]] | None = None,
    activation: str = "relu",
    solver: str = "adam",
    alpha: float = 0.0001,
    batch_size: int = 2048,
    learning_rate_init: float = 0.001,
    max_iter: int = 50,
    early_stopping: bool = True,
    validation_fraction: float = 0.1,
    n_iter_no_change: int = 5,
    tol: float = 0.001,
    random_state: int = 42,
) -> None:
    """
    Gera gráficos para analisar possível overfitting da MLP.

    Diferentes arquiteturas de camadas ocultas são treinadas,
    permitindo comparar o desempenho de treino e teste.

    Cada arquitetura deve ser informada como uma tupla.

    Exemplos:
    - (32,)
    - (64,)
    - (64, 32)
    - (128, 64)
    """

    pasta_saida = Path(caminho_saida)

    pasta_saida.mkdir(
        parents=True,
        exist_ok=True,
    )

    if arquiteturas is None:
        arquiteturas = [
            (32,),
            (64,),
            (64, 32),
            (128, 64),
        ]

    if not arquiteturas:
        raise ValueError(
            "A lista de arquiteturas da MLP não pode estar vazia."
        )

    # ==============================================================
    # validação dos targets
    # ==============================================================
    if pd.Series(y_train_modelo).isna().any():
        raise ValueError(
            "O target balanceado utilizado no treinamento possui valores nulos."
        )

    if pd.Series(y_train_avaliacao).isna().any():
        raise ValueError(
            "O target original utilizado na avaliação possui valores nulos."
        )

    if pd.Series(y_test).isna().any():
        raise ValueError(
            "O target de teste possui valores nulos."
        )

    resultados = {
        "arquitetura": [],
        "total_neuronios": [],
        "accuracy_treino": [],
        "accuracy_teste": [],
        "gap_accuracy": [],
        "macro_f1_treino": [],
        "macro_f1_teste": [],
        "gap_macro_f1": [],
        "n_iteracoes": [],
        "loss_final": [],
    }

    print("\n" + "=" * 80)
    print("ANÁLISE DE OVERFITTING - MLP")
    print("=" * 80)

    print("\nArquiteturas que serão testadas:")

    for arquitetura in arquiteturas:
        print(
            f"- {converter_arquitetura_para_texto(arquitetura)}"
        )

    for numero_teste, arquitetura in enumerate(
        arquiteturas,
        start=1,
    ):
        arquitetura_texto = converter_arquitetura_para_texto(
            arquitetura
        )

        print("\n" + "-" * 80)
        print(
            f"TESTE {numero_teste}/{len(arquiteturas)} "
            f"- ARQUITETURA {arquitetura_texto}"
        )
        print("-" * 80)

        modelo = MLPClassifier(
            hidden_layer_sizes=arquitetura,
            activation=activation,
            solver=solver,
            alpha=alpha,
            batch_size=batch_size,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            tol=tol,
            random_state=random_state,
            verbose=False,
        )

        modelo.fit(
            X_train_modelo,
            y_train_modelo,
        )

        y_pred_train = modelo.predict(
            X_train_avaliacao,
        )

        y_pred_test = modelo.predict(
            X_test,
        )

        accuracy_treino = accuracy_score(
            y_train_avaliacao,
            y_pred_train,
        )

        accuracy_teste = accuracy_score(
            y_test,
            y_pred_test,
        )

        macro_f1_treino = f1_score(
            y_train_avaliacao,
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

        gap_accuracy = (
            accuracy_treino
            - accuracy_teste
        )

        gap_macro_f1 = (
            macro_f1_treino
            - macro_f1_teste
        )

        total_neuronios = sum(
            arquitetura
        )

        n_iteracoes = modelo.n_iter_

        loss_final = modelo.loss_

        resultados["arquitetura"].append(
            arquitetura_texto
        )

        resultados["total_neuronios"].append(
            total_neuronios
        )

        resultados["accuracy_treino"].append(
            accuracy_treino
        )

        resultados["accuracy_teste"].append(
            accuracy_teste
        )

        resultados["gap_accuracy"].append(
            gap_accuracy
        )

        resultados["macro_f1_treino"].append(
            macro_f1_treino
        )

        resultados["macro_f1_teste"].append(
            macro_f1_teste
        )

        resultados["gap_macro_f1"].append(
            gap_macro_f1
        )

        resultados["n_iteracoes"].append(
            n_iteracoes
        )

        resultados["loss_final"].append(
            loss_final
        )

        print(
            f"Arquitetura={arquitetura_texto} | "
            f"acc_treino={accuracy_treino:.4f} | "
            f"acc_teste={accuracy_teste:.4f} | "
            f"gap_acc={gap_accuracy:.4f} | "
            f"f1_treino={macro_f1_treino:.4f} | "
            f"f1_teste={macro_f1_teste:.4f} | "
            f"gap_f1={gap_macro_f1:.4f} | "
            f"iterações={n_iteracoes}"
        )

    dataframe_resultados = pd.DataFrame(
        resultados
    )

    caminho_csv = (
        pasta_saida
        / "resultados_overfitting_mlp.csv"
    )

    dataframe_resultados.to_csv(
        caminho_csv,
        index=False,
        encoding="utf-8-sig",
    )

    caminho_accuracy = (
        pasta_saida
        / "grafico_overfitting_mlp_accuracy.png"
    )

    caminho_macro_f1 = (
        pasta_saida
        / "grafico_overfitting_mlp_macro_f1.png"
    )

    # ==================================================================================
    # Gráfico de Accuracy
    # ==================================================================================

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        dataframe_resultados["arquitetura"],
        dataframe_resultados["accuracy_treino"],
        marker="o",
        label="Treino",
    )

    plt.plot(
        dataframe_resultados["arquitetura"],
        dataframe_resultados["accuracy_teste"],
        marker="o",
        label="Teste",
    )

    plt.title(
        "Overfitting - MLP (Accuracy)"
    )

    plt.xlabel(
        "Arquitetura das camadas ocultas"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        caminho_accuracy,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # ==================================================================================
    # Gráfico de Macro F1-score
    # ==================================================================================

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        dataframe_resultados["arquitetura"],
        dataframe_resultados["macro_f1_treino"],
        marker="o",
        label="Treino",
    )

    plt.plot(
        dataframe_resultados["arquitetura"],
        dataframe_resultados["macro_f1_teste"],
        marker="o",
        label="Teste",
    )

    plt.title(
        "Overfitting - MLP (Macro F1-score)"
    )

    plt.xlabel(
        "Arquitetura das camadas ocultas"
    )

    plt.ylabel(
        "Macro F1-score"
    )

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
        registrar_analise_overfitting_mlp(
            caminho_log=caminho_log,
            dataframe_resultados=dataframe_resultados,
            activation=activation,
            solver=solver,
            alpha=alpha,
            batch_size=batch_size,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            random_state=random_state,
            tol=tol,
        )

    print("\nResultados da análise de overfitting:")

    print(
        dataframe_resultados.to_string(
            index=False
        )
    )

    print("\nArquivos salvos com sucesso em:")
    print(f"- {caminho_accuracy}")
    print(f"- {caminho_macro_f1}")
    print(f"- {caminho_csv}")