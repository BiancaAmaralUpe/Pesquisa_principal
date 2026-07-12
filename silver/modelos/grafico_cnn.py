# ======================================================================================
# grafico_cnn.py
# ======================================================================================
# Responsabilidade:
# - Gerar os gráficos do histórico de treinamento da CNN
# - Gerar a matriz de confusão absoluta
# - Gerar a matriz de confusão normalizada
# ======================================================================================

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix


def gerar_grafico_historico_cnn(
    historico: dict[str, object],
    caminho_saida: str | Path,
) -> None:
    """
    Gera os gráficos de loss e accuracy da CNN ao longo das épocas.

    Arquivos gerados:
    - historico_cnn_loss.png
    - historico_cnn_accuracy.png
    """

    caminho_saida = Path(
        caminho_saida
    )

    caminho_saida.mkdir(
        parents=True,
        exist_ok=True,
    )

    epocas = historico.get(
        "epocas",
        [],
    )

    train_loss = historico.get(
        "train_loss",
        [],
    )

    validation_loss = historico.get(
        "validation_loss",
        [],
    )

    train_accuracy = historico.get(
        "train_accuracy",
        [],
    )

    validation_accuracy = historico.get(
        "validation_accuracy",
        [],
    )

    if not epocas:
        print(
            "\nHistórico da CNN vazio. "
            "Os gráficos não serão gerados."
        )
        return

    quantidades = {
        len(epocas),
        len(train_loss),
        len(validation_loss),
        len(train_accuracy),
        len(validation_accuracy),
    }

    if len(quantidades) != 1:
        raise ValueError(
            "As listas do histórico da CNN possuem "
            "quantidades diferentes de épocas."
        )

    print("\n" + "=" * 80)
    print("GERAÇÃO DOS GRÁFICOS DE TREINAMENTO - CNN")
    print("=" * 80)

    # ==================================================================================
    # Gráfico de loss
    # ==================================================================================
    caminho_loss = (
        caminho_saida
        / "historico_cnn_loss.png"
    )

    figura_loss, eixo_loss = plt.subplots(
        figsize=(10, 6)
    )

    eixo_loss.plot(
        epocas,
        train_loss,
        marker="o",
        label="Treino",
    )

    eixo_loss.plot(
        epocas,
        validation_loss,
        marker="o",
        label="Validação",
    )

    eixo_loss.set_title(
        "Histórico de Loss - CNN 1D"
    )

    eixo_loss.set_xlabel(
        "Época"
    )

    eixo_loss.set_ylabel(
        "Loss"
    )

    eixo_loss.grid(
        alpha=0.30
    )

    eixo_loss.legend()

    figura_loss.tight_layout()

    figura_loss.savefig(
        caminho_loss,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figura_loss
    )

    # ==================================================================================
    # Gráfico de accuracy
    # ==================================================================================
    caminho_accuracy = (
        caminho_saida
        / "historico_cnn_accuracy.png"
    )

    figura_accuracy, eixo_accuracy = plt.subplots(
        figsize=(10, 6)
    )

    eixo_accuracy.plot(
        epocas,
        train_accuracy,
        marker="o",
        label="Treino",
    )

    eixo_accuracy.plot(
        epocas,
        validation_accuracy,
        marker="o",
        label="Validação",
    )

    eixo_accuracy.set_title(
        "Histórico de Accuracy - CNN 1D"
    )

    eixo_accuracy.set_xlabel(
        "Época"
    )

    eixo_accuracy.set_ylabel(
        "Accuracy"
    )

    eixo_accuracy.grid(
        alpha=0.30
    )

    eixo_accuracy.legend()

    figura_accuracy.tight_layout()

    figura_accuracy.savefig(
        caminho_accuracy,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figura_accuracy
    )

    print("\nGráficos salvos com sucesso em:")

    print(
        f"- {caminho_loss}"
    )

    print(
        f"- {caminho_accuracy}"
    )


def gerar_matrizes_confusao_cnn(
    y_real,
    y_predito,
    classes_target: list[str],
    caminho_saida: str | Path,
) -> None:
    """
    Gera a matriz de confusão absoluta e normalizada da CNN.
    """

    caminho_saida = Path(
        caminho_saida
    )

    caminho_saida.mkdir(
        parents=True,
        exist_ok=True,
    )

    y_real_numpy = np.asarray(
        y_real,
        dtype=np.int64,
    ).reshape(-1)

    y_predito_numpy = np.asarray(
        y_predito,
        dtype=np.int64,
    ).reshape(-1)

    if len(y_real_numpy) != len(y_predito_numpy):
        raise ValueError(
            "y_real e y_predito possuem quantidades "
            "diferentes de registros."
        )

    if len(classes_target) < 2:
        raise ValueError(
            "classes_target deve possuir pelo menos duas classes."
        )

    labels_numericos = list(
        range(len(classes_target))
    )

    print("\n" + "=" * 80)
    print("GERAÇÃO DAS MATRIZES DE CONFUSÃO - CNN")
    print("=" * 80)

    print("\nMapeamento utilizado:")

    for indice, nome_classe in enumerate(
        classes_target
    ):
        print(
            f"- {indice}: {nome_classe}"
        )

    # ==================================================================================
    # Matriz de confusão absoluta
    # ==================================================================================
    matriz_absoluta = confusion_matrix(
        y_true=y_real_numpy,
        y_pred=y_predito_numpy,
        labels=labels_numericos,
    )

    caminho_matriz_absoluta = (
        caminho_saida
        / "matriz_confusao_cnn_absoluta.png"
    )

    figura_absoluta, eixo_absoluto = plt.subplots(
        figsize=(10, 8)
    )

    visualizacao_absoluta = ConfusionMatrixDisplay(
        confusion_matrix=matriz_absoluta,
        display_labels=classes_target,
    )

    visualizacao_absoluta.plot(
        ax=eixo_absoluto,
        values_format="d",
        xticks_rotation=45,
        colorbar=False,
    )

    eixo_absoluto.set_title(
        "Matriz de Confusão Absoluta - CNN 1D"
    )

    figura_absoluta.tight_layout()

    figura_absoluta.savefig(
        caminho_matriz_absoluta,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figura_absoluta
    )

    # ==================================================================================
    # Matriz de confusão normalizada
    # ==================================================================================
    matriz_normalizada = confusion_matrix(
        y_true=y_real_numpy,
        y_pred=y_predito_numpy,
        labels=labels_numericos,
        normalize="true",
    )

    caminho_matriz_normalizada = (
        caminho_saida
        / "matriz_confusao_cnn_normalizada.png"
    )

    figura_normalizada, eixo_normalizado = plt.subplots(
        figsize=(10, 8)
    )

    visualizacao_normalizada = ConfusionMatrixDisplay(
        confusion_matrix=matriz_normalizada,
        display_labels=classes_target,
    )

    visualizacao_normalizada.plot(
        ax=eixo_normalizado,
        values_format=".2f",
        xticks_rotation=45,
        colorbar=False,
    )

    eixo_normalizado.set_title(
        "Matriz de Confusão Normalizada - CNN 1D"
    )

    figura_normalizada.tight_layout()

    figura_normalizada.savefig(
        caminho_matriz_normalizada,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figura_normalizada
    )

    print("\nMatrizes salvas com sucesso em:")

    print(
        f"- {caminho_matriz_absoluta}"
    )

    print(
        f"- {caminho_matriz_normalizada}"
    )