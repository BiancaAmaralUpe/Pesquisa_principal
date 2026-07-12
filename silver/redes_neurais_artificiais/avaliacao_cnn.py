# ======================================================================================
# avaliacao_cnn.py
# ======================================================================================
# Responsabilidade:
# - Gerar previsões da CNN
# - Avaliar treino original e teste
# - Calcular métricas gerais
# - Calcular métricas detalhadas por classe
# - Calcular verdadeiros positivos, falsos positivos e falsos negativos
# ======================================================================================

import numpy as np
import pandas as pd
import torch  # pyrefly: ignore [missing-import]
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from torch import nn  # pyrefly: ignore [missing-import]
from torch.utils.data import DataLoader  # pyrefly: ignore [missing-import]


def definir_dispositivo_avaliacao_cnn(
    dispositivo: str | None = None,
) -> torch.device:
    """
    Define o dispositivo utilizado na avaliação.
    """

    if dispositivo is not None:
        return torch.device(
            dispositivo
        )

    if torch.cuda.is_available():
        return torch.device(
            "cuda"
        )

    return torch.device(
        "cpu"
    )


def converter_target_numpy(
    y: pd.Series | np.ndarray | list,
) -> np.ndarray:
    """
    Converte o target para um array NumPy unidimensional.
    """

    if isinstance(y, pd.Series):
        y_numpy = y.to_numpy(
            dtype=np.int64,
            copy=True,
        )
    else:
        y_numpy = np.asarray(
            y,
            dtype=np.int64,
        )

    y_numpy = y_numpy.reshape(
        -1
    )

    return y_numpy


def obter_predicoes_cnn(
    modelo: nn.Module,
    data_loader: DataLoader,
    dispositivo: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Executa as previsões da CNN.

    Retorna:
    - valores reais;
    - valores preditos.
    """

    modelo.eval()

    valores_reais = []
    valores_preditos = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(
                dispositivo,
                dtype=torch.float32,
            )

            logits = modelo(
                X_batch
            )

            predicoes = torch.argmax(
                logits,
                dim=1,
            )

            valores_reais.extend(
                y_batch.cpu().numpy().tolist()
            )

            valores_preditos.extend(
                predicoes.cpu().numpy().tolist()
            )

    return (
        np.asarray(
            valores_reais,
            dtype=np.int64,
        ),
        np.asarray(
            valores_preditos,
            dtype=np.int64,
        ),
    )


def calcular_metricas_por_classe_cnn(
    y_real: np.ndarray,
    y_predito: np.ndarray,
    classes_target: list[str],
) -> dict[str, dict[str, int | float]]:
    """
    Calcula as métricas detalhadas de cada classe.
    """

    labels_numericos = list(
        range(len(classes_target))
    )

    matriz = confusion_matrix(
        y_true=y_real,
        y_pred=y_predito,
        labels=labels_numericos,
    )

    precisoes, recalls, f1_scores, suportes = (
        precision_recall_fscore_support(
            y_true=y_real,
            y_pred=y_predito,
            labels=labels_numericos,
            zero_division=0,
        )
    )

    total_registros = int(
        matriz.sum()
    )

    metricas_por_classe = {}

    print("\n" + "=" * 80)
    print("MÉTRICAS DETALHADAS POR CLASSE - CNN")
    print("=" * 80)

    for indice, nome_classe in enumerate(
        classes_target
    ):
        verdadeiros_positivos = int(
            matriz[indice, indice]
        )

        falsos_negativos = int(
            matriz[indice, :].sum()
            - verdadeiros_positivos
        )

        falsos_positivos = int(
            matriz[:, indice].sum()
            - verdadeiros_positivos
        )

        verdadeiros_negativos = int(
            total_registros
            - verdadeiros_positivos
            - falsos_negativos
            - falsos_positivos
        )

        metricas_por_classe[nome_classe] = {
            "precision": float(
                precisoes[indice]
            ),
            "recall": float(
                recalls[indice]
            ),
            "f1_score": float(
                f1_scores[indice]
            ),
            "support": int(
                suportes[indice]
            ),
            "verdadeiros_positivos": verdadeiros_positivos,
            "falsos_positivos": falsos_positivos,
            "falsos_negativos": falsos_negativos,
            "verdadeiros_negativos": verdadeiros_negativos,
        }

        print("\n" + "-" * 80)
        print(f"Classe: {nome_classe}")
        print("-" * 80)

        print(
            f"Precision: "
            f"{precisoes[indice]:.4f}"
        )

        print(
            f"Recall: "
            f"{recalls[indice]:.4f}"
        )

        print(
            f"F1-score: "
            f"{f1_scores[indice]:.4f}"
        )

        print(
            f"Support: "
            f"{int(suportes[indice])}"
        )

        print(
            "Verdadeiros positivos: "
            f"{verdadeiros_positivos}"
        )

        print(
            "Falsos positivos: "
            f"{falsos_positivos}"
        )

        print(
            "Falsos negativos: "
            f"{falsos_negativos}"
        )

        print(
            "Verdadeiros negativos: "
            f"{verdadeiros_negativos}"
        )

    return metricas_por_classe


def avaliar_cnn(
    modelo: nn.Module,
    train_evaluation_loader: DataLoader,
    test_loader: DataLoader,
    y_train_avaliacao: pd.Series | np.ndarray | list,
    y_test: pd.Series | np.ndarray | list,
    classes_target: list[str],
    dispositivo: str | None = None,
) -> tuple[dict[str, object], np.ndarray]:
    """
    Avalia a CNN no treino original e no conjunto de teste.

    O treino utilizado nesta função deve ser o treino original,
    não balanceado, para permitir comparação adequada com o teste.
    """

    if len(classes_target) < 2:
        raise ValueError(
            "classes_target deve possuir pelo menos duas classes."
        )

    dispositivo_torch = definir_dispositivo_avaliacao_cnn(
        dispositivo=dispositivo,
    )

    modelo = modelo.to(
        dispositivo_torch
    )

    print("\n" + "=" * 80)
    print("AVALIAÇÃO DO MODELO - CNN 1D")
    print("=" * 80)

    print(
        f"Dispositivo utilizado: "
        f"{dispositivo_torch}"
    )

    y_train_loader, y_pred_train = obter_predicoes_cnn(
        modelo=modelo,
        data_loader=train_evaluation_loader,
        dispositivo=dispositivo_torch,
    )

    y_test_loader, y_pred_test = obter_predicoes_cnn(
        modelo=modelo,
        data_loader=test_loader,
        dispositivo=dispositivo_torch,
    )

    y_train_referencia = converter_target_numpy(
        y_train_avaliacao
    )

    y_test_referencia = converter_target_numpy(
        y_test
    )

    if len(y_train_loader) != len(y_train_referencia):
        raise ValueError(
            "A quantidade de registros do treino de avaliação "
            "é diferente da quantidade presente no DataLoader. "
            f"Target: {len(y_train_referencia)} | "
            f"DataLoader: {len(y_train_loader)}"
        )

    if len(y_test_loader) != len(y_test_referencia):
        raise ValueError(
            "A quantidade de registros do teste "
            "é diferente da quantidade presente no DataLoader. "
            f"Target: {len(y_test_referencia)} | "
            f"DataLoader: {len(y_test_loader)}"
        )

    if not np.array_equal(
        y_train_loader,
        y_train_referencia,
    ):
        raise ValueError(
            "A ordem do target de treino é diferente da ordem "
            "dos registros do train_evaluation_loader. "
            "Verifique se o DataLoader de avaliação está com "
            "shuffle=False."
        )

    if not np.array_equal(
        y_test_loader,
        y_test_referencia,
    ):
        raise ValueError(
            "A ordem do target de teste é diferente da ordem "
            "dos registros do test_loader. "
            "Verifique se o DataLoader de teste está com "
            "shuffle=False."
        )

    accuracy_train = accuracy_score(
        y_train_referencia,
        y_pred_train,
    )

    accuracy_test = accuracy_score(
        y_test_referencia,
        y_pred_test,
    )

    balanced_accuracy_test = balanced_accuracy_score(
        y_test_referencia,
        y_pred_test,
    )

    macro_precision_test = precision_score(
        y_test_referencia,
        y_pred_test,
        average="macro",
        zero_division=0,
    )

    macro_recall_test = recall_score(
        y_test_referencia,
        y_pred_test,
        average="macro",
        zero_division=0,
    )

    macro_f1_test = f1_score(
        y_test_referencia,
        y_pred_test,
        average="macro",
        zero_division=0,
    )

    weighted_f1_test = f1_score(
        y_test_referencia,
        y_pred_test,
        average="weighted",
        zero_division=0,
    )

    metricas_por_classe = calcular_metricas_por_classe_cnn(
        y_real=y_test_referencia,
        y_predito=y_pred_test,
        classes_target=classes_target,
    )

    labels_numericos = list(
        range(len(classes_target))
    )

    relatorio_classificacao = classification_report(
        y_true=y_test_referencia,
        y_pred=y_pred_test,
        labels=labels_numericos,
        target_names=classes_target,
        zero_division=0,
    )

    metricas: dict[str, object] = {
        "modelo": "CNN",
        "accuracy_train": float(
            accuracy_train
        ),
        "accuracy_test": float(
            accuracy_test
        ),
        "balanced_accuracy_test": float(
            balanced_accuracy_test
        ),
        "macro_precision_test": float(
            macro_precision_test
        ),
        "macro_recall_test": float(
            macro_recall_test
        ),
        "macro_f1_test": float(
            macro_f1_test
        ),
        "weighted_f1_test": float(
            weighted_f1_test
        ),
        "metricas_por_classe": metricas_por_classe,
        "classification_report_test": relatorio_classificacao,
    }

    print("\nMétricas principais:")

    print(
        f"Accuracy treino: "
        f"{accuracy_train:.4f}"
    )

    print(
        f"Accuracy teste: "
        f"{accuracy_test:.4f}"
    )

    print(
        f"Balanced accuracy teste: "
        f"{balanced_accuracy_test:.4f}"
    )

    print(
        f"Macro precision teste: "
        f"{macro_precision_test:.4f}"
    )

    print(
        f"Macro recall teste: "
        f"{macro_recall_test:.4f}"
    )

    print(
        f"Macro F1-score teste: "
        f"{macro_f1_test:.4f}"
    )

    print(
        f"Weighted F1-score teste: "
        f"{weighted_f1_test:.4f}"
    )

    print("\nClassification report - Teste:")
    print(
        relatorio_classificacao
    )

    return (
        metricas,
        y_pred_test,
    )