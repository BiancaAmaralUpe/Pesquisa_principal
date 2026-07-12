# ======================================================================================
# mlp_modelo.py
# ======================================================================================
# Responsabilidade:
# - Treinar um modelo Multilayer Perceptron para classificação multiclasse
# - Avaliar o modelo
# - Registrar os experimentos em arquivo .txt
# ======================================================================================

from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support


def treinar_mlp(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
    hidden_layer_sizes: tuple[int, ...],
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
) -> MLPClassifier:
    """
    Treina um modelo Multilayer Perceptron.

    O modelo é aplicado sobre a base codificada por One-Hot Encoding
    e balanceada apenas no conjunto de treino.
    """

    print("\n" + "=" * 80)
    print("TREINAMENTO DO MODELO - MLP")
    print("=" * 80)

    print(f"Formato de X_train: {X_train.shape}")
    print(f"Formato de y_train: {y_train.shape}")

    print("\nParâmetros do modelo:")
    print(f"hidden_layer_sizes: {hidden_layer_sizes}")
    print(f"activation: {activation}")
    print(f"solver: {solver}")
    print(f"alpha: {alpha}")
    print(f"batch_size: {batch_size}")
    print(f"learning_rate_init: {learning_rate_init}")
    print(f"max_iter: {max_iter}")
    print(f"early_stopping: {early_stopping}")
    print(f"validation_fraction: {validation_fraction}")
    print(f"n_iter_no_change: {n_iter_no_change}")
    print(f"random_state: {random_state}")

    # ==============================================================
    # ==============================================================
    # verificação do formato do target
    # ==============================================================
    y_train_validacao = pd.Series(y_train)

    if y_train_validacao.isna().any():
        raise ValueError(
            "O target de treinamento do MLP possui valores nulos."
        )

    if not pd.api.types.is_numeric_dtype(y_train_validacao):
        raise TypeError(
            "O target do MLP precisa estar codificado numericamente. "
            f"Tipo recebido: {y_train_validacao.dtype}. "
            "Aplique LabelEncoder antes do treinamento."
        )

    print(f"\nTipo do target recebido pelo MLP: {y_train_validacao.dtype}")
    print(
        "Classes numéricas recebidas pelo MLP: "
        f"{sorted(y_train_validacao.unique().tolist())}"
    )

    modelo = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
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
    )

    modelo.fit(
        X_train,
        y_train,
    )

    print("\nModelo MLP treinado com sucesso.")

    return modelo


def avaliar_mlp(
    modelo: MLPClassifier,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    classes_target: list[str],
) -> tuple[dict, object]:
    """
    Avalia o modelo MLP.
    """

    print("\n" + "=" * 80)
    print("AVALIAÇÃO DO MODELO - MLP")
    print("=" * 80)

    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)
    
        # ==========================================================================
    # Métricas detalhadas por classe
    # ==========================================================================
    labels_numericos = list(
        range(len(classes_target))
    )

    matriz = confusion_matrix(
        y_true=y_test,
        y_pred=y_pred_test,
        labels=labels_numericos,
    )

    precisoes, recalls, f1_scores, suportes = (
        precision_recall_fscore_support(
            y_true=y_test,
            y_pred=y_pred_test,
            labels=labels_numericos,
            zero_division=0,
        )
    )

    total_registros = int(
        matriz.sum()
    )

    metricas_por_classe = {}

    print("\n" + "=" * 80)
    print("MÉTRICAS DETALHADAS POR CLASSE - MLP")
    print("=" * 80)

    for indice, nome_classe in enumerate(classes_target):
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
            "precision": float(precisoes[indice]),
            "recall": float(recalls[indice]),
            "f1_score": float(f1_scores[indice]),
            "support": int(suportes[indice]),
            "verdadeiros_positivos": verdadeiros_positivos,
            "falsos_positivos": falsos_positivos,
            "falsos_negativos": falsos_negativos,
            "verdadeiros_negativos": verdadeiros_negativos,
        }

        print("\n" + "-" * 80)
        print(f"Classe: {nome_classe}")
        print("-" * 80)

        print(f"Precision: {precisoes[indice]:.4f}")
        print(f"Recall: {recalls[indice]:.4f}")
        print(f"F1-score: {f1_scores[indice]:.4f}")
        print(f"Support: {int(suportes[indice])}")

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

    accuracy_train = accuracy_score(
        y_train,
        y_pred_train,
    )

    accuracy_test = accuracy_score(
        y_test,
        y_pred_test,
    )

    balanced_accuracy_test = balanced_accuracy_score(
        y_test,
        y_pred_test,
    )

    macro_precision_test = precision_score(
        y_test,
        y_pred_test,
        average="macro",
        zero_division=0,
    )

    macro_recall_test = recall_score(
        y_test,
        y_pred_test,
        average="macro",
        zero_division=0,
    )

    macro_f1_test = f1_score(
        y_test,
        y_pred_test,
        average="macro",
        zero_division=0,
    )

    weighted_f1_test = f1_score(
        y_test,
        y_pred_test,
        average="weighted",
        zero_division=0,
    )

    metricas = {
        "modelo": "MLP",
        "accuracy_train": accuracy_train,
        "accuracy_test": accuracy_test,
        "balanced_accuracy_test": balanced_accuracy_test,
        "macro_precision_test": macro_precision_test,
        "macro_recall_test": macro_recall_test,
        "macro_f1_test": macro_f1_test,
        "weighted_f1_test": weighted_f1_test,
        "metricas_por_classe": metricas_por_classe,
    }

    print("\nMétricas principais:")
    print(f"Accuracy treino: {accuracy_train:.4f}")
    print(f"Accuracy teste: {accuracy_test:.4f}")
    print(f"Balanced accuracy teste: {balanced_accuracy_test:.4f}")
    print(f"Macro precision teste: {macro_precision_test:.4f}")
    print(f"Macro recall teste: {macro_recall_test:.4f}")
    print(f"Macro F1-score teste: {macro_f1_test:.4f}")
    print(f"Weighted F1-score teste: {weighted_f1_test:.4f}")

    print("\nClassification report - Teste:")
    print(
        classification_report(
            y_test,
            y_pred_test,
            zero_division=0,
        )
    )

    return metricas, y_pred_test

def registrar_treinamento_mlp(
    caminho_log: Path,
    hidden_layer_sizes: tuple[int, ...],
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
    X_train_shape: tuple[int, int],
    X_test_shape: tuple[int, int],
    y_train_shape: tuple[int],
    y_test_shape: tuple[int],
    metricas: dict,
    observacao: str = "",
) -> None:
    """
    Registra em arquivo .txt os parâmetros e métricas do treinamento do MLP.
    """

    caminho_log.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data_execucao = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(caminho_log, "a", encoding="utf-8") as arquivo:
        arquivo.write("\n")
        arquivo.write("=" * 80)
        arquivo.write("\n")
        arquivo.write("EXECUÇÃO DO MODELO FINAL - MLP\n")
        arquivo.write("=" * 80)
        arquivo.write("\n")

        arquivo.write(f"Data/hora da execução: {data_execucao}\n\n")

        arquivo.write("Parâmetros utilizados:\n")
        arquivo.write(f"- hidden_layer_sizes: {hidden_layer_sizes}\n")
        arquivo.write(f"- activation: {activation}\n")
        arquivo.write(f"- solver: {solver}\n")
        arquivo.write(f"- alpha: {alpha}\n")
        arquivo.write(f"- batch_size: {batch_size}\n")
        arquivo.write(f"- learning_rate_init: {learning_rate_init}\n")
        arquivo.write(f"- max_iter: {max_iter}\n")
        arquivo.write(f"- early_stopping: {early_stopping}\n")
        arquivo.write(f"- validation_fraction: {validation_fraction}\n")
        arquivo.write(f"- n_iter_no_change: {n_iter_no_change}\n")
        arquivo.write(f"- RANDOM_STATE: {random_state}\n\n")

        arquivo.write("Formato dos dados:\n")
        arquivo.write(f"- X_train: {X_train_shape}\n")
        arquivo.write(f"- X_test: {X_test_shape}\n")
        arquivo.write(f"- y_train: {y_train_shape}\n")
        arquivo.write(f"- y_test: {y_test_shape}\n\n")

        arquivo.write("Métricas obtidas:\n")
        arquivo.write(f"- Accuracy treino: {metricas['accuracy_train']:.4f}\n")
        arquivo.write(f"- Accuracy teste: {metricas['accuracy_test']:.4f}\n")
        arquivo.write(
            f"- Balanced accuracy teste: "
            f"{metricas['balanced_accuracy_test']:.4f}\n"
        )
        arquivo.write(
            f"- Macro precision teste: "
            f"{metricas['macro_precision_test']:.4f}\n"
        )
        arquivo.write(
            f"- Macro recall teste: "
            f"{metricas['macro_recall_test']:.4f}\n"
        )
        arquivo.write(f"- Macro F1-score teste: {metricas['macro_f1_test']:.4f}\n")

        arquivo.write(
            f"- Weighted F1-score teste: "
            f"{metricas['weighted_f1_test']:.4f}\n"
        )
        # ==============================================================
        # métricas detalhadas por classe
        # ==============================================================
        metricas_por_classe = metricas.get(
            "metricas_por_classe",
            {},
        )

        if metricas_por_classe:
            arquivo.write("\n")
            arquivo.write("Métricas detalhadas por classe:\n")

            for nome_classe, resultados in metricas_por_classe.items():
                arquivo.write("\n")
                arquivo.write("-" * 80)
                arquivo.write("\n")

                arquivo.write(
                    f"Classe: {nome_classe}\n"
                )

                arquivo.write(
                    f"- Precision: "
                    f"{resultados['precision']:.4f}\n"
                )

                arquivo.write(
                    f"- Recall: "
                    f"{resultados['recall']:.4f}\n"
                )

                arquivo.write(
                    f"- F1-score: "
                    f"{resultados['f1_score']:.4f}\n"
                )

                arquivo.write(
                    f"- Support: "
                    f"{resultados['support']}\n"
                )

                arquivo.write(
                    f"- Verdadeiros positivos: "
                    f"{resultados['verdadeiros_positivos']}\n"
                )

                arquivo.write(
                    f"- Falsos positivos: "
                    f"{resultados['falsos_positivos']}\n"
                )

                arquivo.write(
                    f"- Falsos negativos: "
                    f"{resultados['falsos_negativos']}\n"
                )

                arquivo.write(
                    f"- Verdadeiros negativos: "
                    f"{resultados['verdadeiros_negativos']}\n"
                )
                
        arquivo.write(f"- n_iter_no_change: {n_iter_no_change}\n")
        arquivo.write(f"- tol: {tol}\n")
        arquivo.write(f"- RANDOM_STATE: {random_state}\n\n")

        if observacao:
            arquivo.write("\nObservação:\n")
            arquivo.write(f"{observacao}\n")

        arquivo.write("\n")