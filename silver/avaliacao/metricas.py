# ======================================================================================
# metricas.py
# ======================================================================================
# Responsabilidade:
# - Exibir métricas dos modelos de forma organizada no output da Silver
# ======================================================================================


def exibir_metricas_modelo(
    metricas: dict,
) -> None:
    """
    Exibe as métricas do modelo em formato mais legível no output.
    """

    nome_modelo = metricas.get("modelo", "Modelo não informado")

    accuracy_train = metricas.get("accuracy_train", 0)
    accuracy_test = metricas.get("accuracy_test", 0)
    balanced_accuracy_test = metricas.get("balanced_accuracy_test", 0)
    macro_precision_test = metricas.get("macro_precision_test", 0)
    macro_recall_test = metricas.get("macro_recall_test", 0)
    macro_f1_test = metricas.get("macro_f1_test", 0)
    weighted_f1_test = metricas.get("weighted_f1_test", 0)

    diferenca_accuracy = accuracy_train - accuracy_test

    print("\n" + "=" * 80)
    print(f"MÉTRICAS DO MODELO - {nome_modelo.upper()}")
    print("=" * 80)

    print(f"{'Métrica':<35} {'Valor':>12}")
    print("-" * 80)
    print(f"{'Accuracy treino':<35} {accuracy_train * 100:>11.2f}%")
    print(f"{'Accuracy teste':<35} {accuracy_test * 100:>11.2f}%")
    print(f"{'Balanced accuracy teste':<35} {balanced_accuracy_test * 100:>11.2f}%")
    print(f"{'Macro precision teste':<35} {macro_precision_test * 100:>11.2f}%")
    print(f"{'Macro recall teste':<35} {macro_recall_test * 100:>11.2f}%")
    print(f"{'Macro F1-score teste':<35} {macro_f1_test * 100:>11.2f}%")
    print(f"{'Weighted F1-score teste':<35} {weighted_f1_test * 100:>11.2f}%")

    print("-" * 80)
    print(f"{'Diferença accuracy treino-teste':<35} {diferenca_accuracy * 100:>11.2f} p.p.")

    print("\nInterpretação rápida:")

    if abs(diferenca_accuracy) <= 0.02:
        print("- Não há sinal forte de overfitting pela diferença entre treino e teste.")
    elif diferenca_accuracy > 0.02:
        print("- Há possível sinal de overfitting: treino maior que teste.")
    else:
        print("- O teste ficou maior que o treino; verificar estabilidade da divisão ou efeito do balanceamento.")