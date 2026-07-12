# ======================================================================================
# treinamento_cnn.py
# ======================================================================================
# Responsabilidade:
# - Treinar a CNN 1D
# - Calcular loss e accuracy por época
# - Aplicar early stopping
# - Salvar o melhor estado do modelo
# - Restaurar o melhor modelo ao final do treinamento
# ======================================================================================

from pathlib import Path

import torch  # pyrefly: ignore [missing-import]
from torch import nn  # pyrefly: ignore [missing-import]
from torch.optim import Adam  # pyrefly: ignore [missing-import]
from torch.utils.data import DataLoader  # pyrefly: ignore [missing-import]


def definir_dispositivo_cnn(
    dispositivo: str | None = None,
) -> torch.device:
    """
    Define o dispositivo utilizado no treinamento.

    Prioridade:
    - dispositivo informado;
    - CUDA, quando disponível;
    - CPU.
    """

    if dispositivo is not None:
        return torch.device(dispositivo)

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def executar_epoca_cnn(
    modelo: nn.Module,
    data_loader: DataLoader,
    criterio: nn.Module,
    dispositivo: torch.device,
    otimizador: Adam | None = None,
) -> tuple[float, float]:
    """
    Executa uma época de treinamento ou validação.

    Quando o otimizador é informado, o modelo é treinado.
    Quando o otimizador é None, o modelo é apenas avaliado.

    Retorna:
    - loss média;
    - accuracy média.
    """

    modo_treinamento = otimizador is not None

    if modo_treinamento:
        modelo.train()
    else:
        modelo.eval()

    perda_acumulada = 0.0
    quantidade_acertos = 0
    quantidade_registros = 0

    contexto_gradiente = (
        torch.enable_grad()
        if modo_treinamento
        else torch.no_grad()
    )

    with contexto_gradiente:
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(
                dispositivo,
                dtype=torch.float32,
            )

            y_batch = y_batch.to(
                dispositivo,
                dtype=torch.long,
            )

            if modo_treinamento:
                otimizador.zero_grad()

            logits = modelo(
                X_batch
            )

            perda = criterio(
                logits,
                y_batch,
            )

            if modo_treinamento:
                perda.backward()
                otimizador.step()

            quantidade_batch = y_batch.size(0)

            perda_acumulada += (
                perda.item()
                * quantidade_batch
            )

            predicoes = torch.argmax(
                logits,
                dim=1,
            )

            quantidade_acertos += (
                predicoes == y_batch
            ).sum().item()

            quantidade_registros += quantidade_batch

    if quantidade_registros == 0:
        raise ValueError(
            "O DataLoader recebido não possui registros."
        )

    perda_media = (
        perda_acumulada
        / quantidade_registros
    )

    accuracy_media = (
        quantidade_acertos
        / quantidade_registros
    )

    return (
        float(perda_media),
        float(accuracy_media),
    )


def treinar_cnn(
    modelo: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    learning_rate: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    min_delta: float,
    caminho_modelo: str | Path,
    dispositivo: str | None = None,
    random_state: int = 42,
) -> tuple[nn.Module, dict[str, object]]:
    """
    Treina a CNN utilizando early stopping.

    O melhor modelo é definido pela menor loss de validação.

    Retorna:
    - modelo com o melhor estado restaurado;
    - histórico completo do treinamento.
    """

    if learning_rate <= 0:
        raise ValueError(
            "learning_rate deve ser maior que zero."
        )

    if weight_decay < 0:
        raise ValueError(
            "weight_decay não pode ser negativo."
        )

    if max_epochs <= 0:
        raise ValueError(
            "max_epochs deve ser maior que zero."
        )

    if patience <= 0:
        raise ValueError(
            "patience deve ser maior que zero."
        )

    if min_delta < 0:
        raise ValueError(
            "min_delta não pode ser negativo."
        )

    torch.manual_seed(
        random_state
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            random_state
        )

    dispositivo_torch = definir_dispositivo_cnn(
        dispositivo=dispositivo,
    )

    caminho_modelo = Path(
        caminho_modelo
    )

    caminho_modelo.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    modelo = modelo.to(
        dispositivo_torch
    )

    criterio = nn.CrossEntropyLoss()

    otimizador = Adam(
        modelo.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    historico: dict[str, object] = {
        "epocas": [],
        "train_loss": [],
        "validation_loss": [],
        "train_accuracy": [],
        "validation_accuracy": [],
        "melhor_epoca": None,
        "melhor_validation_loss": None,
        "epocas_executadas": 0,
        "early_stopping_acionado": False,
        "dispositivo": str(dispositivo_torch),
    }

    melhor_validation_loss = float("inf")
    melhor_epoca = 0
    contador_sem_melhoria = 0

    print("\n" + "=" * 80)
    print("TREINAMENTO DO MODELO - CNN 1D")
    print("=" * 80)

    print(
        f"Dispositivo utilizado: "
        f"{dispositivo_torch}"
    )

    print(
        f"Quantidade máxima de épocas: "
        f"{max_epochs}"
    )

    print(
        f"Learning rate: "
        f"{learning_rate}"
    )

    print(
        f"Weight decay: "
        f"{weight_decay}"
    )

    print(
        f"Patience: "
        f"{patience}"
    )

    print(
        f"Min delta: "
        f"{min_delta}"
    )

    for epoca in range(
        1,
        max_epochs + 1,
    ):
        train_loss, train_accuracy = executar_epoca_cnn(
            modelo=modelo,
            data_loader=train_loader,
            criterio=criterio,
            dispositivo=dispositivo_torch,
            otimizador=otimizador,
        )

        validation_loss, validation_accuracy = executar_epoca_cnn(
            modelo=modelo,
            data_loader=validation_loader,
            criterio=criterio,
            dispositivo=dispositivo_torch,
            otimizador=None,
        )

        historico["epocas"].append(
            epoca
        )

        historico["train_loss"].append(
            train_loss
        )

        historico["validation_loss"].append(
            validation_loss
        )

        historico["train_accuracy"].append(
            train_accuracy
        )

        historico["validation_accuracy"].append(
            validation_accuracy
        )

        historico["epocas_executadas"] = epoca

        print(
            f"Época {epoca:02d}/{max_epochs} | "
            f"Train loss: {train_loss:.6f} | "
            f"Validation loss: {validation_loss:.6f} | "
            f"Train accuracy: {train_accuracy:.4f} | "
            f"Validation accuracy: {validation_accuracy:.4f}"
        )

        houve_melhoria = (
            validation_loss
            < melhor_validation_loss - min_delta
        )

        if houve_melhoria:
            melhor_validation_loss = validation_loss
            melhor_epoca = epoca
            contador_sem_melhoria = 0

            torch.save(
                modelo.state_dict(),
                caminho_modelo,
            )

            print(
                "Melhor modelo atualizado e salvo em: "
                f"{caminho_modelo}"
            )

        else:
            contador_sem_melhoria += 1

            print(
                "Épocas sem melhoria: "
                f"{contador_sem_melhoria}/{patience}"
            )

        if contador_sem_melhoria >= patience:
            historico["early_stopping_acionado"] = True

            print("\nEarly stopping acionado.")
            print(
                f"Melhor época: {melhor_epoca}"
            )

            break

    if not caminho_modelo.exists():
        raise FileNotFoundError(
            "O arquivo com o melhor modelo da CNN "
            "não foi encontrado após o treinamento."
        )

    melhor_estado = torch.load(
        caminho_modelo,
        map_location=dispositivo_torch,
    )

    modelo.load_state_dict(
        melhor_estado
    )

    modelo.eval()

    historico["melhor_epoca"] = melhor_epoca
    historico["melhor_validation_loss"] = (
        melhor_validation_loss
    )

    print("\n" + "=" * 80)
    print("TREINAMENTO DA CNN FINALIZADO")
    print("=" * 80)

    print(
        f"Épocas executadas: "
        f"{historico['epocas_executadas']}"
    )

    print(
        f"Melhor época: "
        f"{melhor_epoca}"
    )

    print(
        f"Melhor loss de validação: "
        f"{melhor_validation_loss:.6f}"
    )

    print(
        f"Modelo restaurado de: "
        f"{caminho_modelo}"
    )

    return (
        modelo,
        historico,
    )