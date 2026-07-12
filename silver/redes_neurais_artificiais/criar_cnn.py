# ======================================================================================
# cnn_modelo.py
# ======================================================================================
# Responsabilidade:
# - Definir a arquitetura da CNN 1D
# - Receber os atributos codificados por One-Hot Encoding
# - Produzir os logits para classificação multiclasse
# ======================================================================================

import torch  # pyrefly: ignore [missing-import]
from torch import nn  # pyrefly: ignore [missing-import]


class CNN1DClassificador(nn.Module):
    """
    CNN 1D para classificação multiclasse de dados tabulares.

    Formato esperado:
    - Entrada original: (batch_size, quantidade_features)
    - Entrada da convolução: (batch_size, 1, quantidade_features)
    - Saída: (batch_size, quantidade_classes)
    """

    def __init__(
        self,
        quantidade_classes: int,
        canais: tuple[int, int] = (32, 64),
        kernel_size: int = 3,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()

        if quantidade_classes <= 1:
            raise ValueError(
                "A quantidade de classes precisa ser maior que 1."
            )

        if len(canais) != 2:
            raise ValueError(
                "O parâmetro canais deve possuir exatamente dois valores."
            )

        canal_1, canal_2 = canais

        padding = kernel_size // 2

        self.extrator_caracteristicas = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=canal_1,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.BatchNorm1d(
                num_features=canal_1,
            ),
            nn.ReLU(),
            nn.MaxPool1d(
                kernel_size=2,
            ),

            nn.Conv1d(
                in_channels=canal_1,
                out_channels=canal_2,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.BatchNorm1d(
                num_features=canal_2,
            ),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(
                output_size=1,
            ),
        )

        self.classificador = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(
                p=dropout,
            ),
            nn.Linear(
                in_features=canal_2,
                out_features=quantidade_classes,
            ),
        )

    def forward(
        self,
        X: torch.Tensor,
    ) -> torch.Tensor:
        """
        Executa a propagação direta da CNN.
        """

        if X.ndim == 2:
            X = X.unsqueeze(
                dim=1,
            )

        if X.ndim != 3:
            raise ValueError(
                "A entrada da CNN deve possuir formato "
                "(batch_size, features) ou "
                "(batch_size, 1, features). "
                f"Formato recebido: {tuple(X.shape)}"
            )

        caracteristicas = self.extrator_caracteristicas(
            X
        )

        logits = self.classificador(
            caracteristicas
        )

        return logits