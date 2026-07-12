# ======================================================================================
# dados_cnn.py
# ======================================================================================
# Responsabilidade:
# - Converter DataFrame e Series para tensores
# - Disponibilizar os dados para o DataLoader da CNN
# ======================================================================================

import numpy as np
import pandas as pd
import torch  # pyrefly: ignore [missing-import]
from torch.utils.data import Dataset  # pyrefly: ignore [missing-import]


class DatasetTabularCNN(Dataset):
    """
    Dataset PyTorch para a CNN 1D.
    """

    def __init__(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> None:
        if isinstance(X, pd.DataFrame):
            X_numpy = X.to_numpy(
                dtype=np.float32,
                copy=True,
            )
        else:
            X_numpy = np.asarray(
                X,
                dtype=np.float32,
            )

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

        if X_numpy.ndim != 2:
            raise ValueError(
                "X deve possuir duas dimensões. "
                f"Formato recebido: {X_numpy.shape}"
            )

        if y_numpy.ndim != 1:
            raise ValueError(
                "y deve possuir uma dimensão. "
                f"Formato recebido: {y_numpy.shape}"
            )

        if len(X_numpy) != len(y_numpy):
            raise ValueError(
                "X e y possuem quantidades diferentes de registros. "
                f"X: {len(X_numpy)} | y: {len(y_numpy)}"
            )

        if np.isnan(X_numpy).any():
            raise ValueError(
                "X possui valores nulos ou NaN."
            )

        self.X = torch.tensor(
            X_numpy,
            dtype=torch.float32,
        )

        self.y = torch.tensor(
            y_numpy,
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(
        self,
        indice: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.X[indice],
            self.y[indice],
        )