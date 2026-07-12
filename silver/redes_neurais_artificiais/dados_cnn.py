# ======================================================================================
# dados_cnn.py
# ======================================================================================
# Responsabilidade:
# - Converter os dados para tensores do PyTorch
# - Separar treino interno e validação
# - Aplicar balanceamento somente no treino interno
# - Criar os DataLoaders de treino, validação, avaliação e teste
# ======================================================================================

import numpy as np
import pandas as pd
import torch  # pyrefly: ignore [missing-import]

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader  # pyrefly: ignore [missing-import]
from torch.utils.data import Dataset  # pyrefly: ignore [missing-import]


class DatasetTabularCNN(Dataset):
    """
    Dataset PyTorch utilizado pela CNN 1D.
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

        if not np.isfinite(X_numpy).all():
            raise ValueError(
                "X possui valores nulos, infinitos ou inválidos."
            )

        self.X = torch.from_numpy(
            X_numpy
        ).float()

        self.y = torch.from_numpy(
            y_numpy
        ).long()

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


def converter_X_para_dataframe(
    X: pd.DataFrame | np.ndarray,
) -> pd.DataFrame:
    """
    Converte X para DataFrame e reorganiza os índices.
    """

    if isinstance(X, pd.DataFrame):
        return X.reset_index(
            drop=True
        ).copy()

    X_numpy = np.asarray(
        X
    )

    if X_numpy.ndim != 2:
        raise ValueError(
            "X deve possuir duas dimensões. "
            f"Formato recebido: {X_numpy.shape}"
        )

    return pd.DataFrame(
        X_numpy
    )


def converter_y_para_series(
    y: pd.Series | np.ndarray | list,
) -> pd.Series:
    """
    Converte o target para Series e reorganiza os índices.
    """

    if isinstance(y, pd.Series):
        y_series = y.reset_index(
            drop=True
        ).copy()
    else:
        y_series = pd.Series(
            np.asarray(
                y
            ).reshape(-1)
        )

    return y_series.astype(
        np.int64
    )


def balancear_treino_cnn(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    quantidade_por_classe: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Aplica balanceamento híbrido no treino interno.

    Para cada classe:
    - reduz classes com mais registros que a quantidade definida;
    - replica registros de classes com menos registros;
    - mantém exatamente a mesma quantidade por classe.
    """

    if quantidade_por_classe <= 0:
        raise ValueError(
            "quantidade_por_classe deve ser maior que zero."
        )

    if len(X_train) != len(y_train):
        raise ValueError(
            "X_train e y_train possuem tamanhos diferentes."
        )

    nome_target_temporario = "__target_cnn__"

    base_treino = X_train.reset_index(
        drop=True
    ).copy()

    base_treino[nome_target_temporario] = (
        y_train.reset_index(
            drop=True
        )
    )

    partes_balanceadas = []

    classes_ordenadas = sorted(
        base_treino[nome_target_temporario]
        .unique()
        .tolist()
    )

    print("\n" + "=" * 80)
    print("BALANCEAMENTO DO TREINO INTERNO - CNN")
    print("=" * 80)

    print(
        f"Quantidade desejada por classe: "
        f"{quantidade_por_classe}"
    )

    print("\nDistribuição antes do balanceamento:")
    print(
        base_treino[
            nome_target_temporario
        ].value_counts().sort_index()
    )

    for indice_classe, classe in enumerate(
        classes_ordenadas
    ):
        registros_classe = base_treino[
            base_treino[nome_target_temporario]
            == classe
        ]

        quantidade_atual = len(
            registros_classe
        )

        usar_reposicao = (
            quantidade_atual
            < quantidade_por_classe
        )

        registros_amostrados = registros_classe.sample(
            n=quantidade_por_classe,
            replace=usar_reposicao,
            random_state=(
                random_state
                + indice_classe
            ),
        )

        partes_balanceadas.append(
            registros_amostrados
        )

    base_balanceada = pd.concat(
        partes_balanceadas,
        axis=0,
        ignore_index=True,
    )

    base_balanceada = base_balanceada.sample(
        frac=1.0,
        random_state=random_state,
    ).reset_index(
        drop=True
    )

    y_train_balanceado = base_balanceada.pop(
        nome_target_temporario
    ).astype(
        np.int64
    )

    X_train_balanceado = base_balanceada

    print("\nDistribuição após o balanceamento:")
    print(
        y_train_balanceado
        .value_counts()
        .sort_index()
    )

    print("\nFormato após o balanceamento:")
    print(
        f"X_train_balanceado: "
        f"{X_train_balanceado.shape}"
    )
    print(
        f"y_train_balanceado: "
        f"{y_train_balanceado.shape}"
    )

    return (
        X_train_balanceado,
        y_train_balanceado,
    )


def criar_dataloaders_cnn(
    X_train: pd.DataFrame | np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray | list,
    y_test: pd.Series | np.ndarray | list,
    validation_fraction: float,
    quantidade_balanceamento: int,
    batch_size: int,
    num_workers: int,
    random_state: int,
) -> dict[str, object]:
    """
    Prepara os dados utilizados pela CNN.

    Etapas:
    - separa treino interno e validação;
    - aplica balanceamento somente no treino interno;
    - cria DataLoader para treino balanceado;
    - cria DataLoader para validação original;
    - cria DataLoader para avaliação do treino original;
    - cria DataLoader para teste original.
    """

    if not 0 < validation_fraction < 1:
        raise ValueError(
            "validation_fraction deve estar entre 0 e 1."
        )

    if batch_size <= 0:
        raise ValueError(
            "batch_size deve ser maior que zero."
        )

    if num_workers < 0:
        raise ValueError(
            "num_workers não pode ser negativo."
        )

    X_train_dataframe = converter_X_para_dataframe(
        X_train
    )

    X_test_dataframe = converter_X_para_dataframe(
        X_test
    )

    y_train_series = converter_y_para_series(
        y_train
    )

    y_test_series = converter_y_para_series(
        y_test
    )

    if len(X_train_dataframe) != len(y_train_series):
        raise ValueError(
            "X_train e y_train possuem quantidades "
            "diferentes de registros."
        )

    if len(X_test_dataframe) != len(y_test_series):
        raise ValueError(
            "X_test e y_test possuem quantidades "
            "diferentes de registros."
        )

    print("\n" + "=" * 80)
    print("PREPARAÇÃO DOS DADOS - CNN 1D")
    print("=" * 80)

    print(
        f"Formato do treino original: "
        f"{X_train_dataframe.shape}"
    )

    print(
        f"Formato do teste original: "
        f"{X_test_dataframe.shape}"
    )

    print(
        f"Fração de validação: "
        f"{validation_fraction}"
    )

    (
        X_train_interno,
        X_validation,
        y_train_interno,
        y_validation,
    ) = train_test_split(
        X_train_dataframe,
        y_train_series,
        test_size=validation_fraction,
        random_state=random_state,
        stratify=y_train_series,
    )

    X_train_interno = X_train_interno.reset_index(
        drop=True
    )

    X_validation = X_validation.reset_index(
        drop=True
    )

    y_train_interno = y_train_interno.reset_index(
        drop=True
    )

    y_validation = y_validation.reset_index(
        drop=True
    )

    print("\nFormato após divisão interna:")
    print(
        f"X_train_interno: "
        f"{X_train_interno.shape}"
    )
    print(
        f"X_validation: "
        f"{X_validation.shape}"
    )
    print(
        f"y_train_interno: "
        f"{y_train_interno.shape}"
    )
    print(
        f"y_validation: "
        f"{y_validation.shape}"
    )

    print("\nDistribuição do treino interno:")
    print(
        y_train_interno
        .value_counts()
        .sort_index()
    )

    print("\nDistribuição da validação:")
    print(
        y_validation
        .value_counts()
        .sort_index()
    )

    (
        X_train_balanceado,
        y_train_balanceado,
    ) = balancear_treino_cnn(
        X_train=X_train_interno,
        y_train=y_train_interno,
        quantidade_por_classe=quantidade_balanceamento,
        random_state=random_state,
    )

    dataset_train = DatasetTabularCNN(
        X=X_train_balanceado,
        y=y_train_balanceado,
    )

    dataset_validation = DatasetTabularCNN(
        X=X_validation,
        y=y_validation,
    )

    dataset_train_evaluation = DatasetTabularCNN(
        X=X_train_interno,
        y=y_train_interno,
    )

    dataset_test = DatasetTabularCNN(
        X=X_test_dataframe,
        y=y_test_series,
    )

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    validation_loader = DataLoader(
        dataset=dataset_validation,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    train_evaluation_loader = DataLoader(
        dataset=dataset_train_evaluation,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        dataset=dataset_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    print("\nDataLoaders criados com sucesso.")

    print(
        f"Quantidade de batches de treino: "
        f"{len(train_loader)}"
    )

    print(
        f"Quantidade de batches de validação: "
        f"{len(validation_loader)}"
    )

    print(
        f"Quantidade de batches de avaliação do treino: "
        f"{len(train_evaluation_loader)}"
    )

    print(
        f"Quantidade de batches de teste: "
        f"{len(test_loader)}"
    )

    return {
        "train_loader": train_loader,
        "validation_loader": validation_loader,
        "train_evaluation_loader": train_evaluation_loader,
        "test_loader": test_loader,
        "X_train_balanceado_shape": X_train_balanceado.shape,
        "y_train_balanceado_shape": y_train_balanceado.shape,
        "X_train_avaliacao_shape": X_train_interno.shape,
        "y_train_avaliacao": y_train_interno,
        "X_validation_shape": X_validation.shape,
        "y_validation": y_validation,
        "X_test_shape": X_test_dataframe.shape,
        "y_test": y_test_series,
    }