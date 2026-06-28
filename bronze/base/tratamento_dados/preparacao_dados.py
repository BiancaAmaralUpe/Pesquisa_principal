# ======================================================================================
# preparacao_dados.py
# ======================================================================================
# Responsabilidade:
# - Preparar a base analítica para etapas futuras de modelagem
# - Separar variável alvo e variáveis explicativas
# - Remover colunas que causam vazamento de informação
# - Preencher nulos com marcadores informativos
# - Identificar colunas categóricas
# - Validar estrutura final antes de encoding/normalização
# ======================================================================================
# Import libs
import pandas as pd

# Imports tratamento dos dados
from Pesquisa_principal.bronze.base.tratamento_dados.limpeza_dados import salvar_dataframe_limpo
from Pesquisa_principal.bronze.analise_dados.definir_metodologia import validar_dataframe

# Imports da analise das analise de dados
from Pesquisa_principal.bronze.analise_dados.analise_completude import imprimir_secao

# Imports das variaveis das constants
from Pesquisa_principal.constants import COLUNAS_MANTER_BASE_TREINO_TESTE
from Pesquisa_principal.constants import MARCADORES_NULOS_MODELAGEM
from Pesquisa_principal.constants import COLUNAS_REMOVER_BASE_TREINO
from Pesquisa_principal.constants import ESPECIES_REMOVER_BASE_TREINO_TESTE

def validar_coluna_alvo(
    dataframe: pd.DataFrame,
    coluna_alvo: str,
) -> None:
    """
    Valida se a coluna alvo existe no DataFrame.
    """

    imprimir_secao("VALIDAÇÃO DA COLUNA ALVO")

    if coluna_alvo not in dataframe.columns:
        raise ValueError(f"Coluna alvo não encontrada: {coluna_alvo}")

    print(f"Coluna alvo encontrada: {coluna_alvo}")

def validar_base_pre_modelagem(
    dataframe: pd.DataFrame,
) -> None:
    """
    Valida a estrutura da base antes da modelagem.
    """

    imprimir_secao("VALIDAÇÃO DA BASE PRÉ-MODELAGEM")

    print(f"Linhas: {dataframe.shape[0]}")
    print(f"Colunas: {dataframe.shape[1]}")
    print(f"Total de valores nulos: {dataframe.isna().sum().sum()}")

    print("\nTipos de dados:")
    print(dataframe.dtypes)

def remover_colunas_base_treino(
    dataframe: pd.DataFrame,
    coluna_alvo: str,
) -> pd.DataFrame:
    """
    Remove colunas que não devem entrar como variáveis explicativas na base de treino.
    """

    dataframe = dataframe.copy()

    imprimir_secao("REMOÇÃO DE COLUNAS DA BASE DE TREINO")

    colunas_para_remover = [
        coluna
        for coluna in COLUNAS_REMOVER_BASE_TREINO
        if coluna in dataframe.columns and coluna != coluna_alvo
    ]

    if not colunas_para_remover:
        print("Nenhuma coluna removida da base de treino.")
        return dataframe

    dataframe = dataframe.drop(columns=colunas_para_remover)

    print("Colunas removidas do X para evitar vazamento ou redundância:")
    for coluna in colunas_para_remover:
        print(f"- {coluna}")

    print(f"\nQuantidade de colunas após remoção: {dataframe.shape[1]}")

    return dataframe

def preencher_nulos_com_marcadores(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Preenche valores nulos com marcadores informativos para modelagem.
    """

    dataframe = dataframe.copy()

    imprimir_secao("PREENCHIMENTO DE NULOS COM MARCADORES")

    for coluna, marcador in MARCADORES_NULOS_MODELAGEM.items():
        if coluna not in dataframe.columns:
            print(f"- {coluna}: Coluna deletada.")
            continue

        qtd_nulos = dataframe[coluna].isna().sum()

        dataframe[coluna] = dataframe[coluna].fillna(marcador)

        print(f"- {coluna}: {qtd_nulos} nulos preenchidos com '{marcador}'")

    print(f"\nTotal de nulos após preenchimento: {dataframe.isna().sum().sum()}")

    return dataframe
def remover_registros_com_nulos_em_colunas_obrigatorias(
    dataframe: pd.DataFrame,
    colunas_obrigatorias: list[str],
) -> pd.DataFrame:
    """
    Remove registros que possuem valores nulos nas colunas obrigatórias
    para a base de treino e teste.
    """

    dataframe = dataframe.copy()

    imprimir_secao("REMOÇÃO DE REGISTROS COM NULOS EM COLUNAS OBRIGATÓRIAS")

    colunas_existentes = [
        coluna for coluna in colunas_obrigatorias
        if coluna in dataframe.columns
    ]

    colunas_ausentes = [
        coluna for coluna in colunas_obrigatorias
        if coluna not in dataframe.columns
    ]

    for coluna in colunas_ausentes:
        print(f"[AVISO] Coluna deletada: {coluna}")

    linhas_antes = len(dataframe)

    dataframe = dataframe.dropna(subset=colunas_existentes)

    linhas_depois = len(dataframe)
    linhas_removidas = linhas_antes - linhas_depois

    print(f"Linhas antes: {linhas_antes}")
    print(f"Linhas depois: {linhas_depois}")
    print(f"Linhas removidas: {linhas_removidas}")

    print("\nColunas exigidas com preenchimento:")
    for coluna in colunas_existentes:
        print(f"- {coluna}")

    return dataframe

def identificar_colunas_categoricas(
    dataframe: pd.DataFrame,
) -> list[str]:
    """
    Identifica colunas categóricas da base.
    """

    colunas_categoricas = dataframe.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    imprimir_secao("COLUNAS CATEGÓRICAS IDENTIFICADAS")

    for coluna in colunas_categoricas:
        print(f"- {coluna}")

    return colunas_categoricas

def separar_variavel_alvo(
    dataframe: pd.DataFrame,
    coluna_alvo: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separa a base em variáveis explicativas X e variável alvo y.
    """

    if coluna_alvo not in dataframe.columns:
        raise ValueError(f"Coluna alvo não encontrada: {coluna_alvo}")

    X = dataframe.drop(columns=[coluna_alvo])
    y = dataframe[coluna_alvo]

    imprimir_secao("SEPARAÇÃO DA VARIÁVEL ALVO")

    print(f"Coluna alvo: {coluna_alvo}")
    print(f"Formato de X: {X.shape}")
    print(f"Formato de y: {y.shape}")

    return X, y

def selecionar_colunas_base_treino_teste(
    dataframe: pd.DataFrame,
    colunas_manter: list[str],
) -> pd.DataFrame:
    """
    Seleciona somente as colunas definidas para a base de treino e teste.
    """

    dataframe = dataframe.copy()

    imprimir_secao("SELEÇÃO DE COLUNAS DA BASE DE TREINO E TESTE")

    colunas_existentes = [
        coluna
        for coluna in colunas_manter
        if coluna in dataframe.columns
    ]

    colunas_ausentes = [
        coluna
        for coluna in colunas_manter
        if coluna not in dataframe.columns
    ]

    for coluna in colunas_ausentes:
        print(f"[AVISO] Coluna deletada: {coluna}")

    dataframe = dataframe[colunas_existentes]

    print("Colunas mantidas no df_limpo_teste_treino.csv:")
    for coluna in colunas_existentes:
        print(f"- {coluna}")

    print(f"\nQuantidade de colunas mantidas: {dataframe.shape[1]}")

    return dataframe

def remover_especies_indesejadas_base_treino_teste(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove da base de treino e teste espécies de violação que foram
    descartadas por decisão metodológica.
    """

    dataframe = dataframe.copy()

    imprimir_secao("REMOÇÃO DE ESPÉCIES INDESEJADAS DA BASE DE TREINO E TESTE")

    coluna_especie = "especie_violacao"

    if coluna_especie not in dataframe.columns:
        print(f"[AVISO] Coluna deletada: {coluna_especie}")
        return dataframe

    linhas_antes = len(dataframe)

    dataframe = dataframe[
        ~dataframe[coluna_especie].isin(ESPECIES_REMOVER_BASE_TREINO_TESTE)
    ].copy()

    linhas_depois = len(dataframe)
    linhas_removidas = linhas_antes - linhas_depois

    print(f"Linhas antes: {linhas_antes}")
    print(f"Linhas depois: {linhas_depois}")
    print(f"Linhas removidas: {linhas_removidas}")

    print("\nEspécies removidas:")
    for especie in ESPECIES_REMOVER_BASE_TREINO_TESTE:
        print(f"- {especie}")

    return dataframe

def preparar_dados_modelagem(
    dataframe: pd.DataFrame,
    coluna_alvo: str,
    caminho_saida: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Executa a preparação inicial dos dados para modelagem.
    """

    imprimir_secao("INÍCIO DA PREPARAÇÃO DOS DADOS PARA MODELAGEM")

    if not validar_dataframe(dataframe):
        return dataframe.copy(), pd.Series(dtype="object"), dataframe.copy()

    validar_coluna_alvo(
        dataframe=dataframe,
        coluna_alvo=coluna_alvo,
    )

    validar_base_pre_modelagem(dataframe)

    dataframe_treino = selecionar_colunas_base_treino_teste(
        dataframe=dataframe,
        colunas_manter=COLUNAS_MANTER_BASE_TREINO_TESTE,
    )

    dataframe_treino = remover_especies_indesejadas_base_treino_teste(
        dataframe=dataframe_treino,
    )

    dataframe_treino = preencher_nulos_com_marcadores(
        dataframe=dataframe_treino,
    )

    dataframe_treino = remover_colunas_base_treino(
        dataframe=dataframe_treino,
        coluna_alvo=coluna_alvo,
    )

    validar_base_pre_modelagem(dataframe_treino)

    identificar_colunas_categoricas(dataframe_treino)

    X, y = separar_variavel_alvo(
        dataframe=dataframe_treino,
        coluna_alvo=coluna_alvo,
    )

    salvar_dataframe_limpo(
        dataframe=dataframe_treino,
        caminho_saida=caminho_saida,
    )

    print("\nBase preparada para análise/modelagem.")
    print(f"X: {X.shape}")
    print(f"y: {y.shape}")
    print(f"Base treino/teste: {dataframe_treino.shape}")

    print(f"\nArquivo salvo em: {caminho_saida}")

    imprimir_secao("FIM DA PREPARAÇÃO DOS DADOS PARA MODELAGEM")

    return X, y, dataframe_treino