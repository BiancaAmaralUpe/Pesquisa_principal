# ======================================================================================
# normalizacao_violencias.py
# ======================================================================================
# Responsabilidade:
# - Criar variáveis analíticas relacionadas aos tipos de violência
# - Consolidar informações complementares da base
# - Classificar ocorrências letais tentadas ou consumadas
# - Preservar rastreabilidade das colunas originais
# ======================================================================================
# imports das libs 
import pandas as pd

# imports das variaveis constants.
from Pesquisa_principal.constants import COLUNA_ESPECIE_VIOLACAO
from Pesquisa_principal.constants import COLUNA_INDICADOR_LETALIDADE
from Pesquisa_principal.constants import COLUNA_SUBTIPO_LETALIDADE
from Pesquisa_principal.constants import MAPEAMENTO_INDICADOR_LETALIDADE
from Pesquisa_principal.constants import MAPEAMENTO_SUBTIPO_LETALIDADE
from Pesquisa_principal.constants import COLUNA_TIPO_VIOLENCIA_NORMALIZADO
from Pesquisa_principal.constants import MAPEAMENTO_TIPO_VIOLENCIA_NORMALIZADO
from Pesquisa_principal.constants import COLUNA_FAIXA_ETARIA_SUSPEITO
from Pesquisa_principal.constants import COLUNA_SUSPEITO_FAIXA_ETARIA_INFORMADA
from Pesquisa_principal.constants import VALOR_INFO_SUSPEITO_NAO_INFORMADA
from Pesquisa_principal.constants import MAPEAMENTO_CORRECAO_ESPECIE_VIOLACAO
from Pesquisa_principal.constants import COMPONENTES_VIOLENCIA

def imprimir_secao(titulo: str) -> None:
    """
    Imprime uma seção formatada no terminal/output.
    """

    print("\n" + "=" * 80)
    print(titulo)
    print("=" * 80)

def unificar_agravantes(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Cria uma coluna consolidada de agravantes.

    Regra:
    - Se agravantes e agravantes_policiais estiverem nulos, cria 'sem_agravante_informado'
    - Se apenas uma coluna estiver preenchida, usa o valor preenchido
    - Se as duas estiverem preenchidas, concatena as duas informações
    """

    dataframe = dataframe.copy()

    imprimir_secao("UNIFICAÇÃO DE AGRAVANTES")

    coluna_agravantes = "agravantes"
    coluna_agravantes_policiais = "agravantes_policiais"
    coluna_saida = "agravantes_unificados"

    if coluna_agravantes not in dataframe.columns:
        print(f"[AVISO] Coluna deletada: {coluna_agravantes}")
        dataframe[coluna_agravantes] = pd.NA

    if coluna_agravantes_policiais not in dataframe.columns:
        print(f"[AVISO] Coluna deletada: {coluna_agravantes_policiais}")
        dataframe[coluna_agravantes_policiais] = pd.NA

    dataframe[coluna_saida] = dataframe.apply(
        lambda linha: " | ".join(
            str(valor)
            for valor in [
                linha[coluna_agravantes],
                linha[coluna_agravantes_policiais],
            ]
            if pd.notna(valor)
        ),
        axis=1,
    )

    dataframe[coluna_saida] = dataframe[coluna_saida].replace(
        "",
        "sem_agravante_informado",
    )

    print(f"Coluna criada: {coluna_saida}")

    print("\nDistribuição da coluna criada:")
    print(
        dataframe[coluna_saida]
        .value_counts(dropna=False)
        .to_string()
    )

    return dataframe

def classificar_letalidade(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Cria colunas auxiliares para identificar violência letal tentada ou consumada.

    Colunas criadas:
    - indicador_letalidade
    - subtipo_letalidade
    """

    dataframe = dataframe.copy()

    imprimir_secao("CLASSIFICAÇÃO DE LETALIDADE")

    if COLUNA_ESPECIE_VIOLACAO not in dataframe.columns:
        print(f"[AVISO] Coluna deletada: {COLUNA_ESPECIE_VIOLACAO}")
        return dataframe

    dataframe[COLUNA_INDICADOR_LETALIDADE] = (
        dataframe[COLUNA_ESPECIE_VIOLACAO]
        .map(MAPEAMENTO_INDICADOR_LETALIDADE)
        .fillna("nao_letal")
    )

    dataframe[COLUNA_SUBTIPO_LETALIDADE] = (
        dataframe[COLUNA_ESPECIE_VIOLACAO]
        .map(MAPEAMENTO_SUBTIPO_LETALIDADE)
        .fillna("nao_aplicavel")
    )

    print(f"Coluna criada: {COLUNA_INDICADOR_LETALIDADE}")
    print(f"Coluna criada: {COLUNA_SUBTIPO_LETALIDADE}")

    print("\nDistribuição do indicador de letalidade:")
    print(
        dataframe[COLUNA_INDICADOR_LETALIDADE]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nDistribuição do subtipo de letalidade:")
    print(
        dataframe[COLUNA_SUBTIPO_LETALIDADE]
        .value_counts(dropna=False)
        .to_string()
    )

    return dataframe

def criar_indicador_faixa_etaria_suspeito_informada(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Cria uma coluna indicando se a faixa etária do suspeito foi informada.

    Valores:
    - sim
    - nao
    """

    dataframe = dataframe.copy()

    print("\n" + "=" * 80)
    print("CRIAÇÃO DO INDICADOR DE FAIXA ETÁRIA DO SUSPEITO INFORMADA")
    print("=" * 80)

    if COLUNA_FAIXA_ETARIA_SUSPEITO not in dataframe.columns:
        print(f"[AVISO] Coluna deletada: {COLUNA_FAIXA_ETARIA_SUSPEITO}")
        return dataframe

    dataframe[COLUNA_SUSPEITO_FAIXA_ETARIA_INFORMADA] = dataframe[
        COLUNA_FAIXA_ETARIA_SUSPEITO
    ].apply(
        lambda valor: "nao"
        if valor == VALOR_INFO_SUSPEITO_NAO_INFORMADA
        else "sim"
    )

    print(f"Coluna criada: {COLUNA_SUSPEITO_FAIXA_ETARIA_INFORMADA}")

    print("\nDistribuição:")
    print(
        dataframe[COLUNA_SUSPEITO_FAIXA_ETARIA_INFORMADA]
        .value_counts(dropna=False)
        .to_string()
    )

    return dataframe

def corrigir_especie_violacao(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Corrige variações textuais identificadas na coluna especie_violacao.
    """

    dataframe = dataframe.copy()

    imprimir_secao("CORREÇÃO TEXTUAL DA ESPÉCIE DE VIOLAÇÃO")

    if COLUNA_ESPECIE_VIOLACAO not in dataframe.columns:
        print(f"[AVISO] Coluna deletada: {COLUNA_ESPECIE_VIOLACAO}")
        return dataframe

    dataframe[COLUNA_ESPECIE_VIOLACAO] = (
        dataframe[COLUNA_ESPECIE_VIOLACAO]
        .replace(MAPEAMENTO_CORRECAO_ESPECIE_VIOLACAO)
        .astype("string")
    )

    print("Correções aplicadas em especie_violacao:")
    for valor_antigo, valor_novo in MAPEAMENTO_CORRECAO_ESPECIE_VIOLACAO.items():
        print(f"- {valor_antigo} -> {valor_novo}")

    return dataframe

def classificar_tipo_violencia_normalizado(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Cria a coluna tipo_violencia_normalizado a partir da espécie da violação.
    """

    dataframe = dataframe.copy()

    print("\n" + "=" * 80)
    print("CLASSIFICAÇÃO DO TIPO DE VIOLÊNCIA NORMALIZADO")
    print("=" * 80)

    if COLUNA_ESPECIE_VIOLACAO not in dataframe.columns:
        print(f"[AVISO] Coluna deletada: {COLUNA_ESPECIE_VIOLACAO}")
        return dataframe

    dataframe[COLUNA_TIPO_VIOLENCIA_NORMALIZADO] = (
        dataframe[COLUNA_ESPECIE_VIOLACAO]
        .map(MAPEAMENTO_TIPO_VIOLENCIA_NORMALIZADO)
        .fillna("outras_violencias")
    )

    print(f"Coluna criada: {COLUNA_TIPO_VIOLENCIA_NORMALIZADO}")

    print("\nDistribuição:")
    print(
        dataframe[COLUNA_TIPO_VIOLENCIA_NORMALIZADO]
        .value_counts(dropna=False)
        .to_string()
    )

    return dataframe

def texto_contem_algum_termo(
    valor: object,
    termos: list[str],
) -> bool:
    """
    Verifica se um texto contém algum termo de interesse.
    """

    if pd.isna(valor):
        return False

    texto = str(valor).upper()

    return any(termo.upper() in texto for termo in termos)

def criar_indicadores_risco_feminicidio(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Cria indicadores analíticos associados a sinais de risco de feminicídio.

    Observação:
    - Os indicadores são proxies analíticos.
    - Eles não substituem avaliação profissional de risco.
    - Alguns sinais dependem da existência de texto/categorias na base.
    """

    dataframe = dataframe.copy()

    imprimir_secao("CRIAÇÃO DE INDICADORES DE RISCO DE FEMINICÍDIO")

    colunas_necessarias = [
        "especie_violacao",
        "denuncia_emergencial",
        "agravantes_unificados",
        "relacao_vitima_suspeito",
        "tipo_violencia_normalizado",
        "subtipo_letalidade",
    ]

    for coluna in colunas_necessarias:
        if coluna not in dataframe.columns:
            print(f"[AVISO] Coluna deletada: {coluna}")
            dataframe[coluna] = pd.NA

    dataframe["risco_ameaca_morte"] = dataframe.apply(
        lambda linha: (
            linha["denuncia_emergencial"] == "RISCO IMINENTE DE MORTE DA VÍTIMA"
            or texto_contem_algum_termo(
                linha["agravantes_unificados"],
                ["RISCO DE MORTE"],
            )
            or linha["especie_violacao"] == "AMEAÇA/COAÇÃO"
        ),
        axis=1,
    )

    dataframe["risco_violencia_fisica_grave"] = dataframe.apply(
        lambda linha: (
            linha["tipo_violencia_normalizado"] == "violencia_fisica"
            or linha["subtipo_letalidade"]
            in [
                "tentativa_feminicidio",
                "tentativa_homicidio",
                "feminicidio",
                "homicidio",
            ]
            or linha["denuncia_emergencial"] == "VÍTIMA EM SANGRAMENTO"
        ),
        axis=1,
    )

    dataframe["risco_escalada_agressoes"] = dataframe[
        "agravantes_unificados"
    ].apply(
        lambda valor: texto_contem_algum_termo(
            valor,
            ["RESULTADO DAS AGRESSÕES SE PROLONGAM AO LONGO DO TEMPO"],
        )
    )

    dataframe["risco_controle_extremo"] = dataframe["especie_violacao"].isin(
        [
            "AUTONOMIA DE VONTADE",
            "CÁRCERE PRIVADO",
            "CONSTRANGIMENTO",
            "CRIMES CONTRA A SEGURANÇA PSÍQUICA",
            "CRIMES CONTRA A SEGURANÇA PSÍ QUICA",
        ]
    )

    dataframe["risco_separacao_termino"] = dataframe[
        "relacao_vitima_suspeito"
    ].isin(
        [
            "EX-MARIDO",
            "EX-COMPANHEIRO(A)",
            "EX- NAMORADO(A)",
            "EX-ESPOSA",
        ]
    )

    dataframe["risco_violencia_sexual_associada"] = (
        dataframe["tipo_violencia_normalizado"] == "violencia_sexual"
    )

    colunas_sinais_risco = [
        "risco_ameaca_morte",
        "risco_violencia_fisica_grave",
        "risco_escalada_agressoes",
        "risco_controle_extremo",
        "risco_separacao_termino",
        "risco_violencia_sexual_associada",
    ]

    dataframe["qtd_sinais_risco_feminicidio"] = dataframe[
        colunas_sinais_risco
    ].sum(axis=1)

    dataframe["classificacao_sinais_risco_feminicidio"] = dataframe[
        "qtd_sinais_risco_feminicidio"
    ].apply(
        lambda quantidade_sinais: (
            "risco_elevado"
            if quantidade_sinais >= 3
            else "risco_moderado"
            if quantidade_sinais == 2
            else "risco_baixo"
            if quantidade_sinais == 1
            else "sem_sinal_identificado"
        )
    )

    print("Colunas criadas:")
    for coluna in colunas_sinais_risco:
        print(f"- {coluna}")

    print("- qtd_sinais_risco_feminicidio")
    print("- classificacao_sinais_risco_feminicidio")

    print("\nDistribuição da classificação de sinais de risco:")
    print(
        dataframe["classificacao_sinais_risco_feminicidio"]
        .value_counts(dropna=False)
        .to_string()
    )

    return dataframe

def criar_componentes_violencia(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Cria colunas booleanas indicando componentes sobrepostos de violência.

    Exemplo:
    - TORTURA FÍSICA pode ter componente físico e psicológico.
    - ASSÉDIO SEXUAL PSÍQUICO pode ter componente sexual e psicológico.
    - TENTATIVA DE FEMINICÍDIO pode ter componente físico e letal.
    """

    dataframe = dataframe.copy()

    print("\n" + "=" * 80)
    print("CRIAÇÃO DE COMPONENTES SOBREPOSTOS DE VIOLÊNCIA")
    print("=" * 80)

    coluna_especie = "especie_violacao"

    if coluna_especie not in dataframe.columns:
        print(f"[AVISO] Coluna não encontrada: {coluna_especie}")
        return dataframe

    colunas_componentes = [
        "componente_fisico",
        "componente_psicologico",
        "componente_sexual",
        "componente_patrimonial",
        "componente_moral",
        "componente_letal",
    ]

    for coluna in colunas_componentes:
        dataframe[coluna] = False

    for especie, componentes in COMPONENTES_VIOLENCIA.items():
        mascara = dataframe[coluna_especie] == especie

        for componente, valor in componentes.items():
            if componente not in dataframe.columns:
                dataframe[componente] = False

            dataframe.loc[mascara, componente] = valor

    print("Colunas criadas:")
    for coluna in colunas_componentes:
        print(f"- {coluna}")

    print("\nDistribuição dos componentes:")
    for coluna in colunas_componentes:
        print(f"\n{coluna}:")
        print(dataframe[coluna].value_counts(dropna=False))

    return dataframe
    