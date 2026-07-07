# ======================================================================================
# pipeline_silver.py
# ======================================================================================
# Responsabilidade:
# - Ler a base de modelagem gerada pela Bronze
# - Preparar os dados para machine learning
# - Executar modelos de classificação
# - Avaliar e comparar os modelos
# - Salvar métricas, gráficos e melhor modelo
# ======================================================================================
from Pesquisa_principal.constants import ARQUIVO_CSV_LIMPO_TESTE_TREINO
from Pesquisa_principal.constants import COLUNA_ALVO_MODELAGEM
from Pesquisa_principal.constants import RANDOM_STATE
from Pesquisa_principal.constants import TEST_SIZE
from Pesquisa_principal.constants import ARQUIVO_ENCODER

from Pesquisa_principal.silver.processamento.normalizacao_colunas_modelo import normalizar_nomes_colunas_modelo

from Pesquisa_principal.silver.processamento.leitura_dados_bronze import ler_base_modelagem_bronze
from Pesquisa_principal.silver.processamento.separacao_target import separar_variaveis_explicativas_e_target
from Pesquisa_principal.silver.processamento.divisao_treino_teste import dividir_treino_teste_estratificado
from Pesquisa_principal.silver.processamento.encoding import aplicar_encoding_treino_teste

def dataset() -> dict:

    # Retorna a base de dados dividida em treino e teste, com encoding 
    # aplicado e nomes das colunas normalizados para compatibilidade com modelos.

    print("=" * 80)
    print("DIVISÃO TREINO/TESTE ESTRATIFICADA")
    print("=" * 80)

    # ==============================================================
    # leitura da base de modelagem
    # ==============================================================
    dataframe_modelagem = ler_base_modelagem_bronze(
        caminho_entrada=ARQUIVO_CSV_LIMPO_TESTE_TREINO,
        coluna_alvo=COLUNA_ALVO_MODELAGEM,
    )

    # ==============================================================
    # separação variáveis explicativas e target
    # ==============================================================
    X, y = separar_variaveis_explicativas_e_target(
        dataframe=dataframe_modelagem,
        coluna_alvo=COLUNA_ALVO_MODELAGEM,
    )


    # ==============================================================
    # divisão treino/teste
    # ==============================================================
    X_train, X_test, y_train, y_test = dividir_treino_teste_estratificado(
        X=X,
        y=y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    # ==============================================================
    # encoding
    # ==============================================================
    X_train_encoded, X_test_encoded, encoder = aplicar_encoding_treino_teste(
        X_train=X_train,
        X_test=X_test,
        caminho_encoder=ARQUIVO_ENCODER,
    )

    # ==============================================================
    # normalização dos nomes das colunas para compatibilidade com modelos
    # ==============================================================
    X_train_encoded, X_test_encoded = normalizar_nomes_colunas_modelo(
        X_train=X_train_encoded,
        X_test=X_test_encoded,
    )

    return {
        'X_train' : X_train_encoded,
        'X_test'  : X_test_encoded,
        'y_train' : y_train,
        'y_test'  : y_test,
    }

    return X_train_encoded, X_test_encoded, y_train, y_test
    