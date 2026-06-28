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
from Pesquisa_principal.constants import ARQUIVO_OUTPUT_SILVER
from Pesquisa_principal.constants import COLUNA_ALVO_MODELAGEM
from Pesquisa_principal.bronze.utils import OutputTerminalEArquivo


def pipeline_silver() -> None:
    """
    Executa a camada Silver.
    """

    with OutputTerminalEArquivo(ARQUIVO_OUTPUT_SILVER):
        print("=" * 80)
        print("INÍCIO DA PIPELINE SILVER - MODELAGEM")
        print("=" * 80)

        print("\nArquivo de entrada da Silver:")
        print(ARQUIVO_CSV_LIMPO_TESTE_TREINO)

        print("\nColuna alvo da modelagem:")
        print(COLUNA_ALVO_MODELAGEM)

        print("\nPróximas etapas da Silver:")
        print("1. Leitura da base de treino/teste")
        print("2. Separação de X e y")
        print("3. Split treino/teste")
        print("4. Encoding")
        print("5. Normalização")
        print("6. Treinamento dos modelos")
        print("7. Avaliação")
        print("8. Comparação")
        print("9. Salvamento do melhor modelo")

        print("\n" + "=" * 80)
        print("FIM DA PIPELINE SILVER - MODELAGEM")
        print("=" * 80)

        print(f"\nOutput Silver salvo em: {ARQUIVO_OUTPUT_SILVER}")