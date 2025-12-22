import pandas as pd


def map_df_to_text(df_to_translate: pd.DataFrame, path_excel: str, columns_names: list[str]) -> pd.DataFrame:
    """
    Função para mapear os valores numéricos do DataFrame para seus equivalentes textuais,
    por meio de um dicionário de dados fornecido em um arquivo Excel.

    Parâmetros:
    - df_to_translate: pd.DataFrame
        DataFrame contendo os dados com valores numéricos a serem traduzidos.
    - path_excel: str
        Caminho para o arquivo Excel que contém o dicionário de dados.
    - columns_names: list[str]
        Lista com os novos nomes das colunas para o DataFrame traduzido.
    """
    try:
        # --- Passo 1: Carregar o Dicionário de Dados (arquivo XLSX) ---
        df_excel = pd.read_excel(path_excel)

        # --- Passo 2: Dicionário para VALORES das colunas ---
        # Preenche as células vazias na coluna 'Código da variável' com o valor não nulo mais recente
        df_excel['Código da variável'] = df_excel['Código da variável'].ffill()

        # Máscara para filtrar apenas as linhas que contêm códigos válidos das categorias (não textuais)
        mask = pd.to_numeric(df_excel['Código da categoria'], errors='coerce').notna()

        # Filtra pela máscara e pelas colunas "Código da variável", "Código da categoria" e "Descrição da categoria"
        df_excel = df_excel.loc[mask, ['Código da variável', 'Código da categoria', 'Descrição da categoria']]
        df_excel['Código da categoria'] = pd.to_numeric(df_excel['Código da categoria'], errors='coerce').astype('Int64')

        # Cria o dicionário aninhado para a substituição de valores das colunas
        map_values = {}
        for var_code, group in df_excel.groupby('Código da variável'):
            map_values[str(var_code).strip()] = group.set_index('Código da categoria')['Descrição da categoria'].to_dict()

    except FileNotFoundError:
        print(f"Erro: O arquivo '{path_excel}' não foi encontrado.")
        exit()
    
    df_to_return = df_to_translate.replace(map_values)
    df_to_return.columns = columns_names

    return df_to_return
