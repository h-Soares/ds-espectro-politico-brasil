import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import streamlit as st
import altair as alt
from catboost import CatBoostClassifier
    

# Configuração da página
st.set_page_config(
    page_title="Análise e predição do posicionamento político do eleitor brasileiro",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título da aplicação
st.title("Análise e predição do posicionamento político do eleitor brasileiro")
st.markdown("""
<span style="font-size:20px;">
O presente trabalho tem como objetivo realizar, primeiramente, uma <em>análise exploratória</em> dos dados provenientes de uma pesquisa de opinião sobre o perfil ideológico dos eleitores brasileiros, feita pelo Instituto DataSenado no ano de 2024. O conjunto de dados original contém diversas variáveis de natureza sociodemográfica, comportamental e política, mas nem todas serão utilizadas. Serão mantidas somente aquelas que possam contribuir para compreender a variável alvo (<em>target</em>) <b>Posicionamento Político</b>, que representa o espectro político autodeclarado por cada participante. A análise de dados buscará, portanto, investigar as relações entre a variável target e as demais variáveis selecionadas, por meio de visualizações gráficas e métodos estatísticos. 

Após a fase exploratória, o projeto avançará para a segunda etapa: a construção de um modelo preditivo. Com o uso de técnicas de <em>machine learning</em>, o objetivo será desenvolver um modelo capaz de prever o espectro político com base nas respostas às variáveis selecionadas da pesquisa.

<em>Importante mencionar que este trabalho não se preocupará em fazer inferência sobre a preferência política da população brasileira como um todo, tendo em vista que uma parcela significativa dos dados originais não será utilizada.</em>

Projeto final realizado para a disciplina de Ciência de Dados do curso de Ciência da Computação, <em>Universidade de São Paulo</em>.
</span>

<span style="font-size:18px;">
            
* Os dados utilizados neste trabalho foram obtidos por meio dos microdados disponibilizados pelo Instituto DataSenado em <a href="https://www12.senado.leg.br/institucional/datasenado/publicacaodatasenado?id=pesquisa-traca-perfil-ideologico-dos-eleitores-brasileiros" target="_blank" rel="noopener noreferrer">Pesquisa traça perfil ideológico dos eleitores brasileiros</a>.
* Junto com os dados foi disponibilizado um dicionário para auxiliar na interpretação.
</span>
""", unsafe_allow_html=True)

# Estilo das abas e perguntas (aumenta tamanho da fonte)
st.markdown(
    """
    <style>
    /* Botão da tab */
    div.stTabs button[role="tab"] {
        font-size: 22px !important;
        padding: 0.35rem 0.75rem !important;
    }
    /* Cor da tab ativa e underline (azul mais escuro para combinar com o preto) */
    div.stTabs button[role="tab"][aria-selected="true"] {
        color: #2563eb !important; /* azul 600 */
    }
    /* Interações devem ficar azuis (sem vermelho ao clicar) */
    div.stTabs button[role="tab"]:hover,
    div.stTabs button[role="tab"]:focus,
    div.stTabs button[role="tab"]:active {
        color: #2563eb !important; /* azul 600 */
    }
    div.stTabs [data-baseweb="tab-highlight"] {
        background-color: #2563eb !important; /* underline azul 600 */
    }
    /* Texto dentro do botão */
    div.stTabs button[role="tab"] p {
        font-size: 22px !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    /* Labels dos selectboxes e inputs */
    label {
        font-size: 20px !important;
    }
    label span {
        font-size: 20px !important;
    }
    /* Reduz espaço do hr (---) */
    hr {
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    /* Texto das análises sob as imagens */
    .analysis-text {
        font-size: 20px !important;
        line-height: 1.6 !important;
    }
    /* Botão de enviar do formulário de predição (azul mais escuro) */
    div.stForm button {
        background-color: #2563eb !important; /* azul base 600 */
        color: #ffffff !important;
        border: 1px solid #1e40af !important; /* azul 800 */
    }
    div.stForm button:hover {
        background-color: #1d4ed8 !important; /* azul 700 */
        border-color: #1e3a8a !important; /* azul 800/900 */
    }
    div.stForm button:active {
        background-color: #1e40af !important; /* azul 800 */
        border-color: #1e3a8a !important;
    }
    /* Container para links do topo */
    .links-row {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        align-items: center;
    }
    /* Link do GitHub fixado no canto superior esquerdo */
    .github-link {
        position: relative; /* não fica fixo ao rolar */
        color: #dbeafe !important;
        text-decoration: none !important;
        font-weight: 700;
        font-size: 18px;
        background: rgba(15, 23, 42, 0.92);
        padding: 6px 10px;
        border-radius: 10px;
        display: inline-flex;
        align-items: center;
        gap: 8px;
    }
    .github-link:hover {
        color: #bfdbfe !important;
        background: rgba(30, 41, 59, 0.96);
    }
    /* Link da apresentação em slides */
    .slides-link {
        position: relative;
        color: #dbeafe !important;
        text-decoration: none !important;
        font-weight: 700;
        font-size: 18px;
        background: rgba(15, 23, 42, 0.92);
        padding: 6px 10px;
        border-radius: 10px;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        border: 1px solid #0b162a;
    }
    .slides-link:hover {
        color: #bfdbfe !important;
        background: rgba(30, 41, 59, 0.96);
        border-color: #0b162a;
    }
    /* Selectbox border azul */
    div[data-baseweb="select"] {
        border-color: #2563eb !important;
    }
    div[data-baseweb="select"] > div {
        border-color: #2563eb !important;
    }
    div[data-baseweb="select"] input {
        border-color: #2563eb !important;
    }
    /* Focus state para selectbox */
    div[data-baseweb="select"]:focus-within {
        border-color: #1d4ed8 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="links-row">'
    '<a class="github-link" href="https://github.com/h-Soares/ds-espectro-politico-brasil" target="_blank">'
    '<svg aria-hidden="true" height="18" viewBox="0 0 16 16" version="1.1" width="18" style="fill:#dbeafe; margin-right:6px;">'
    '<path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8Z"></path>'
    '</svg>GitHub do projeto</a>'
    '<a class="slides-link" href="https://docs.google.com/presentation/d/e/2PACX-1vSeCWRmjrOq4eQA0pxj9Ybh-GfI0uRyIGz-V-9Hdrqo9z9Lw0Hw7vE65Bpko5eW9cMiaYk_ELcd6fw2/pub?start=true&loop=false&delayms=3000" target="_blank">'
    '<svg aria-hidden="true" height="18" viewBox="0 0 24 24" width="18" style="fill:#fbbc04; margin-right:6px;">'
    '<path d="M6 2h9l5 5v13a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2Zm0 2v16h12V9h-5V4H6Zm10 10H8v4h8v-4Z"/>'
    '</svg>Apresentação em slides</a>'
    '</div>',
    unsafe_allow_html=True,
)
st.markdown("---")

# Criar abas (análises primeiro, predição em seguida)
tab_analysis, tab_pred = st.tabs(["📊 Análise de dados", "🔮 Predição do posicionamento político"])

# Carregar o modelo
@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model_path = project_root / "models" / "catboost_posicionamento_politico.cbm"
    model.load_model(str(model_path))
    return model

# Carregar dados completos para análises (arquivo já textual)
@st.cache_data
def load_full_df_text():
    """Carrega o DataFrame completo com valores textuais para análises"""
    csv_path = project_root / "data" / "processed" / "data_ml2024.csv"
    df_ml = pd.read_csv(str(csv_path), sep=';')
    return df_ml

# Carregar dados de exemplo para extrair os valores únicos das variáveis (arquivo já textual)
@st.cache_data
def load_variable_options():
    df_ml = load_full_df_text()
    
    # Extrair valores únicos de cada variável (excluindo 'posicionamento_politico')
    feature_columns = [col for col in df_ml.columns if col != 'posicionamento_politico']
    
    # Criar dicionário com valores únicos para cada variável
    variable_options = {}
    for col in feature_columns:
        # Obter valores únicos e remover "Não se aplica"
        unique_values = sorted([val for val in df_ml[col].unique() if val != 'Não se aplica' and pd.notna(val)])
        variable_options[col] = unique_values
    
    return variable_options

# Carregar modelo e opções
model = load_model()
variable_options = load_variable_options()

# Dicionário com descrições das variáveis (mais amigáveis)
variable_labels = {
    'empresas_impedir_fnews': '1. Na sua opinião, as empresas donas das redes sociais deveriam ser responsáveis por impedir a divulgação de notícias falsas?',
    'importancia_controle_fnews': '2. Para garantir uma disputa justa nas eleições, quão importante é o controle das notícias falsas nas redes sociais?',
    'petrobras_combustiveis': '3. Você é a favor ou contra: "Diminuir o lucro da Petrobrás para reduzir o preço dos combustíveis"',
    'uso_maconha': '4. Você é a favor ou contra: "Autorizar que as pessoas usem maconha como quiserem"',
    'cotas_universidades': '5. Você concorda ou discorda: "O sistema de cotas para negros em universidades é justo"',
    'direito_aborto': '6. Você concorda ou discorda: "As mulheres devem ter o direito de interromper a gravidez com segurança, caso elas queiram"',
    'pena_morte': '7. Você concorda ou discorda: "Deveria existir pena de morte no Brasil"',
    'posse_armas': '8. Você concorda ou discorda: "Facilitar a posse de armas aumenta a segurança no Brasil"',
    'confianca_urnas': '9. Você concorda ou discorda: "O resultado das urnas eletrônicas em eleições é confiável"',
    'satisfacao_democracia': '10. Em geral, qual o seu nível de satisfação com a democracia no Brasil?',
    'regime_governo': '11. Em sua opinião, qual o melhor regime de governo?',
    'religiao': '12. Qual sua religião ou crença?',
    'escolaridade': '13. Qual sua escolaridade?',
    'faixa_etaria': '14. Qual sua faixa etária?'
}

# ============================================================
# ABA 1: PREVISÃO DE POSICIONAMENTO POLÍTICO
# ============================================================
with tab_pred:
    # Seção informativa sobre o modelo
    st.markdown("### Sobre o modelo de predição")
    st.markdown(
        '<span style="font-size:18px;">Algoritmo utilizado: <em><strong>CatBoost Classifier</strong></em></span>',
        unsafe_allow_html=True,
    )
    
    st.markdown(
        """
        <div style="font-size:18px; margin-top:10px;">
        O <strong>CatBoost</strong> é um algoritmo que utiliza a técnica de <em>gradient boosting</em> em <em>árvores de decisão</em> 
        e destaca-se pela sua eficiência em lidar com variáveis categóricas de forma nativa. Foi escolhido por trabalhar diretamente 
        com dados textuais, sem necessidade de pré-processamento complexo como <em>One-Hot Encoding</em>.
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown("---")
    
    # Métricas de Desempenho
    st.markdown("#### Métricas de Desempenho do Modelo")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("F1-Score Macro", "≈0.57", help="Avalia de forma equilibrada todos os espectros políticos")
    
    with col2:
        st.metric("F1-Score Weighted", "≈0.63", help="Considera a proporção de cada classe no dataset")
    
    with col3:
        st.metric("Acurácia Geral", "≈62%", help="Porcentagem de predições corretas")
    
    st.markdown(
        """
        <div style="font-size:17px; margin-top:15px;">
        <strong>Desempenho por posicionamento político:</strong>
        <ul style="margin-top:8px;">
            <li><strong>Direita:</strong> F1-Score de 0.73 (melhor desempenho)</li>
            <li><strong>Esquerda:</strong> F1-Score de 0.61 (desempenho intermediário)</li>
            <li><strong>Centro:</strong> F1-Score de 0.38 (maior dificuldade de classificação)</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Aviso importante sobre limitações
    st.info(
        "⚠️ **Limitação importante:** O modelo apresenta maior dificuldade em classificar o posicionamento de **Centro**, "
        "pois este espectro frequentemente compartilha características tanto com a Esquerda quanto com a Direita. "
    )
    
    st.markdown("---")
    
    # Visualizações do modelo
    st.markdown("#### Visualizações do Modelo")
    
    # Criar duas colunas para as imagens
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        st.markdown("**Matriz de Confusão Normalizada**")
        confusion_matrix_path = project_root / 'images' / 'matriz_confusao_catboost.png'
        if confusion_matrix_path.exists():
            st.image(str(confusion_matrix_path), use_container_width=True)
            st.markdown(
                '<div style="font-size:16px; margin-top:8px;"><b>Análise:</b> A matriz revela que o modelo confunde <strong>Centro</strong> '
                'com <strong>Direita</strong> e <strong>Esquerda</strong>, mas os espectros bem definidos '
                '(Esquerda e Direita) não se confundem muito entre si.</div>',
                unsafe_allow_html=True,
            )
    
    with viz_col2:
        st.markdown("**Top 5 Variáveis Mais Importantes**")
        feature_importance_path = project_root / 'images' / 'feature_importance_catboost.png'
        if feature_importance_path.exists():
            st.image(str(feature_importance_path), use_container_width=True)
            st.markdown(
                '<div style="font-size:16px; margin-top:8px;"><b>Análise:</b> O gráfico revela que, dentre as questões mais polêmicas, ' \
                'a <strong>confiança nas urnas eletrônicas</strong> é a variável mais importante para o modelo.</div>',
                unsafe_allow_html=True,
            )
    
    st.markdown("---")
    
    # Formulário de predição
    st.markdown("### Responda às seguintes questões:")
    st.markdown(
        '<span style="font-size:19px;">Selecione uma opção para cada pergunta abaixo. Após responder todas as perguntas, clique no botão para obter a previsão do seu posicionamento político.</span>',
        unsafe_allow_html=True,
    )

    # Criar um formulário para as respostas
    with st.form("prediction_form"):
        user_responses = {}
        
        # Criar selectbox para cada variável
        for col in variable_options.keys():
            label = variable_labels.get(col)
            # Renderizar label personalizado com fonte maior e esconder label padrão do selectbox
            st.markdown(f'<div style="font-size:20px; font-weight:500; margin:0.25rem 0;">{label}</div>', unsafe_allow_html=True)
            user_responses[col] = st.selectbox(
                label="",
                options=variable_options[col],
                index=None,
                placeholder="Escolha uma opção",
                key=col,
                label_visibility="collapsed"
            )
        
        # Botão para fazer a previsão
        submit_button = st.form_submit_button("Prever posicionamento político")

    # Verificar se todas as perguntas foram respondidas e fazer a previsão
    if submit_button:
        # Verificar se todas as variáveis foram respondidas
        unanswered = [col for col, answer in user_responses.items() if answer is None]
        
        if unanswered:
            st.error(f"⚠️ Por favor, responda todas as perguntas antes de continuar!")
            st.error(f"Perguntas não respondidas: {len(unanswered)}")
        else:
            # Criar DataFrame com as respostas do usuário
            user_df = pd.DataFrame([user_responses])
            
            # Fazer a previsão
            prediction = model.predict(user_df)[0]
            prediction_probabilities = model.predict_proba(user_df)[0]

            # Normalizar saída para texto simples e maiúsculo
            def _extract_label(pred_value):
                if isinstance(pred_value, (list, tuple)):
                    return str(pred_value[0]) if pred_value else ""
                try:
                    import numpy as np  # noqa: WPS433 - import inside helper to avoid global dependency
                    if isinstance(pred_value, np.ndarray):
                        return str(pred_value.flatten()[0]) if pred_value.size else ""
                except Exception:
                    pass
                return str(pred_value)

            prediction_text = _extract_label(prediction)
            prediction_display = prediction_text.upper()
            
            # Obter as classes do modelo
            classes = model.classes_
            
            # Exibir resultado
            st.markdown("---")
            st.markdown("### Resultado da previsão")
            
            # Mostrar a previsão principal
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.success(f"Posicionamento político previsto: **{prediction_display}**")
            
            with col2:
                st.markdown(" ")
            
            # Mostrar probabilidades
            st.markdown("#### Probabilidade de posicionamento político")
            
            prob_df = pd.DataFrame({
                'Posicionamento': classes,
                'Probabilidade (%)': [f"{prob*100:.2f}%" for prob in prediction_probabilities]
            })
            
            st.table(prob_df)
            
            # Visualização em gráfico de barras (labels horizontais)
            chart_df = pd.DataFrame({
                'Posicionamento': classes,
                'Probabilidade': prediction_probabilities
            })

            chart = alt.Chart(chart_df).mark_bar(color="#86c5f8").encode(
                x=alt.X('Posicionamento:N', axis=alt.Axis(labelAngle=0, title='Posicionamento', labelFontSize=16, titleFontSize=18)),
                y=alt.Y('Probabilidade:Q', axis=alt.Axis(format='%', title='Probabilidade', labelFontSize=16, titleFontSize=18)),
                tooltip=[
                    alt.Tooltip('Posicionamento:N', title='Posicionamento'),
                    alt.Tooltip('Probabilidade:Q', format='.2%', title='Probabilidade')
                ]
            ).properties(height=300)

            st.altair_chart(chart, use_container_width=True)
            
            st.markdown("---")
            st.markdown("**Nota**: A previsão é baseada no modelo de Machine Learning treinado com dados de pesquisa política do ano de 2024.")

# ============================================================
# ABA 2: ANÁLISES DE DADOS
# ============================================================
with tab_analysis:
    st.markdown('<span style="font-size:18px;">Nesta seção será analisada a relação do espectro político com as demais questões importantes.</span>', unsafe_allow_html=True)
    st.markdown('<span style="font-size:18px;"><b>IMPORTANTE:</b> o conjunto de dados está desbalanceado, pois existem muito mais pessoas de Direita do que de Esquerda ou Centro.</span>', unsafe_allow_html=True)
    st.markdown("---")
    
    images_dir = project_root / 'images'
    
    # Ordem das imagens conforme a sequência apresentada no notebook data_analysis.ipynb
    desired_order = [
        'evolucao_posicionamento_politico.png',
        'cotas-posicionamento_politico.png',
        'direito_aborto-posicionamento_politico.png',
        'confianca_urnas-posicionamento_politico.png',
        'posse_armas-posicionamento_politico.png',
        'pena_morte-posicionamento_politico.png',
        'uso_maconha-posicionamento_politico.png',
        'lucro_petrobras-posicionamento_politico.png',
        'motivo_compart_fnews-posicionamento_politico.png',
        'identificar_fnews-posicionamento_politico.png',
        'empresas_impedir_fnews-posicionamento_politico.png',
        'importancia_controle_fnews-posicionamento_politico.png',
        'satisfacao_democracia-posicionamento_politico.png',
        'regime_governo-posicionamento_politico.png',
        'religiao-posicionamento_politico.png',
        'renda_familiar-posicionamento_politico.png',
        'escolaridade-posicionamento_politico.png',
        'faixa_etaria-posicionamento_politico.png',
        'sexo-posicionamento_politico.png',
        'cor_raca-posicionamento_politico.png',
        'estado-posicionamento_politico.png',
        'regiao-posicionamento_politico.png'
    ]

    # Encontrar todas as imagens que correspondem a análises de posicionamento_politico
    # (todas as imagens que terminam com "-posicionamento_politico.png"), respeitando a ordem do notebook
    analysis_images = []
    if images_dir.exists():
        analysis_images = [images_dir / name for name in desired_order if (images_dir / name).exists()]
    
    # Mapeamento amigável para os nomes das imagens
    image_labels = {
        'evolucao_posicionamento_politico.png': 'Série histórica: evolução do posicionamento político (2021–2024)',
        'estado-posicionamento_politico.png': 'Posicionamento político por estado',
        'motivo_compart_fnews-posicionamento_politico.png': 'Em sua opinião, qual o principal motivo para as pessoas compartilharem uma notícia falsa nas redes sociais?',
        'identificar_fnews-posicionamento_politico.png': 'Em sua opinião, é fácil ou difícil saber quais notícias são falsas nas redes sociais?',
        'empresas_impedir_fnews-posicionamento_politico.png': 'Na sua opinião, as empresas donas das redes sociais deveriam ser responsáveis por impedir a divulgação de notícias falsas?',
        'importancia_controle_fnews-posicionamento_politico.png': 'Para garantir uma disputa justa nas eleições, quão importante é o controle das notícias falsas nas redes sociais?',
        'lucro_petrobras-posicionamento_politico.png': 'Você é a favor ou contra: "Diminuir o lucro da Petrobrás para reduzir o preço dos combustíveis"',
        'uso_maconha-posicionamento_politico.png': 'Você é a favor ou contra: "Autorizar que as pessoas usem maconha como quiserem"',
        'cotas-posicionamento_politico.png': 'Você concorda ou discorda: "O sistema de cotas para negros em universidades é justo"',
        'direito_aborto-posicionamento_politico.png': 'Você concorda ou discorda: "As mulheres devem ter o direito de interromper a gravidez com segurança, caso elas queiram"',
        'pena_morte-posicionamento_politico.png': 'Você concorda ou discorda: "Deveria existir pena de morte no Brasil"',
        'posse_armas-posicionamento_politico.png': 'Você concorda ou discorda: "Facilitar a posse de armas aumenta a segurança no Brasil"',
        'confianca_urnas-posicionamento_politico.png': 'Você concorda ou discorda: "O resultado das urnas eletrônicas em eleições é confiável"',
        'satisfacao_democracia-posicionamento_politico.png': 'Em geral, qual o seu nível de satisfação com a democracia no Brasil?',
        'regime_governo-posicionamento_politico.png': 'Em sua opinião, qual o melhor regime de governo?',
        'sexo-posicionamento_politico.png': 'Sexo por posicionamento político',
        'cor_raca-posicionamento_politico.png': 'Cor ou raça por posicionamento político',
        'religiao-posicionamento_politico.png': 'Religião ou crença por posicionamento político',
        'renda_familiar-posicionamento_politico.png': 'Renda familiar por posicionamento político',
        'escolaridade-posicionamento_politico.png': 'Escolaridade por posicionamento político',
        'regiao-posicionamento_politico.png': 'Posicionamento político por região',
        'faixa_etaria-posicionamento_politico.png': 'Faixa etária por posicionamento político'
    }
    
    # Dicionário com textos de análise extraídos do notebook data_analysis.ipynb
    ANALYSIS_TEXTS = {
                'evolucao_posicionamento_politico.png': 'O gráfico acima mostra a série temporal do posicionamento político declarado pelos entrevistados em cada ano em que a pesquisa em questão foi realizada (excluindo respostas como "nenhuma" ou "não sei"). Um fato notório a se perceber é o número de entrevistados que se identificam com o posicionamento político de direita: mais de 50% nas três pesquisas mais recentes. Outro ponto importante é a quantidade de pessoas que se identificam como sendo de centro, número que aparentemente está em uma tendência de crescimento após queda nos anos anteriores. O espectro político de esquerda, por outro lado, apresenta uma tendência de queda.',
                
                'confianca_urnas-posicionamento_politico.png': 'A polarização política é observada: enquanto as pessoas que se identificam com o espectro político de esquerda e de centro confiam, em sua maioria, no sistema de votação brasileiro, os entrevistados de direita apresentam forte desconfiança.',
        
                'cor_raca-posicionamento_politico.png': 'Nota-se a predominância de pardos em todos os espectros políticos. A esquerda apresenta pequena vantagem na porcentagem de pessoas pretas em comparação aos demais espectros, enquanto a proporção de entrevistados da cor branca é semelhante em todos.',
        
                'cotas-posicionamento_politico.png': 'Observa-se que, entre pessoas de esquerda, há extrema concordância com o sistema de cotas, enquanto no espectro político de direita predomina uma rejeição considerável. No centro, a maioria concorda mas não tão fortemente quanto na esquerda.',
        
                'direito_aborto-posicionamento_politico.png': 'Esse gráfico representa um exemplo claro de polarização política: as pessoas de esquerda apoiam, em sua maioria, o direito ao aborto, enquanto no espectro político de direita há extrema discordância. Já as pessoas de centro não têm um consenso nessa questão.',
        
                'empresas_impedir_fnews-posicionamento_politico.png': 'É possível observar que os entrevistados de todos os espectros políticos apoiam, em mais de sua maioria, que empresas sejam responsáveis por impedir fake news, embora esteja no espectro de direita a maior taxa de rejeição, enquanto os entrevistados de esquerda a apoiam em quase sua totalidade.',
        
                'escolaridade-posicionamento_politico.png': 'Um ponto a se destacar é o alto grau de escolaridade dos entrevistados que se consideram de centro: quase a metade frequentou o ensino superior. Nos demais espectros a escolaridade é semelhante, com os entrevistados de esquerda tendo frequentado mais o ensino superior que os de direita.',
        
                'estado-posicionamento_politico.png': 'Por meio do gráfico, observa-se uma predominância do espectro político de direita em quase todos os estados da federação, com os estados da região Nordeste sendo os que abrigam maior porcentagem de entrevistados de esquerda. O centro atinge uma marca quase constante de 20% em todos os estados.',
        
                'faixa_etaria-posicionamento_politico.png': 'O espectro político de centro é o que contém a maior porcentagem de faixa etária até 29 anos, e a menor porcentagem de pessoas com mais de 50 anos, o que indica que boa parte dos entrevistados desse espectro são jovens. Os espectros de esquerda e de direita têm distribuições semelhantes.',
        
                'identificar_fnews-posicionamento_politico.png': 'Observa-se que os entrevistados dos diversos espectros políticos não possuem um consenso em relação à dificuldade em identificar notícias falsas nas redes sociais: a maioria considera fácil, mas essa vantagem em relação aos que consideram difícil é pequena.',
        
                'importancia_controle_fnews-posicionamento_politico.png': 'Os entrevistados de todos os espectros políticos apoiam, em mais de sua maioria, o controle de notícias falsas nas redes sociais durante o período eleitoral. Na esquerda ocorre a maior taxa de aprovação, enquanto os entrevistados de direita são os que mais a rejeitam.',
        
                'lucro_petrobras-posicionamento_politico.png': 'É possível observar um consenso em todos os espectros políticos em relação a esse tema, pois todos são em quase sua totalidade a favor de diminuir o lucro da Petrobrás, com o espectro político de esquerda sendo o que mais se destaca.',
        
                'motivo_compart_fnews-posicionamento_politico.png': 'Predomina a percepção de que notícias falsas são compartilhadas para mudar a opinião das pessoas. O segundo motivo mais apontado é que as pessoas não sabem que a notícia é falsa. A esquerda considera mais que notícias falsas são compartilhadas por representarem o que a pessoa pensa.',
        
                'pena_morte-posicionamento_politico.png': 'É possível observar que os entrevistados de centro são muito divididos em relação a esse tema, com quase metade a favor e a outra metade contrária. Em relação à esquerda e direita, o comportamento divergente é semelhante a outras questões polêmicas.',
        
                'posse_armas-posicionamento_politico.png': 'Observa-se que os entrevistados de esquerda são, em quase sua totalidade, contrários à posse de armas. Uma rejeição bastante significativa também ocorre no espectro político de centro. Já as pessoas de direita, em sua maioria, são favoráveis.',
        
                'regiao-posicionamento_politico.png': 'Pode-se observar que o espectro político de direita é predominante em todas as regiões, com maior dominância na região Norte. A região que possui maior concentração de esquerda é a Nordeste. O centro possui uma distribuição semelhante em quase todas as regiões.',
        
                'regime_governo-posicionamento_politico.png': 'Observa-se um consenso: mais da metade dos entrevistados de todos os espectros políticos consideram a democracia sempre o melhor regime de governo. Tanto no centro como na direita, o número que considera um governo autoritário como aceitável é quase o dobro em relação ao espectro de esquerda.',
        
                'religiao-posicionamento_politico.png': 'É possível observar uma predominância da religião católica tanto no espectro de esquerda quanto no de centro. No espectro de direita, a religião evangélica apresenta uma pequena vantagem em relação ao catolicismo e é a religião dominante nesse espectro político.',
        
                'renda_familiar-posicionamento_politico.png': 'É possível notar que em todos os espectros políticos predominam uma renda familiar abaixo de R$ 2.824,00. O espectro de centro se destaca por ser o que mais ganha acima de R$ 8.472,00.',
        
                'satisfacao_democracia-posicionamento_politico.png': 'Observa-se uma grande variedade de opiniões nos espectros políticos. Na esquerda se encontram os entrevistados que mais estão satisfeitos com a democracia, enquanto é no centro e na direita onde predomina a insatisfação.',
        
                'sexo-posicionamento_politico.png': 'Observa-se a predominância do sexo feminino em todos os espectros, principalmente no centro e na direita. A maior proporcionalidade de homens se encontra no espectro político de esquerda.',
        
                'uso_maconha-posicionamento_politico.png': 'O gráfico deixa bem claro que os entrevistados de todos os espectros políticos são, em sua maioria, contrários ao uso irrestrito da maconha, sendo as pessoas de direita as que se posicionam mais fortemente nesse sentido.'
            }
    
    # Criar um dicionário de análises com nome da imagem e índice
    analysis_list = [(img.name, image_labels.get(img.name)) for img in analysis_images]
    
    # Selectbox para escolher qual análise visualizar
    st.markdown("### Selecione uma análise para visualizar:")
    selected_analysis_name = st.selectbox(
        "Análise",
        options=[name for name, _ in analysis_list],
        format_func=lambda name: image_labels.get(name),
        label_visibility="collapsed"
    )
    
    # Exibir apenas a análise selecionada
    img_path = images_dir / selected_analysis_name
    if img_path.exists():
        title = image_labels.get(selected_analysis_name)
        
        st.markdown(f"#### {title}")
        
        # Centralizar a imagem na página
        col1, col2, col3 = st.columns([0.5, 2, 0.5])
        with col2:
            st.image(str(img_path), caption="Fonte: do autor", width=800)
        
            # Exibir texto de análise se disponível
            analysis_text = ANALYSIS_TEXTS.get(selected_analysis_name)
            if analysis_text:
                st.markdown(f'<div class="analysis-text"><b>Análise:</b> {analysis_text}</div>', unsafe_allow_html=True)
        
        st.markdown("---")

    # Análise final com conclusões sobre as posições políticas
    st.markdown("#### **Pontos a se destacar**")

    st.markdown("##### **Esquerda**")
    st.markdown(
                """
<div class="analysis-text">
    <ul>
        <li><b>Temas sociais:</b> Apresenta extrema concordância com o sistema de cotas raciais em universidades e apoia majoritariamente o direito ao aborto seguro. É quase totalmente contrária a facilitar a posse de armas e discorda de forma ampla da pena de morte.</li>
        <li><b>Democracia e Instituições:</b> Confia em quase sua totalidade nas urnas eletrônicas, sendo o espectro com maior confiança registrada. É o grupo que mais se considera muito satisfeito com a democracia no Brasil e tem a maior taxa de preferência pela democracia como regime de governo.</li>
        <li><b>Economia e Drogas:</b> É em quase sua totalidade a favor de diminuir o lucro da Petrobrás para reduzir o preço dos combustíveis, sendo o grupo com maior apoio a esta medida. Apresenta o maior grau de aceitação em relação ao uso irrestrito da maconha.</li>
        <li><b>Notícias falsas:</b> Considera mais fácil identificar fake news que os outros espectros. Apoia em quase sua totalidade que as empresas donas de redes sociais sejam responsáveis por impedir a divulgação de notícias falsas e é o grupo que mais vê o controle de fake news nas redes sociais como muito importante para eleições justas, além de ser o campo que mais considera que notícias falsas são compartilhadas por representarem o que a pessoa pensa.</li>
        <li><b>Perfil:</b> Possui a maior concentração de entrevistados na região Nordeste e de pessoas pretas. Em termos de escolaridade, frequentou mais o ensino superior do que os entrevistados de direita. É o espectro com a maior porcentagem de pessoas da religião espírita e sem religião ou crença. Além disso, é o grupo que possui a maior taxa de entrevistados acima de 50 anos, o que representa uma oposição ao senso comum de que pessoas mais velhas são do espectro político de direita. Em relação à renda, é o espectro com maior porcentagem de entrevistados que possuem uma renda familiar abaixo de R$ 2.824,00, mas a proporção de renda acima de R$ 8.472,00 é maior que o campo da direita.</li>
    </ul>
</div>
                """,
            unsafe_allow_html=True,
            )

    st.markdown("##### **Centro**")
    st.markdown(
                """
<div class="analysis-text">
    <ul>
        <li><b>Temas sociais:</b> Concorda majoritariamente com o sistema de cotas, mas de forma menos intensa que a esquerda. Não apresenta consenso sobre o direito ao aborto, apenas uma pequena tendência à rejeição. Apresenta grande rejeição à posse de armas e opiniões divididas sobre a pena de morte.</li>
        <li><b>Democracia e Instituições:</b> Confia majoritariamente nas urnas eletrônicas. Apresenta elevado grau de insatisfação com a democracia no Brasil e é o campo que apresenta maior indiferença em relação ao regime de governo.</li>
        <li><b>Economia e Drogas:</b> É a favor, em sua maioria, de diminuir o lucro da Petrobrás para reduzir o preço dos combustíveis, mas com considerável rejeição. Apresenta alto grau de discordância em relação ao uso irrestrito da maconha.</li>
        <li><b>Notícias falsas:</b> É o grupo que considera mais difícil identificar fake news, mas sem consenso perceptível. Apoia majoritariamente que as empresas donas de redes sociais sejam responsáveis por impedir a divulgação de notícias falsas, mas com um considerável nível de rejeição. Considera muito importante o controle de notícias falsas nas redes sociais para eleições justas e apresenta opinião dividida sobre o principal motivo para notícias falsas serem publicadas nas redes sociais.</li>
        <li><b>Perfil:</b> Apresenta distribuição semelhante em todas as regiões, com uma leve vantagem nas regiões Sul e Sudeste. Possui a maior concentração de pessoas pardas e brancas entre todos os espectros políticos. Destaca-se por ter o mais alto grau de escolaridade, com quase a metade tendo frequentado o ensino superior. A religião católica é a predominante neste grupo. Possui a maior proporção de entrevistados com até 29 anos e a menor com mais de 50 anos, o que indica que grande parte dos entrevistados desse espectro político é jovem. Além disso, é o espectro que tem a maior taxa de renda familiar acima de R$ 8.472,00.</li>
    </ul>
    <p>Como é possível observar, esse campo político aparenta ser o menos afetado pela polarização política em torno de temas polêmicos.</p>
</div>
                """,
            unsafe_allow_html=True,
            )

    st.markdown("##### **Direita**")
    st.markdown(
                """
<div class="analysis-text">
    <ul>
        <li><b>Temas sociais:</b> Apresenta elevada rejeição ao sistema de cotas raciais em universidades e extrema discordância em relação ao direito ao aborto seguro. É majoritariamente favorável a facilitar a posse de armas e concorda de forma ampla com a pena de morte.</li>
        <li><b>Democracia e Instituições:</b> Apresenta ampla desconfiança em relação às urnas eletrônicas. É o espectro político que mais se considera nada satisfeito com a democracia no Brasil. É o campo que mais considera aceitável um regime de governo autoritário em algumas situações.</li>
        <li><b>Economia e Drogas:</b> A rejeição em relação a diminuir o lucro da Petrobrás para reduzir o preço dos combustíveis é a maior entre todos os campos políticos. É o espectro que apresenta a maior discordância em relação ao uso irrestrito da maconha.</li>
        <li><b>Notícias falsas:</b> Considera, em sua maioria, fácil identificar fake news. Apresenta a maior rejeição à ideia de que as empresas donas de redes sociais sejam responsáveis por impedir a divulgação de notícias falsas. Em relação ao controle de fake news nas redes socias para ocorrerem eleições justas, é o grupo que mais considera pouco importante ou nada importante. É o campo que mais considera que notícias falsas são compartilhadas porque as pessoas não sabem que a notícia é falsa.</li>
        <li><b>Perfil:</b> É o espectro predominante em todas as regiões da pesquisa, com maior dominância nas regiões Norte e Centro-Oeste. Possui a maior taxa de pessoas amarelas. Em relação à escolaridade, apresenta a maior taxa de entrevistados com ensino fundamental incompleto, a maior proporção de entrevistados que completaram o ensino médio e a menor porcentagem em relação aos que frequentaram o ensino superior. A religião evangélica é a predominante nesse grupo, com pequena vantagem em relação ao catolicismo. Possui a maior porcentagem de faixa etária de 40 a 49 anos, e uma taxa significativa de entrevistados de até 39 anos. Assim como a esquerda, predomina uma renda familiar abaixo de R$ 2.824,00.</li>
    </ul>
    <p>Esse é o espectro que mais apresenta divergências em relação aos entrevistados de esquerda, o que indica uma forte polarização entre esses dois campos políticos.</p>
</div>
                """,
            unsafe_allow_html=True,
            )