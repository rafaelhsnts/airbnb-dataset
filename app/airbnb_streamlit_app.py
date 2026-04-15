import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import folium as flm
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.airbnb_cleaner as cleaner
import kagglehub
from kagglehub import KaggleDatasetAdapter
from pywaffle import Waffle
from streamlit_folium import st_folium

path_raw = "src/data/raw/airbnb_raw.csv"
path_processed = "src/data/processed/airbnb_processed.csv"

st.set_page_config(layout="wide")

@st.cache_data
def load_raw():
    if not os.path.exists(path_raw):
        file_path = "Airbnb_Open_Data.csv"
        df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "arianazmoudeh/airbnbopendata",
        file_path
        )
        df.to_csv(path_raw, index=False)
    df_raw = pd.read_csv(path_raw)
    return df_raw

@st.cache_data
def load_processed():
    df_processed = pd.read_csv()
    return df_processed
df = load_raw()

@st.cache_data
def clean_data(df, verbose):
    df_clean, log = cleaner.df_cleaner(df, verbose)
    return df_clean, log



st.title("Airbnb Open Data", text_alignment="center")

st.markdown("---")

st.sidebar.title("Menu", text_alignment="center")
page = st.sidebar.radio("**Navegação**", ["Dataset", "Tratamento de dados", "Visualização"])

if page == "Dataset":
    st.header("Sobre os dados", text_alignment="center")
    st.markdown("<br>", unsafe_allow_html=True)

    # col1, col2, col3 = st.columns(3, )
    # col1.metric("Total de anúncios", df.shape[0], border=True)
    # col2.metric("Preço médio", df["Price In $"].mean().round(2), border=True)
    # col3.metric("Avaliação média", df["Review Rate Number"].mean().round(2), border=True)
    df = load_raw()
    st.markdown("A plataforma Airbnb, é um site/aplicativo disponível tanto para computadores, quanto para celulares, que surgiu com a proposta de facilitar a vida as pessoas que buscam alugar imóveis em diversas partes do mundo, e ao mesmo tempo que traz ao dono do imóvel uma forma prática de gerenciar a locação de seus imóveis. O app traz imagens internas do imóvel, além de descrições detalhadas que possibilitam uma avaliação a distância, e permite que com um único clique se proceda com a reserva.")
    st.subheader("Colunas")
    st.markdown("""
                **Id:** Número de identificação único reservado para um anúncio na plataforma Airbnb.  
                **Name:** Título do anúncio registrado na plataforma Airbnb.  
                **Host Id:** Número de identificação único reservado para cada anfitrião na plataforma Airbnb.  
                **Host Identity Verified:** indicação da confirmação da verificação de identidade para cada anfitrião na plataforma Airbnb.  
                **Host Name:** Primeiro nome do anfitrião.  
                **Neighbourhood Group:** Nome da região/zona da localização do imóvel.  
                **Neighbourhood:** Nome do bairro da localização do imóvel.  
                **Lat:** Número da Latitude da localização do imóvel.  
                **Long:** Número da longitude da localização do imóvel.  
                **Country:** Nome do país da localização do imóvel.  
                **Country Code:** Código do país da localização do imóvel.  
                **Instant Bookable:** Número indicador da disponibilidade de reserva instantâneo do imóvel.   
                **Cancellation Policy:** Indicador do grau de severidade da política de cancelamento do imóvel.  
                **Room Type:** Número indicador do tipo de quarto disponíbilizado no imóvel.  
                **Construction Year:** Ano de construção do imóvel.  
                **Price:** Valor pago pelo dia da locação do imóvel.  
                **Service Fee:** Valor total da taxa de serviço paga no ato da locação do imóvel.  
                **Minimum Nights:** Quantidade total de noites mínimas necessárias para a locação do imóvel.  
                **Number Of Reviews:** Quantidade total de avaliações recebida pelo imóvel na plataforma Airbnb.  
                **Last Review:** Data da última avaliação recebida pelo imóvel na plataforma Airbnb.  
                **Reviews Per Month:** Taxa de avaliações recebida por mes desde a criação do anúncio.  
                **Review Rate Number:** Valor médio da nota recebida pelo anúncio na plataforma Airbnb.  
                **Calculated Host Listings Count:** Número total de anúncios feito pelo mesmo anfitrião na plataforma Airbnb.  
                **Availability 365:** Número total de dias até a disponibilização do imóvel para locação.  
                **House Rules:** Regras únicas para a utilização do imóvel editadas pelo anfitrião.  
                **License:** Código único da licença do imóvel informada pelo anfitrião.  
    """)
    st.markdown(f"""
                O dataset utilizado é o **Airbnb_Open_Data.csv**, disponível na base de dados do <a href="https://www.kaggle.com/datasets/arianazmoudeh/airbnbopendata?select=Airbnb_Open_Data.csv" style="text-decoration:none; font-weight:bold">Kaggle</a>, ele possui **{df.shape[0]} linhas** e **{df.shape[1]} colunas**.
                """, unsafe_allow_html=True)
    st.dataframe(df.head(100))

elif page == "Tratamento de dados":
    st.header("Limpeza dos dados", text_alignment="center")
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Valores antes da limpeza")
    lines = df.shape[0]
    st.dataframe(df.isna().sum().rename("Total de valores nulos"))
    st.info(f"Existem **{df.shape[0]} linhas**. Foram removidas **{lines - df.shape[0]} linhas.**")

    df, logs = clean_data(df, verbose=True)
    with st.expander("Logs de tratamento:"):
        for log in logs:
            log = log.replace("'", "**").replace("$", "\$").replace("\n", "<br>").replace("NaN", "**NaN**").replace(".", '"."').replace(",", '","')
            st.markdown(log, unsafe_allow_html=True)

    st.subheader("Valores após a limpeza")
    st.dataframe(df.isna().sum().rename("Total de valores nulos"))
    st.info(f"Existem **{df.shape[0]} linhas**. Foram removidas **{lines - df.shape[0]} linhas.**")

elif page == "Visualização":

    def create_corr(df):
        list_not_corr = ["Id", "Host Id", "Neighbourhood", "Lat", "Long"]
        df_corr = df.drop(columns=list_not_corr, errors="ignore")
        cols_cat = [c for c in ["Host Identity Verified", "Neighbourhood Group", "Instant Bookable", "Cancellation Policy", "Room Type"] if c in df_corr.columns]
        df_corr = pd.get_dummies(df_corr, columns=cols_cat, drop_first=True, dtype=int)
        corr = df_corr.select_dtypes(include="number").corr()
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(corr, cmap="coolwarm", ax=ax, square=True)
        plt.xticks(rotation=45, ha="right")
        st.markdown("A análise do mapa de calor demonstra que **Service Fee In \\$** apresenta uma correlação praticamente perfeita com **Price In \\$**, indicando uma relação linear praticamente constante entre elas, o que garante que **Service Fee In \\$** possa ser descartada. Além disso é possivel notar que **Price In \\$**, **Host Identity Verified_Verified** e **Instant Bookable_True** apresentam baixíssima ou quase nenhuma correlação com as demais variáveis, indicando a ausencia de relações lineares fortes entre elas.", text_alignment="justify")
        return fig

    def create_boxgraph(df):
        box_list = df.drop(columns=["Id", "Host Id", "Lat", "Long"]).select_dtypes(include="number")
        fig, axs = plt.subplots(4, 2, figsize=(15, 15))
        sns.boxplot(data=box_list.iloc[:, 0], orient="h", ax=axs[0][0])
        sns.boxplot(data=box_list.iloc[:, 1], orient="h", ax=axs[0][1])
        sns.boxplot(data=box_list.iloc[:, 2], orient="h", ax=axs[1][0])
        sns.boxplot(data=box_list.iloc[:, 3], orient="h", ax=axs[1][1])
        sns.boxplot(data=box_list.iloc[:, 4], orient="h", ax=axs[2][0])
        sns.boxplot(data=box_list.iloc[:, 5], orient="h", ax=axs[2][1])
        sns.boxplot(data=box_list.iloc[:, 6], orient="h", ax=axs[3][0])
        sns.boxplot(data=box_list.iloc[:, 7], orient="h", ax=axs[3][1])
        st.markdown("""A variável **Minimum Nights** representa a quantidade de dias corridos, e existem entre seus dados, valores superiores a 730 dias, já que a plataforma Airbnb é conhecida por trabalhar com alugueis curtos, entretanto não há evidências suficientes que permitam confirmar a existencia de algum erro, somente com essa informação.
        As variáveis **Number of Reviews** e **Review Per Month** demonstram que dentre os imoveis disponiveis para locação, existem aqueles que recebem uma quantidade elevada de reviews, o que pode indicar que estes sejam extremamente requisitados, indicando alta rotatividade, atravéz de alugueis rápidos e consecutivos, e ou a maior taxa de aceitação deste imóvel. Considerando o segundo caso, pode-se verificar se a aceitação se deve a localidade, ao tipo de quarto ofertado ou ao preço sugerido para locação, e então desenvolver um conjunto de melhores práticas, a ser disponibilizado aos anfitriões dos imóveis, a fim de elevar a expectativa do cliente em imóveis pouco procurados.
        Já a variável **Calculated Host Listings Count** demonstra que alguns anfitriões possuem uma quantidade elevada de imóveis listados. Esses dados revelam uma forma adicional para expandir a disponibilidade de imóveis, atravéz de anfitriões que já anunciam na plataforma, o que pode ser feito por meio de incentivos a novas listagens, como planos premium para anfitriões com imóves extras anunciados, ou planos com desconto progressivos para cada nova listagem.""", text_alignment="justify")
        return fig

    def create_neiggroupgraph(df):
        fig, axs = plt.subplots(3, 2, figsize=(15, 18))
        ng_cp = df.groupby(["Neighbourhood Group"])["Cancellation Policy"].value_counts().reset_index(name="Count")
        sns.barplot(data=ng_cp, x="Count", y="Neighbourhood Group", hue="Cancellation Policy", ax=axs[0][0])
        axs[0][0].set_title("Cancellation Policy x Neighbourhood Group", weight="bold")
        axs[0][0].set_xlim(0, ng_cp["Count"].max()*1.1)
        ng_rt = df.groupby(["Neighbourhood Group"])["Room Type"].value_counts().reset_index(name="Count")
        sns.barplot(data=ng_rt, x="Neighbourhood Group", y="Count", hue="Room Type", ax=axs[0][1])
        axs[0][1].set_title("Neighbourhood Group x Room Type", weight="bold")
        axs[0][1].set(yscale="log", ylabel="Count (log scale)")
        axs[0][1].legend(loc="upper right", bbox_to_anchor=(1, 1.01))
        ng_cy = df.loc[:, ["Neighbourhood Group", "Construction Year"]]
        sns.boxenplot(data=ng_cy, x="Neighbourhood Group", y="Construction Year", ax=axs[1][0])
        axs[1][0].set_title("Neighbourhood Group x Construction Year", weight="bold")
        ng_p = df.loc[:, ["Neighbourhood Group", "Price In $"]]
        sns.violinplot(data=ng_p, y="Price In $", hue="Neighbourhood Group", ax=axs[1][1])
        axs[1][1].set_title("Neighbourhood Group x Price In $", weight="bold")
        axs[1][1].set_ylim(ng_p["Price In $"].min()*-4, ng_p["Price In $"].max()*1.4)
        axs[1][1].legend(loc="upper center", ncols=3)
        ng_rrn = df.loc[df["Number Of Reviews"] >= 10]
        ng_rrn = ng_rrn.groupby(["Neighbourhood Group"])["Review Rate Number"].value_counts().reset_index(name="Count")
        sns.barplot(data=ng_rrn, x="Neighbourhood Group", y="Count", hue="Review Rate Number", ax=axs[2][0])
        axs[2][0].set_title("Review Rate Number x Neighbourhood Group", weight="bold")
        ng_id = df.groupby(["Neighbourhood Group"])["Id"].nunique()
        ng_id.plot(kind="pie", autopct="%1.1f%%", pctdistance=1.15, labels=None, ax=axs[2][1])
        axs[2][1].legend(ng_id.index, title="Neighbourhood Group", loc="center left", bbox_to_anchor=(1.0, .5))
        axs[2][1].set(ylabel=None)
        axs[2][1].set_title("Total Airbnb x Neighbourhood Group", weight="bold")
        st.markdown("A variável **Neighbourhood Group** agrupa em apenas 5 regiões, um total de 224 bairros, permitindo uma análise limpa da distribuição geográfica dos anúncios do Airbnb. Quando comparada com **Room Type**, é possível verificar que os tipos de Airbnb mais disponibilizados para locação, são **Private room** e **Entire home/apt**, essas informações sugerem que talvez a baixa oferta entre quartos do tipo **Hotel room** seja uma lacuna de mercado, principalmente em regiões como o Bronx e Staten Island, locais onde quartos desse tipo são inexistentes. Os imóveis disponibilizados para locação no Airbnb dependem de anúncios de terceiros, então é impossível para a plataforma disponibilizar todas as modalidades de quartos, sendo assim, seria interessante aumentar a disponibilidade de quartos de todos os tipos, então oferecer valores diferenciados ou ofertas para anúncios de **Room Type** pouco ofertados, poderia incentivar esse aumento.", text_alignment="justify")
        return fig

    def create_priceroom(df):
        fig, ax = plt.subplots(figsize=(16, 8))
        df2 = df.copy()
        lim = np.linspace(df2["Price In $"].min(), df2["Price In $"].max(), 13)
        df2["Price Range"] = pd.cut(df2["Price In $"], bins=lim, include_lowest=True)
        rt_p = df2.groupby(["Room Type"])["Price Range"].value_counts().reset_index(name="Count")
        sns.barplot(data=rt_p, x="Room Type", y="Count", hue="Price Range", palette="magma", ax=ax)
        ax.set(yscale="log", ylabel="Count (log scale)", ylim=(1, rt_p["Count"].max()*5))
        ax.legend(title="Price Range", loc="upper center", ncols=6)
        st.markdown("A distribuição do preço na variável **Room Type**, é relativamente semelhante, por esse motivo não é possível explicar o preço a partir do tipo de quarto. Porém algo que esse gráfico revela, é que **Hotel room** é o tipo de quarto menos disponibilizado para locação, sugerindo uma limitação na oferta desse tipo de segmento, o que reforça a existência de uma margem para a ampliação dessa modalidade de quarto na plataforma.", text_alignment="justify")
        return fig

    def create_pywaffle(df):
        df2 = df.groupby(["Neighbourhood Group"])["Review Rate Number"].mean()
        fig = plt.figure(FigureClass=Waffle, figsize=(15, 5), rows=18, columns=50, values=df2, legend = {"labels": [f"{i} ({v})" for i, v in zip(df2.index, df2.values)], "loc":"lower center", "bbox_to_anchor":(.5, -.1), "ncols":5})
        plt.title("Neighbourhood Group with the best-rated Airbnbs.", fontweight="bold", pad=10)
        st.markdown("O gráfico de waffle está representando visualmente os valores médios da variável **Review Rate Number** em cada uma das regiões de **Neighbourhood Group**. Nessa análise, as médias são extremamente próximos, então a diferença visual não é muito aparente, porém é possível notar que em média as regiões do Bronx e Staten Island, são as que receberam as maiores notas, já a região de Manhattan a menor. Vale notar que Bronx e Staten Island são cidades menores e com menos imóveis para locação, então consequentemente o volume de avaliações é consideravelmente menor, gerando médias mais sensíveis a variações e menos consistentes, já a região de Manhattan é a que detém a maior quantidade, resultando em médias com baixa oscilação. Porém cabe uma verificação aprofundada, para averiguar se a menor avaliação média em cidades maiores, não se deve a uma menor satisfação dos usuários nessa região.", text_alignment="justify")
        return fig

    def create_worldmap(df):
        latlong_mean = df.groupby("Neighbourhood").agg({"Lat":"mean", "Long":"mean", "Neighbourhood":"count"}).rename(columns={"Neighbourhood":"Count"}).reset_index()
        # latlong_mean = df.groupby(["Neighbourhood"])[["Lat", "Long"]].agg("mean").reset_index()
        # latlong_mean["Count"] = df.groupby(["Neighbourhood"])["Lat"].count().values
        lat_mean = df["Lat"].mean()
        long_mean = df["Long"].mean()
        max_count = latlong_mean["Count"].max()
        def scale_radius(x):
            return 5 + (x / max_count) * 30
        carto = flm.Map(location=[lat_mean, long_mean], zoom_start=11, tiles="CartoDB positron")
        for _, row in latlong_mean.iterrows():
            popup_html=f"""
                    <div style='max-width:200px; display:flex; flex-direction:column; align-items:center; white-space:nowrap'>
                        <b>{row['Neighbourhood']}</b><br>
                        <span>Count: ({row['Count']})</span>
                    </div>
                    """
            popup=flm.Popup(popup_html)
            flm.CircleMarker(
                location=[row["Lat"], row["Long"]],
                color=None,
                weight=0,
                radius=scale_radius(row["Count"]),
                fill=True,
                fill_color="#4444aa",
                fill_opacity=.5,
                popup=popup
            ).add_to(carto)
        st.markdown("O worldmap do folium foi usado com o objetivo de exibir a quantidade de imóveis listados na plataforma da Airbnb em cada um dos bairros. Com essa visualização geográfica é possível confirmar a existencia de varios bairros com poucos imóveis listados, sugerindo que nessas localidades existem margem para que mais imóveis possam ser anunciados. Essa informação, reforça a necessidade de politicas de incentivo, para que mais anfitriões interessem-se pelos serviços ofertados pela plataforma. A plataforma deve tornar cada vez mais atrativa, pratica e barata, a listagem de imóveis, tanto para novos anfitriões, quanto para novas listagens ofertadas pelos já existentes.", text_alignment="justify")
        st_folium(carto, use_container_width=True, height=500)

    df, _ = clean_data(df, verbose=False)
    st.header("Visualizações", text_alignment="center")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,2,1])

    graph = st.sidebar.radio("**Análise**", ["Correlação", "Análise de outliers", "Distribuição por região", "Preço por tipo de quarto", "Avaliação por região", "Distribuição geográfica"])

    if graph == "Correlação":
        fig = create_corr(df)
        with col2:
            st.pyplot(fig)
        plt.close(fig)
    elif graph == "Análise de outliers":
        fig = create_boxgraph(df)
        with col2:
            st.pyplot(fig)
        plt.close(fig)
    elif graph == "Distribuição por região":
        fig = create_neiggroupgraph(df)
        with col2:
            st.pyplot(fig)
        plt.close(fig)
    elif graph == "Preço por tipo de quarto":
        fig = create_priceroom(df)
        with col2:
            st.pyplot(fig)
        plt.close(fig)
    elif graph == "Avaliação por região":
        fig = create_pywaffle(df)
        with col2:
            st.pyplot(fig)
        plt.close(fig)
    elif graph == "Distribuição geográfica":
        create_worldmap(df)

st.markdown("---")
