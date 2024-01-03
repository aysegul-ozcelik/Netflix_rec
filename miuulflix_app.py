import streamlit as st
import plotly.express as px
import joblib
import pandas as pd
import requests
import pickle
import base64
from streamlit_extras.colored_header import colored_header
from streamlit_extras.app_logo import add_logo
from PIL import Image

from plotly import graph_objects as go
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
st.set_page_config(layout="wide")

@st.cache_data
def get_data():
    df = pd.read_csv('data/netflix_titles.csv')
    df_1 = df[['title', 'type', 'listed_in', 'description']]
    return df_1





st.sidebar.image('media/fontbolt.png',width=150,use_column_width=False)


st.image("media/fontbolt.png", width=150, use_column_width=False)  # Genişlik: 100 piksel

st.header(":red[**RECOMMENDATION**]")

tab_home, tab_vis,tab_recommend, mag = st.tabs(["Ana Sayfa", "Grafikler","Ne İzlesek", "Birazcık Magazin"])


# TAB HOME

column_miuulflix, column_dataset = tab_home.columns([2,1], gap="large")

column_miuulflix.subheader("İçerik Tabanlı Tavsiye Sistemi: Netflix Veri Seti ")

column_miuulflix.markdown("""Tavsiye sistemleri, kullanıcıların tercihlerine göre içerikleri öneren yapay zeka tabanlı sistemlerdir. İçerik tabanlı tavsiye sistemleri, kullanıcının geçmiş tercihlerine veya belirli bir öğe hakkındaki bilgilere dayalı olarak benzer öğeleri önermek için kullanılan bir türdür. Bu sistemler, kullanıcının zevklerini ve ilgi alanlarını anlamak için içerik özelliklerini analiz eder ve buna göre öneriler sunar.


Netflix gibi bir platformda, İçerik Tabanlı Bir Tavsiye Sistemi (Content-Based Recommendation System), kullanıcıların belirli bir dizi veya film adını girdiğinde, o içeriğe benzer özelliklere sahip diğer yapıtları önerir. Örneğin, veri setinde bulunan 'Movies' ve 'TV Series' değişkenleri, her bir içeriğin özelliklerini içerir.


İçerik Tabanlı Öneri Sistemi, kullanıcının girdiği dizi veya film adını analiz ederek, o içeriğin benzersiz özelliklerini belirler. Daha sonra bu özelliklere sahip diğer içerikleri bulur ve kullanıcıya beğenebileceği birkaç dizi ve film önerisinde bulunur.

Örneğin, kullanıcı "Stranger Things" dizisini aradığında, sistem bu dizinin özelliklerine dayalı olarak bilgi alır ve aynı türde veya benzer özelliklere sahip diğer yapıtları, örneğin " Nightflyers ", "The OA", "Manifest", “Helix”, “Star-Crossed” gibi içerikleri kullanıcıya önerir.""")





column_dataset.image("media/image_processing20200922-8646-1jjgp13.gif")

column_dataset.markdown("""İçerik tabanlı tavsiye sistemleri, kullanıcıların tercihlerini anlamak ve benzer içerikleri keşfetmelerine yardımcı olmak için veri analizi ve özellik çıkarımı gibi teknikleri kullanır. Bu sistemler, kullanıcı deneyimini zenginleştirirken aynı zamanda kullanıcının yeni içerikler keşfetmesine olanak tanır.""")

df_1 = get_data()
tab_home.dataframe(df_1, width=4000)

import base64

def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio controls autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        tab_home.markdown(
            md,
            unsafe_allow_html=True,
        )

autoplay_audio("media/netflixintr_n1r71833.mp3")






#magazin
mag.image("media/reed-hastings-2-1.png")
mag.subheader("Her şey, Hastings’in Blockbuster’dan bir film kiralaması ve iade etmeyi unutması ile başlıyor.")
mag.markdown("""Reed Hastings ve Marc Randolph tarafından 1997 yılında kurulan Netflix’in bu başarısının ardında tabii ki anlatmaya değer bir hikâye yatıyor.
 
O dönemin film kiralama firması olan Blockbuster’dan Apollo 13 filmini kiralayan Hastings, filmi zamanında iade etmediği için  ceza ödemek zorunda kalıyor ve bu, onu “acaba film kiralama işinde bu soruna çözüm getirecek başka bir iş modeli oluşturulabilir mi?” diye düşündürtüyor.

Hastings’in aklına spor salonlarında kullanılan üyelik sisteminin film kiralama sektörü için uygulanabileceği geliyor.

Bu esnada Randolph ise Amazon’un iş modelinden etkilenerek internet ortamında var olacak bir iş modeli üzerine düşünüyordu.

Randolph ve Hastings bir araba yolculuğu esnasında nasıl bir girişim yapabileceklerine dair fikirler üretirken, akıllarındaki iki fikri birleştirerek Amazon'a benzeyen ama film kiralama üzerine bir girişim fikrini ortaya atıyor.
A
ncak o dönemki VHS kasetlerin çok dayanıklı olmamasından ve kargo esnasında kırılabilecek olmasından dolayı bu fikri reddediyorlar.

İlerleyen yıllarda DVD’nin ortaya çıkışı ile ilk başta reddetmiş oldukları fikri hayata geçirmek için kolları sıvıyorlar.

DVD, VHS kasetlere göre hem çok daha ince hem de çok daha dayanıklı olduğu için online film kiralama fikrini reddetmelerine sebep olan sorun DVD ile ortadan kalkmış oluyor ve Netflix’in macerası başlıyor.

1997 yılında hayal ettikleri işi internete atıfta bulunan “net” ve film anlamında gelen “flick” kelimesine atıfta bulunan iki kelimenin birleşiminden oluşturdukları “Netflix” ismiyle resmi olarak faaliyete geçiriyorlar.

Oluşturdukları abonelik tabanlı iş modeli ile müşterilerin aylık bir ücret karşılığında sınırsız DVD kiralamasına izin veren bu girişimde DVD’ler müşterilere posta yoluyla gönderiliyor ve müşteri, DVD’yi iade ettiğinde diğer müşteri o DVD’yi kiralayabiliyordu.

Hastings, 2000 yılında Netflix’i Blockbuster’a 50 milyon dolara satmayı teklif etti.

Kazandığı popülerliğe rağmen o dönem kâr edemeyen Netflix, Blockbuster tarafından satın alınmadı. Tabii Blockbuster yetkilileri bunun çok büyük bir yanlış olduğunu daha sonra çok acı bir şekilde anlayacaklardı…

Netflix, 2003 yılında kâr elde etmeye başlayarak 1.000.000 aboneye ulaştı ve Blockbuster, yaptığı hatayı fark ederek Netflix’e benzer olarak online DVD kiralama işine girmeye çalışsa da bu çabaları büyük bir başarısızlık ile sonuçlandı.

2010 yılında iflas eden Blockbuster, zamanında önlerine altın tepside sunulan teklifi değerlendirmeyerek çok kritik bir hata yapmış oldu.

Netflix’in, 2007 yılında yayıncılık işine kolları sıvaması ile birlikte durdurulamaz büyümesi başladı.

2013 yılında Netflix, ilk kez kendi orijinal televizyon dizisi olan House of Cards’ı yayınladı. Yani artık Netflix yapım şirketlerinden dizi satın almanın yanı sıra, kendi dizisini de çeker hale geldi. Sayısız ödül alan House of Cards’ın da etkisiyle 2013’ün sonunda Netflix’in hisse senetleri 3 kat değer kazandı. 2020 yılı itibariyle şirketin üye sayısı 200 milyona yaklaşmış durumda.

ABD’li ünlü yatırımcı Jim Cramer şu anda dünyada yapay zekadan en iyi Netflix’in istifade ettiğini belirtiyor. Makine öğrenmesinin yanı sıra, kullanıcının interneti yavaşladığında Netflix’in Dynamic Optimizer adlı algoritması görüntüyü sıkıştırarak kaliteden ödün vermiyor. İnternet hızınız düşük olsa bile Netflix içeriklerini kaliteli bir şekilde izleyebiliyorsunuz. Hatta görüntününü hızlı değiştiği sahnelerde bu algoritma iki kat daha fazla çalışarak seyir keyfine zeval getirmemesi gerektiğini bile biliyor. Bu nedenle her türlü koşulda görüntü kalitesini maksimum düzeyde tutmak için yapay zekadan çok iyi faydalanıyor.

10 Temmuz 2020’de Netflix, piyasa değeri bakımından en büyük eğlence/medya şirketi oldu.
""")

mag.image("media/WfVA.gif")






#tab_vis


import streamlit as st



tab_vis.columns([1], gap="large")



#st.audio(data)
#tab_vis.audio("netflixintr_n1r71833.mp3")

#graf1

tab_ülke=st.sidebar
selected_tabülke= tab_ülke.selectbox("Funnel Chart", ["Lütfen Seçiniz","Ülkelerin Film ve Dizi Sayıları"])



fig = go.Figure(go.Funnel(
    y=["United States", "India", "United Kingdom", "Japan","Turkey"],
    x=[2818, 972, 419, 245,105],
    textposition="inside",
    textinfo="value",
    opacity=0.65,
    marker={"color": ["firebrick", "lightsalmon", "tan", "teal","red"],
            "line": {"width": [5, 5, 4, 4, 3, 3], "color": ["wheat", "wheat", "blue", "wheat","white"]}},
    connector={"line": {"color": "royalblue", "dash": "dot", "width": 3}})
)

fig.update_layout(height=500, width=900)

if selected_tabülke == "Lütfen Seçiniz":
    "         "

elif selected_tabülke == "Ülkelerin Film ve Dizi Sayıları":
     tab_vis.plotly_chart(fig)






#graf2


@st.cache_data
def generate_treemap(selected_option):
    df = pd.read_csv('data/netflix_titles.csv')
    netflix_Turkey = df[df['country'] == 'Turkey']
    nannef = netflix_Turkey.dropna()

    fig1 = None
    if selected_option == "Türk Yönetmenlerin Film ve Dizileri":
        fig1 = px.treemap(nannef, path=['country', 'director', 'title'], width=1500,
                          height=750, color='director', hover_data=['director', 'type'],
                          color_continuous_scale='Purples')

    return fig1


# Streamlit arayüzü
selected_tabyonetmen = st.sidebar.selectbox("Treemap Grafiği",
                                            ["Lütfen Seçiniz", "Türk Yönetmenlerin Film ve Dizileri"])
filtered_data = generate_treemap(selected_tabyonetmen)

if filtered_data is not None:
    tab_vis.plotly_chart(filtered_data)

tab_yönetmen = st.sidebar


#grafik3
tabpasta = st.sidebar
selected_tabpasta = tabpasta.selectbox("Pie Chart", ["Lütfen Seçiniz","Film ve Diziler"])



#df = pd.read_csv('data/netflix_titles.csv')

#@st.cache_data
#def plot_distribution_pie_chart(dataframe, figure_size=(5, 5),bg_color="black"):

#    labels = ['Movie', 'TV show']
#   size = dataframe['type'].value_counts()
#    colors = plt.cm.Wistia(np.linspace(0, 1, 2))
#    explode = [0, 0.1]

    # Set the figure size
#    plt.figure(figsize=figure_size)

#    plt.pie(size, labels=labels, colors=colors, explode=explode, shadow=True, startangle=90, autopct='%1.1f%%')
#    plt.title('Türlerin Dağılımı', fontsize=5)
#    plt.legend()

#    return plt


if selected_tabpasta == "Lütfen Seçiniz":
    "         "

elif selected_tabpasta == "Film ve Diziler":
    tab_vis.image("media/Pie.png")

#graf4

barchart_path = "media/barchartmv.png"
matrixchart_path = "media/matrixchartmv.png"


tab_eksik = st.sidebar
selected_tabeksik = tab_eksik.selectbox("Eksik Veri Grafikleri", ["Lütfen Seçiniz","Bar Chart", "Matrix"])

if selected_tabeksik == "Lütfen Seçiniz":
    "         "

elif selected_tabeksik == "Bar Chart":
    # Show Bar Chart
    tab_vis.image(barchart_path, use_column_width=True)

elif selected_tabeksik == "Matrix":
    # Show Matrix
    tab_vis.image(matrixchart_path, use_column_width=True)







#bar_fig, ax = plt.subplots(figsize=(10, 5))
#msno.bar(df, color="black", ax=ax)

#buf = io.BytesIO()
#bar_fig.savefig(buf, format='png')
#buf.seek(0)

#tab_vis.image(buf)


#fig, ax = plt.subplots()
#msno.matrix(df, ax=ax)
#tab_vis.pyplot(fig)  # Streamlit üzerinde görseli gösterin


#fig2, ax2 = plt.subplots()
#msno.heatmap(df, ax=ax2)
#st.pyplot(fig2)  # Streamlit üzerinde görseli gösterin

#graf5

tab_filmdizi = st.sidebar
selected_tabfilmdizi = tab_filmdizi.selectbox("İzleyici Kitlesine Göre Film-Dizi Dağılımları", ["Lütfen Seçiniz","Film","Dizi"])

if selected_tabfilmdizi == "Lütfen Seçiniz":
    "         "

elif selected_tabfilmdizi == "Film":
     tab_vis.image("media/distirbitionofmovierating.png")

elif selected_tabfilmdizi == "Dizi":
     tab_vis.image("media/distributionooftvshowrating.png")


#graf6


tab_relationcore = st.sidebar
selected_relationcore = tab_relationcore.selectbox("Benzerliklerine Göre Korelasyon Analizi",["Lütfen Seçiniz","Film","Dizi"])

if selected_relationcore == "Lütfen Seçiniz":
    "         "

elif selected_relationcore == "Film":
     tab_vis.image("media/Relationmovie.png")

elif selected_relationcore == "Dizi":
    tab_vis.image("media/Relationseries.png")



#tab_recommend.subheader("Miuulflix")






movies_series = pickle.load(open("pkl/movies_list.pkl", 'rb'))
similarity = pickle.load(open("pkl/similarity.pkl", 'rb'))
movies_list = movies_series['title'].values

tab_recommend.header(":red[Miuulflix Recommender System]")
selected_movie_series = tab_recommend.selectbox("Select Movie or Series", movies_series)

def recommend(movie):
    index=movies_series[movies_series['title']==movie].index[0]
    distance = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda vector:vector[1])
    recommend_movies_series=[]
    for i in distance[1:4]:
        recommend_movies_series.append(movies_series.iloc[i[0]].title)
    return recommend_movies_series
if tab_recommend.button("Show Recommend"):
    movieseries_name=recommend(selected_movie_series)
    col1,col2,col3=st.columns(3)
    with col1:
        tab_recommend.text(movieseries_name[0])
    with col2:
        tab_recommend.text(movieseries_name[1])
    with col3:
        tab_recommend.text(movieseries_name[2])






tab_recommend.markdown("""Yukarıdaki kodlarda, CountVectorizer kullanılarak metin verileri üzerinde vektörleme işlemi gerçekleştirilmiş ve ardından cosine_similarity ile benzerlik skorları hesaplanmış. Bu işlem genellikle metin verilerinin anlamsal benzerliklerini ölçmek için kullanılır.

CountVectorizer, metin verilerini vektörlere dönüştürmek için kullanılan bir yöntemdir. max_features parametresi ile en fazla kaç özelliğin kullanılacağını belirleyebilirsiniz.

cosine_similarity fonksiyonu, vektörler arasındaki kosinüs benzerliğini hesaplar. Bu benzerlik, vektörler arasındaki açıyı ölçer ve benzerlik skorlarını sağlar.

Bu kodlar, bir önceki adımda vektörize edilen metin verileri üzerinde benzerlik hesaplamak için kullanılmış gibi görünüyor. Eğer daha fazla işlem yapmak veya bu benzerlik matrisini kullanarak bir öneri sistemi oluşturmak istiyorsanız, benzerlik matrisini kullanarak ilgili işlemleri gerçekleştirebilirsiniz. Örneğin, belirli bir film veya içerik için en benzer diğer içerikleri bulmak gibi. Bu benzerlik skorlarını kullanarak tavsiye sistemleri geliştirebilirsiniz.
""")


