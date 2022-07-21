import streamlit as st
import xlrd
import nltk
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import string
import re
import tweepy
import data_api
import pandas as pd
import time
nltk.download('punkt')

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(
    page_title="Klasifikasi Kepribadian",
    page_icon="images/favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)


@st.cache(suppress_st_warning=True)
def naive_klasifikasi(a):
    book = xlrd.open_workbook("dataseet/dataset_final2.xls")
    sheet = book.sheet_by_name('Sheet1')
    dokumen = []
    ext = []
    neu = []
    agr = []
    con = []
    opn = []
    for n in range(1, sheet.nrows):
        text = sheet.cell(n, 0).value
        dokumen.append(text)
        ext1 = sheet.cell(n, 1).value
        ext.append(ext1)
        neu1 = sheet.cell(n, 2).value
        neu.append(neu1)
        agr1 = sheet.cell(n, 3).value
        agr.append(agr1)
        con1 = sheet.cell(n, 4).value
        con.append(con1)
        opn1 = sheet.cell(n, 5).value
        opn.append(opn1)
    df = pd.DataFrame({'text': dokumen, 'ext': ext, 'neu': neu,
                       'agr': agr, 'con': con, 'opn': opn})
    corpus = df['text'].values
    y = df[['ext', 'neu', 'agr', 'con', 'opn']]
    X_train, X_test, y_train, y_test = train_test_split(
        corpus, y, train_size=0.8, test_size=0.2, random_state=1)
    counter = CountVectorizer()
    counter.fit(X_train, y_train)
    X_train = counter.transform(X_train)
    X_test = counter.transform(X_test)
    data_pengguna = counter.transform(a)
    clf = BinaryRelevance(MultinomialNB(alpha=0.6, fit_prior=True))
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)
    clf_predictions = clf.predict(data_pengguna).toarray()
    st.write('Accuracy score :', accuracy_score(y_test, prediction)*1000)
    return clf_predictions


def run_klasifikasi(dat):
    b = naive_klasifikasi(dat)
    st.markdown('**Hasil Klasifikasi Kepribadian :**')
    h_ext = []
    c_ext = []
    h_neu = []
    h_agr = []
    h_con = []
    h_opn = []
    if b[0][0] == 1:
        h_ext.append("Ya")
        c_ext = "senang berhubungan dan bersosialisasi dengan orang lain"
    if b[0][0] == 0:
        h_ext.append("Tidak")
        c_ext = "menghindari hubungan dan sosialisasi terhadap orang lain"
    if b[0][1] == 1:
        h_neu.append("Ya")
        c_neu = "Mudah mengalami perubahan suasana hati dan terpengaruh emosi negatif"
    if b[0][1] == 0:
        h_neu.append("Tidak")
        c_neu = "tidak mudah terpengaruh oleh emosi negatif"
    if b[0][2] == 1:
        h_agr.append("Ya")
        c_agr = "senang berkerja sama dengan orang lain"
    if b[0][2] == 0:
        h_agr.append("Tidak")
        c_agr = "kurang peduli terhadap orang lain dan kurang memiliki empati"
    if b[0][3] == 1:
        h_con.append("Ya")
        c_con = "hidup teratur dan mempunyai motivasi tinggi"
    if b[0][3] == 0:
        h_con.append("Tidak")
        c_con = "Santai dan kurang termotivasi untuk sukses"
    if b[0][4] == 1:
        h_opn.append("Ya")
        c_opn = "Menyukai perubahan dan mempunyai keingintahuan tinggi terhadap hal-hal baru."
    if b[0][4] == 0:
        h_opn.append("Tidak")
        c_opn = "tidak menyukai perubahan dan keingintahuan yang kurang terhadap hal-hal baru."

    if b[0][0] == 0 and b[0][2] == 1:
        combi1 = "Berhati lembut, Menyenangkan, Patuh,"
    elif b[0][0] == 1 and b[0][2] == 0:
        combi1 = "Kasar, Spontan,"
    else:
        combi1 = ""

    if b[0][0] == 0 and b[0][3] == 1:
        combi2 = "Teliti, Waspada, Tepat Waktu,"
    elif b[0][0] == 1 and b[0][3] == 0:
        combi2 = "Ceroboh, Susah diatur,"
    else:
        combi2 = ""

    if b[0][0] == 0 and b[0][1] == 1:
        combi3 = "Tidak bersemangat, Sederhana,"
    elif b[0][0] == 1 and b[0][1] == 0:
        combi3 = "Sangat tegang,"
    else:
        combi3 = ""

    if b[0][0] == 0 and b[0][4] == 1:
        combi4 = "suka introspeksi, Suka merenung,"
    elif b[0][0] == 1 and b[0][4] == 0:
        combi4 = "Jahat, sombong,"
    else:
        combi4 = ""

    if b[0][2] == 0 and b[0][0] == 1:
        combi5 = "Mendominasi, Tegas,"
    elif b[0][2] == 1 and b[0][0] == 0:
        combi5 = "Pemalu, Lembut, Penurut,"
    else:
        combi5 = ""

    if b[0][2] == 0 and b[0][3] == 1:
        combi6 = "Berwatak keras, Tegas, Tidak tergesa-gesa,"
    else:
        combi6 = ""

    if b[0][2] == 0 and b[0][1] == 1:
        combi7 = "Tidak emosional, Maskulin,"
    elif b[0][2] == 1 and b[0][1] == 0:
        combi7 = "Emosional, mudah tertipu,"
    else:
        combi7 = ""

    if b[0][2] == 0 and b[0][4] == 1:
        combi8 = "Individualistis, Eksentrik,"
    elif b[0][2] == 1 and b[0][4] == 1:
        combi8 = "Sederhana, Bergantung pada orang lain,"
    else:
        combi8 = ""

    if b[0][3] == 0 and b[0][0] == 1:
        combi9 = "Menyukai keributan, Nakal,"
    elif b[0][3] == 1 and b[0][0] == 0:
        combi9 = "Pendiam, Terkendali, Serius,"
    else:
        combi9 = ""

    if b[0][3] == 0 and b[0][2] == 1:
        combi10 = "Toleran, Penyayang, Penurut,"
    elif b[0][3] == 1 and b[0][2] == 0:
        combi10 = "Keras, kaku,"
    else:
        combi10 = ""

    if b[0][3] == 0 and b[0][1] == 1:
        combi11 = "Santai,"
    elif b[0][3] == 1 and b[0][1] == 0:
        combi11 = "Mementingkan kepentingan pribadi,"
    else:
        combi11 = ""

    if b[0][3] == 1 and b[0][4] == 0:
        combi12 = "Keras kepala, Suka melanggar peraturan,"
    else:
        combi12 = ""

    if b[0][1] == 0 and b[0][0] == 1:
        combi13 = "Genit, Emosi meledak-ledak, Detail,"
    elif b[0][1] == 1 and b[0][0] == 0:
        combi13 = "Tenang, Sabar, Tidak mudah marah,"
    else:
        combi13 = ""

    if b[0][1] == 0 and b[0][2] == 1:
        combi14 = "Sentimental, Penyayang, Sensitif,"
    elif b[0][1] == 1 and b[0][2] == 0:
        combi14 = "Tidak peka, tidak penyayang,"
    else:
        combi14 = ""

    if b[0][1] == 1 and b[0][3] == 0:
        combi15 = "Mudah puas, Berprinsip,"
    else:
        combi15 = ""

    if b[0][1] == 0 and b[0][4] == 1:
        combi16 = "Mengikuti naluri,"
    elif b[0][1] == 1 and b[0][4] == 0:
        combi16 = "Tidak berfikir secara matang, Tidak peka,"
    else:
        combi16 = ""

    if b[0][4] == 0 and b[0][0] == 1:
        combi17 = "Bertele-tele,"
    elif b[0][4] == 1 and b[0][0] == 0:
        combi17 = "Mudah terpengaruh pikiran orang lain,"
    else:
        combi17 = ""

    if b[0][4] == 1 and b[0][2] == 0:
        combi18 = "Cerdas,"
    else:
        combi18 = ""

    if b[0][4] == 0 and b[0][3] == 1:
        combi19 = "Menyukai hal-hal konvensional, Tradisional,"
    elif b[0][4] == 1 and b[0][3] == 0:
        combi19 = "Bertindak diluar kebiasaan,"
    else:
        combi19 = ""

    if b[0][4] == 0 and b[0][1] == 1:
        combi20 = "Tenang sekali."
    elif b[0][4] == 1 and b[0][1] == 0:
        combi20 = "Cemas berlebihan, Ingin menjadi pusat perhatian."
    else:
        combi20 = ""

    hasil = pd.DataFrame({'extraversion': h_ext, 'Neuroticism': h_neu,
                         'Agreeableness': h_agr, 'Conscientiousness': h_con, 'Openness': h_opn})
    st.table(hasil[:5])
    st.markdown('**Karakteristik Utama :**')
    st.write(c_ext, ', ', c_neu, ', ', c_agr, ', ', c_con, ', ', c_opn)
    st.markdown(
        '**Karakteristik Berdasarkan Gabungan antara dua kepribadian  :**')
    st.write(combi1, combi2, combi3, combi4, combi5, combi6, combi7, combi8, combi9, combi10,
             combi11, combi12, combi13, combi14, combi15, combi16, combi17, combi18, combi19, combi20)


def privasi():
    st.markdown(
        '**Privasi Pengguna :** _aplikasi ini tidak menyimpan, menyalin atau mengambil informasi apapun dari akun twitter yang diinputkan oleh pengguna, semua data pengguna otomatis akan hilang setelah pengguna meninggalkan aplikasi._')


def crawling(nama_pengguna):
    api_key = data_api.API_KEY
    api_key_secret = data_api.API_SECRET
    access_token = data_api.ACCESS_TOKEN
    access_token_secret = data_api.ACCESS_TOKEN_SECRET
    auth = tweepy.OAuthHandler(api_key, api_key_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)
    if nama_pengguna:
        limit = 100
        tweets = tweepy.Cursor(api.user_timeline, screen_name=nama_pengguna,
                               count=70, tweet_mode="extended").items(limit)
        columns = ['Tweet']
        hide_table_row_index = """
                <style>
                tbody th {display:none}
                .blank {display:none}
                </style>
                """
        st.markdown(hide_table_row_index, unsafe_allow_html=True)
        data = []
        for tweet in tweets:
            data.append(tweet.full_text)
            df = pd.DataFrame(data, columns=columns)
        data = str(data)
        # Case Folding
        lower_case = data.lower()
        # Tokenizing
        # Username removal
        lower_case = re.sub(r'@[^\s]+', '', lower_case)
        # Hastag Removal
        lower_case = re.sub(r'#([^\s]+)', '', lower_case)
        # URL removal
        lower_case = re.sub(r'https:[^\s]+', '', lower_case)
        # Symbol removal
        lower_case = lower_case.translate(
            str.maketrans("", "", string.punctuation))
        # ASCII chars
        lower_case = re.sub(r'[^\x00-\x7f]+', '', lower_case)
        # Double spasi
        lower_case = re.sub(r'\s+', ' ', lower_case)
        # number removal
        lower_case = re.sub(r"\d+", "", lower_case)
        # Token
        tokens = nltk.tokenize.word_tokenize(lower_case)
        freq_tokens = nltk.FreqDist(tokens)
        freq_tokens.plot(30, cumulative=False)
        grafik = plt.show()
        # Stemming
        token = str(tokens)
        factorySt = StemmerFactory()
        stemmer = factorySt.create_stemmer()
        hasil_stemming = stemmer.stem(token)
        # Stopword Removal
        factorySw = StopWordRemoverFactory()
        stopword = factorySw.create_stop_word_remover()
        hasil_stopword_removal = stopword.remove(hasil_stemming)
        # Slang-word converting
        slangwords = dict()
        with open('dataseet/slangword-id.txt') as wordfile:
            for word in wordfile:
                word = word.split('=')
                slangwords[word[0]] = word[1].replace('\n', '')
                wordsArray, fixed = hasil_stopword_removal.split(' '), []
            for word in wordsArray:
                if word in slangwords:
                    word = slangwords[word]
                fixed.append(word)
                hasil_slang_word = ' '.join(fixed)
            # Hapus character yang berulang
            pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
            hps_loop_char = ''
            for word in hasil_slang_word.split(' '):
                if word != '':
                    hps_loop_char += pattern.sub(r"\1", word) + ' '
            dat = []
            dat.append(hps_loop_char)
        st.subheader("User Lookup")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('**Profil Pengguna**')
            st.image(tweet.user.profile_image_url, width=70)
        with col2:
            st.write("Created :", tweet.user.created_at)
            st.write("Username : @", tweet.user.screen_name)
            st.markdown("**_Lima_ Tweet Terbaru :**")
        tabel_tweet = st.table(df[:5])
        with st.spinner('Wait for it...'):
            time.sleep(5)
            st.success('Done!')
        run_klasifikasi(dat)
        st.markdown(
            """<hr style="height:2px;border:none;color:#333;background-color:#333;margin-top:5px;" /> """, unsafe_allow_html=True)
    else:
        st.warning('Masukkan username terlebih dahulu...')
        st.markdown(
            """<hr style="height:2px;border:none;color:#333;background-color:#333;margin-top:5px;" /> """, unsafe_allow_html=True)
        privasi()


st.title("Sistem Klasifikasi Kepribadian Berdasarkan Postingan di Twitter")
st.sidebar.subheader("Masukkan username Twitter")
nama_pengguna = st.sidebar.text_input(
    "Username Twitter", placeholder="masukkan username", autocomplete=None, on_change=None)
st.sidebar.button("klasifikasi", key=None, help=None,
                  on_click=crawling(nama_pengguna), disabled=False)
st.sidebar.markdown(
    """<hr style="height:2px;border:none;color:#333;background-color:#333;margin-top:5px;" /> """, unsafe_allow_html=True)
