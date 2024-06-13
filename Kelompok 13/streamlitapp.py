import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import streamlit as st
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
from sklearn.metrics import mean_squared_error

# Set page config
st.set_page_config(page_title="Prediksi Konsumsi Listrik", initial_sidebar_state="expanded")


# Load data
data = pd.read_excel('Generated_Data_Listrik_Stasioner.xlsx', index_col="tanggal", parse_dates=True)

# Sidebar configuration
with st.sidebar:
    selected_page = option_menu("Menu", ["Prediksi Data", "Analisis Data"],
                                icons=['graph-up', 'bar-chart'],
                                menu_icon="menu-button-wide", default_index=0)

# Main content
link = "https://www.analyticsvidhya.com/blog/2020/10/how-to-create-an-arima-model-for-time-series-forecasting-in-python/"

if selected_page == "Prediksi Data":

    nama_kolom = st.sidebar.selectbox(
        "Pilih Nama Daerah",
        data.columns,
        help="Pilih daerah untuk melihat dan memprediksi konsumsi listrik."
    )

    forecast_steps = st.sidebar.slider(
        "Pilih Jumlah Hari untuk Prediksi",
        min_value=30,
        max_value=3650,
        value=180,
        step=10,
        help="Tentukan jumlah hari untuk memprediksi konsumsi listrik."
    )

    st.title("Prediksi Konsumsi Listrik untuk Daerah Bandung")
    st.markdown(f"""Aplikasi ini memprediksi konsumsi listrik di berbagai daerah menggunakan model ARIMA. Metode ARIMA (Autoregressive Integrated Moving Average) adalah 
    salah satu metode yang paling populer dan banyak digunakan dalam analisis deret waktu untuk keperluan peramalan (forecasting). 
    ARIMA merupakan model statistik yang menggabungkan tiga komponen utama: autoregressive (AR), differencing (I untuk Integrated), dan moving average (MA). 
    Untuk informasi lebih lanjut, kunjungi artikel ini: [How to create an ARIMA model]({link}). Kami menyediakan 5 data konsumsi listrik dari 5 daerah yang kami jadikan studi kasus. Daerah yang dimaksud adalah domisili Bandung, Jawa Barat,
    yaitu ada Ciwidey, Dago, Lembang, Banjaran dan Cibiru.
    Pilih daerah yang diinginkan, tentukan jumlah hari untuk prediksi, dan lihat hasilnya pada grafik interaktif.
    """)

    df = data[nama_kolom]

    # Menampilkan head dari dataset
    st.subheader("Dataset overview")
    st.write("""
    Dataset yang kami gunakan adalah dataset hasil generate program yang berisikan 3650 baris data dan terdapat 6 kolom. 
    Kolom "tanggal" berisikan data tanggal dan waktu, sedangkan kolom lainnya merupakan data konsumsi listrik Daerah dalam satuan (kWh).
    """)
    num_rows = st.slider(
        "Pilih Jumlah Baris untuk Ditampilkan",
        min_value=5,
        max_value=100,
        value=5,
        step=5
    )
    st.write(data.head(num_rows))

    df = data[nama_kolom]

    # Check for missing values
    missing_values = df.isnull().values.any()
    if missing_values:
        print("Terdapat missing values dalam DataFrame.")
    else:
        print("Tidak ada missing values dalam DataFrame.")

    # Plot data asli menggunakan Plotly
    st.subheader('Plot Data Konsumsi Listrik')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df, mode='lines+markers', name='Konsumsi Listrik', line=dict(color='royalblue'), marker=dict(size=5, color='royalblue')))
    st.write("Berikut merupakan Plot dari Data Historis yang merupakan Data Train/latih untuk melakukan prediksi. Data ini disimulasikan selama 10 tahun terakhir, tepatnya dari tanggal 1 Januari 2014 hingga 1 Januari 2024.")
    fig.update_layout(title=f'Data Konsumsi Listrik untuk daerah {nama_kolom}',
                      xaxis_title='Tanggal',
                      yaxis_title='Konsumsi Listrik (kWh)',
                      title_x=0.25,  # Center the title
                      template='plotly_white')

    # Menampilkan grafik interaktif
    st.plotly_chart(fig)

    # Function for ADF test
    def adf_test(data, column_name):
        result = adfuller(data)
        uji_statistik, p_value, _, _, nilai_kritis, _ = result
        print(f"Hasil uji ADF untuk kolom '{column_name}':")
        print(f"Nilai statistik uji: {uji_statistik}")
        print(f"Nilai p-value: {p_value}")
        print(f"Nilai Kritis:")
        for key, value in nilai_kritis.items():
            print(f"    {key}: {value}")
        if p_value < 0.05 and uji_statistik < nilai_kritis['5%']:
            print("Data cenderung stasioner")
        else:
            print("Data cenderung tidak stasioner. Data perlu diolah lebih lanjut.")

    # ADF test
    adf_test(df, nama_kolom)

    # Model parameters
    model_params = {
        "Ciwidey": (2, 0, 1),
        "Dago": (1, 0, 1),
        "Lembang": (2, 0, 2),
        "Banjaran": (1, 0, 1),
        "Cibiru": (4, 0, 4)
    }

    if nama_kolom not in model_params:
        st.write(f"Model untuk kolom '{nama_kolom}' tidak ditemukan.")
        st.stop()

    p, d, q = model_params[nama_kolom]

    # Apply ARIMA model
    model_arma = sm.tsa.ARIMA(df, order=(p, d, q))
    results = model_arma.fit()

    # Forecasting
    forecast = results.get_forecast(steps=forecast_steps)
    forecast_index = pd.date_range(start=df.index[-1], periods=forecast_steps + 1, inclusive='right')
    forecast_series = pd.Series(forecast.predicted_mean, index=forecast_index)

    # Simulasi data ARIMA
    np.random.seed(42)
    pred = results.get_prediction(start=len(df), end=len(df) + forecast_steps - 1, dynamic=True)
    predicted_values = pred.predicted_mean
    std_errors = pred.se_mean
    residuals = results.resid
    simulated_data = np.zeros(forecast_steps)

    for i in range(forecast_steps):
        simulated_data[i] = predicted_values[i] + np.random.normal(0, std_errors[i]) + residuals[-1]

    arma_sim = simulated_data.copy()

    # Interactive plot with plotly
    st.subheader('Plot hasil Prediksi Simulasi ARIMA')
    st.write(f'Data Prediksi Konsumsi Listrik dengan Simulasi ARIMA dengan Forecasting {forecast_steps} Hari')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df, mode='lines', name='Data Asli'))
    fig.add_trace(go.Scatter(x=forecast_series.index, y=arma_sim, mode='lines', name='Simulasi ARIMA (Prediksi)', line=dict(color='mediumseagreen')))
    fig.update_layout(xaxis_title='Tanggal', yaxis_title='Konsumsi Listrik (kWh)', hovermode='x unified')
    st.plotly_chart(fig)

    # Menampilkan rata-rata konsumsi listrik yang diprediksi
    rata_rata_prediksi = forecast_series.mean()

    # Menentukan tren prediksi dibandingkan dengan data historis
    trend = "meningkat" if rata_rata_prediksi > df.mean() else "menurun"

    # Kesimpulan hasil prediksi
    rata_rata_historis = df.mean()

    kesimpulan = f"""
    Rata-rata konsumsi listrik yang diprediksi adalah {rata_rata_prediksi:.2f} kWh (kilowatt-hours) dan
    Tren hasil prediksi cenderung {trend}.
    {"Prediksi menunjukkan bahwa konsumsi listrik akan meningkat dibandingkan dengan data historis." if trend == "meningkat" else "Prediksi menunjukkan bahwa konsumsi listrik akan menurun dibandingkan dengan data historis."}
    """

    st.markdown(kesimpulan)

     # Calculate and display RMSE as percentage
    rmse = np.sqrt(mean_squared_error(df[-forecast_steps:], forecast_series[:forecast_steps]))
    rata_rata_historis = df.mean()
    rmse_percentage = (rmse / rata_rata_historis) * 100
    st.subheader(f"Root Mean Square Error (RMSE) dari prediksi adalah {rmse:.2f} kWh ({rmse_percentage:.2f}%)")
    st.write(f"RMSE penting dalam evaluasi model prediksi, membantu dalam mengukur keakuratan dan mengidentifikasi model yang lebih baik. Menyajikan RMSE dalam bentuk persentase memberikan wawasan tambahan yang lebih mudah dimengerti tentang seberapa besar kesalahan prediksi relatif terhadap rata-rata konsumsi listrik.")
    

else:
    st.title("Analisis Data Konsumsi Listrik Daerah Bandung")

    # Heatmap for correlation
    st.subheader('Heatmap Korelasi Konsumsi Listrik')
    fig, ax = plt.subplots()
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    st.write(""" Secara keseluruhan, heatmap ini menunjukkan bahwa konsumsi listrik di lima daerah tersebut tidak memiliki korelasi yang signifikan. Hampir semua nilai korelasi berada sangat dekat dengan 0, menunjukkan bahwa perubahan konsumsi listrik di satu daerah tidak terkait erat dengan perubahan di daerah lain. Hal ini bisa mengindikasikan bahwa setiap daerah memiliki pola konsumsi listrik yang unik dan tidak dipengaruhi oleh daerah lain.""")

    # Data distribution
    st.subheader('Distribusi Konsumsi Listrik')

    nama_kolom = st.selectbox(
        "Pilih Nama Daerah untuk Analisis Distribusi",
        data.columns,
        help="Pilih daerah untuk melihat distribusi konsumsi listrik."
    )

    def plot_distribution(data, column_name):
        fig, ax = plt.subplots()
        sns.histplot(data[column_name], kde=True, ax=ax)
        ax.set_title(f'Distribusi Konsumsi Listrik di {column_name}')
        st.pyplot(fig)

        mean_val = data[column_name].mean()
        median_val = data[column_name].median()
        std_val = data[column_name].std()
    
        # Kesimpulan berdasarkan daerah
        if column_name == 'Ciwidey':
            kesimpulan = f"""
            **Kesimpulan untuk {column_name}:**
            - Rata-rata konsumsi listrik: {mean_val:.2f} kWh
            - Median konsumsi listrik: {median_val:.2f} kWh
            - Standar deviasi konsumsi listrik: {std_val:.2f} kWh
        
            Dari data tersebut, dapat disimpulkan bahwa konsumsi listrik di {column_name} memiliki rata-rata {mean_val:.2f} kWh dengan median {median_val:.2f} kWh.
            Hal ini menunjukkan bahwa distribusi data memiliki kecenderungan ke arah {'kanan (positif skewed)' if mean_val > median_val else 'kiri (negatif skewed)'}. Standar deviasi yang sebesar {std_val:.2f} kWh menunjukkan bahwa variasi konsumsi listrik di {column_name} {'tinggi' if std_val > mean_val else 'rendah'}.
            """
        elif column_name == 'Dago':
            kesimpulan = f"""
            **Kesimpulan untuk {column_name}:**
            - Rata-rata konsumsi listrik: {mean_val:.2f} kWh
            - Median konsumsi listrik: {median_val:.2f} kWh
            - Standar deviasi konsumsi listrik: {std_val:.2f} kWh
        
            Distribusi konsumsi listrik di daerah Dago menunjukkan pola distribusi normal dengan puncak konsumsi berada di sekitar nilai 300. Sebagian besar konsumsi listrik di Dago terkonsentrasi dalam rentang yang sempit, mencerminkan penggunaan listrik yang seragam. Distribusi ini menunjukkan skewness yang sangat rendah, mendekati simetris."""
        elif column_name == 'Lembang':
            kesimpulan = f"""
            **Kesimpulan untuk {column_name}:**
            - Rata-rata konsumsi listrik: {mean_val:.2f} kWh
            - Median konsumsi listrik: {median_val:.2f} kWh
            - Standar deviasi konsumsi listrik: {std_val:.2f} kWh
        
            Distribusi konsumsi listrik di daerah Lembang menunjukkan pola distribusi normal dengan puncak konsumsi berada di sekitar nilai 350. Sebagian besar konsumsi listrik di Lembang terkonsentrasi dalam rentang yang relatif sempit, mencerminkan pola penggunaan listrik yang seragam di daerah tersebut. Distribusi ini menunjukkan skewness (kemencengan) yang sangat rendah, mendekati simetris.
            """
        elif column_name == 'Banjaran':
            kesimpulan = f"""
            **Kesimpulan untuk {column_name}:**
            - Rata-rata konsumsi listrik: {mean_val:.2f} kWh
            - Median konsumsi listrik: {median_val:.2f} kWh
            - Standar deviasi konsumsi listrik: {std_val:.2f} kWh
        
            Distribusi konsumsi listrik di daerah Banjaran menunjukkan pola distribusi normal dengan puncak konsumsi berada di sekitar nilai 400. Sebagian besar konsumsi listrik di Banjaran terkonsentrasi dalam rentang yang sempit, mencerminkan penggunaan listrik yang seragam. Distribusi ini menunjukkan skewness yang sangat rendah, mendekati simetris.
            """
        elif column_name == 'Cibiru':
            kesimpulan = f"""
            **Kesimpulan untuk {column_name}:**
            - Rata-rata konsumsi listrik: {mean_val:.2f} kWh
            - Median konsumsi listrik: {median_val:.2f} kWh
            - Standar deviasi konsumsi listrik: {std_val:.2f} kWh
        
            Distribusi konsumsi listrik di daerah Cibiru menunjukkan pola yang mendekati distribusi normal dengan nilai konsumsi listrik yang paling umum berada di sekitar 450. Skewness dari distribusi ini relatif rendah, menunjukkan bahwa data terdistribusi secara simetris di sekitar rata-rata. Ini menunjukkan bahwa tidak ada penyimpangan besar dalam konsumsi listrik di daerah tersebut, dengan sebagian besar rumah tangga atau unit cenderung mengonsumsi listrik dalam kisaran yang sama.
            """
        else:
            kesimpulan = f"""
            **Kesimpulan untuk {column_name}:**
            - Rata-rata konsumsi listrik: {mean_val:.2f} kWh
            - Median konsumsi listrik: {median_val:.2f} kWh
            - Standar deviasi konsumsi listrik: {std_val:.2f} kWh
            """
    
        st.markdown(kesimpulan)

    plot_distribution(data, nama_kolom)

    # Comparison between regions
    st.subheader('Perbandingan Konsumsi Listrik Antar Daerah')
    selected_columns = st.multiselect('Pilih Daerah untuk Dibandingkan', data.columns.tolist(), default=data.columns.tolist())
    fig, ax = plt.subplots()
    for col in selected_columns:
        ax.plot(data.index, data[col], label=col)
    ax.legend()
    st.pyplot(fig)
