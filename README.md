# Proyek Klasifikasi Kualitas Anggur Merah

## 1. Domain Proyek
Industri anggur global merupakan sektor yang sangat kompetitif, di mana kualitas produk menjadi faktor utama dalam menentukan kepuasan konsumen dan keberhasilan bisnis. Kualitas anggur yang tinggi tidak hanya memperkuat citra merek, tetapi juga berkontribusi langsung terhadap nilai jual produk di pasar. Oleh karena itu, evaluasi kualitas anggur secara akurat dan konsisten merupakan aspek krusial dalam proses produksi.

Namun, penilaian kualitas anggur secara tradisional umumnya masih mengandalkan metode sensorik oleh panel ahli. Meskipun pendekatan ini dianggap akurat, terdapat sejumlah kendala yang melekat pada proses tersebut. Evaluasi sensorik bersifat subjektif, memerlukan waktu yang tidak sedikit, dan menuntut biaya tinggi. Selain itu, penerapannya dalam skala besar menjadi tantangan tersendiri karena keterbatasan sumber daya manusia dan potensi ketidakkonsistenan antar evaluator. Akibatnya, fluktuasi kualitas yang tidak terdeteksi dapat terjadi dan berdampak negatif terhadap reputasi serta pendapatan produsen.

Permasalahan ini menunjukkan adanya kebutuhan mendesak akan metode penilaian kualitas anggur yang lebih objektif, efisien, dan dapat diandalkan. Salah satu pendekatan yang menjanjikan adalah pemanfaatan teknologi machine learning untuk mengembangkan sistem prediksi berbasis data.

Dengan memanfaatkan data fisikokimia yang terukur seperti tingkat keasaman, kadar alkohol, dan nilai pH, machine learning dapat digunakan untuk membangun model klasifikasi yang mampu mengidentifikasi kualitas anggur secara otomatis. Model ini diharapkan dapat membantu produsen dalam:
* Mengoptimalkan proses produksi, melalui identifikasi faktor-faktor kunci yang memengaruhi kualitas.
* Menjaga konsistensi produk, dengan memastikan standar kualitas sebelum produk dipasarkan.
* Mendukung pengambilan keputusan strategis, seperti penentuan harga, seleksi bahan baku, atau modifikasi proses fermentasi.
Dengan pendekatan ini, produsen anggur tidak lagi sepenuhnya bergantung pada evaluasi manual yang mahal dan bervariasi, melainkan dapat memanfaatkan sistem prediksi cepat dan akurat berbasis data.

**Referensi:**

* Aich, S., Al-Absi, A. A., Hui, K. L., Lee, J. T., & Sain, M. (2018, February). A classification approach with different feature sets to predict the quality of different types of wine using machine learning techniques. In 2018 20th International Conference on Advanced Communication Technology (ICACT) (pp. 1–2). Chuncheon, Korea (South). https://doi.org/10.23919/ICACT.2018.8323673
* Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Modelling wine preferences by data mining from physicochemical properties. Decision Support Systems, 47(4), 547–553. https://doi.org/10.1016/j.dss.2009.05.016

## 2. Business Understanding

### 2.1. Pernyataan Masalah (Problem Statement)

1. Bagaimana cara secara akurat memprediksi kualitas anggur merah (dikategorikan sebagai Baik atau Buruk) berdasarkan atribut fisikokimia yang terukur?
2. Fitur fisikokimia mana saja yang paling berpengaruh terhadap kualitas anggur merah?
3. Sejauh mana performa model klasifikasi machine learning dalam memberikan prediksi yang konsisten dan dapat diandalkan untuk mendukung proses produksi?
### 2.2. Tujuan (Goals)

1. Mengembangkan model klasifikasi machine learning yang mampu memprediksi kualitas anggur merah berdasarkan data fisikokimia.
2. Mengidentifikasi dan menganalisis fitur-fitur fisikokimia yang paling signifikan dalam menentukan kualitas anggur.
3. Mencapai tingkat akurasi prediksi di atas 75% pada data pengujian untuk memastikan keandalan model dalam praktik nyata.


### 2.3. Pernyataan Solusi (Solution Statement)

Untuk mencapai tujuan tersebut, kami mengusulkan solusi sebagai berikut:

1.  **Penggunaan Multiple Algoritma:** Akan digunakan dua atau lebih algoritma klasifikasi (`Logistic Regression`, `Random Forest Classifier`, dan `XGBoost Classifier`) untuk membandingkan performa dan memilih model terbaik. Untuk menjawab masalah tentang fitur paling berpengaruh, akan dilakukan analisis feature importance pada model berbasis pohon (seperti Random Forest dan XGBoost) untuk mengidentifikasi atribut fisikokimia yang paling signifikan dalam menentukan kualitas anggur.
3.  **Hyperparameter Tuning:** Untuk model berkinerja tinggi seperti `Random Forest` dan `XGBoost`, akan dilakukan *hyperparameter tuning* menggunakan `GridSearchCV` untuk mengoptimalkan performa mereka melampaui *baseline*.
4.  **Metrik Evaluasi Akurasi:** Akurasi (`accuracy_score`) akan menjadi metrik utama untuk mengukur kinerja model. Selain itu, `Confusion Matrix` dan `Classification Report` akan digunakan untuk analisis yang lebih mendalam mengenai *precision*, *recall*, dan *F1-score* untuk setiap kelas.

## 3. Data Understanding

Dataset yang digunakan adalah "Wine Quality Red" dari UCI Machine Learning Repository. Dataset ini berisi data kimia dan fisik anggur merah Portugis yang dikaitkan dengan evaluasi sensorik.

* **Jumlah Data:** 1599 entri (baris)
* **Jumlah Kolom/Fitur:** 12 kolom
* **Kondisi Data:** Data bersih, tidak ada nilai yang hilang (non-null count = 1599 untuk semua kolom). Tipe data sebagian besar float64, kecuali kolom target asli `quality` yang bertipe int64.

**Tautan Sumber Data:** [https://archive.ics.uci.edu/dataset/186/wine+quality]

**Variabel/Fitur pada Data:**

| Fitur                 | Deskripsi                                                                                                          |
| :-------------------- | :----------------------------------------------------------------------------------------------------------------- |
| `fixed acidity`       | Sebagian besar asam non-volatil yang terlibat dalam kualitas anggur.                                             |
| `volatile acidity`    | Jumlah asam asetat dalam anggur, yang pada tingkat tinggi dapat menyebabkan rasa cuka yang tidak menyenangkan. |
| `citric acid`         | Dapat menambah "kesegaran" dan rasa pada anggur.                                                                  |
| `residual sugar`      | Jumlah gula yang tersisa setelah fermentasi berhenti. Manisnya anggur.                                             |
| `chlorides`           | Jumlah garam dalam anggur.                                                                                         |
| `free sulfur dioxide` | Bagian SO2 bebas yang mencegah pertumbuhan mikroba dan oksidasi anggur.                                         |
| `total sulfur dioxide`| Jumlah SO2 bebas dan terikat.                                                                                     |
| `density`             | Kepadatan anggur.                                                                                                  |
| `pH`                  | Tingkat keasaman atau kebasaan anggur.                                                                             |
| `sulphates`           | Menambah tingkat sulfur dioksida, yang berfungsi sebagai antimikroba dan antioksidan.                         |
| `alcohol`             | Persentase kadar alkohol dalam anggur.                                                                             |
| `quality`             | Skor kualitas anggur (berdasarkan data sensorik, skala 0-10). Ini adalah target asli.                              |

**Eksplorasi Data Awal:**
| Langkah                    | Deskripsi                                                                     |
| -------------------------- | ----------------------------------------------------------------------------- |
| Pemeriksaan missing values | Tidak ditemukan nilai kosong pada dataset                                     |
| Pemeriksaan duplikasi      | Terdapat **240** baris duplikat – telah dihapus dengan `df.drop_duplicates()` |
| Pemeriksaan outlier        | Menggunakan metode **IQR** ditemukan outlier di sebagian besar kolom          |
| Penanganan outlier         | Dilakukan **capping (winsorizing)** untuk mengurangi dampak nilai ekstrem     |

## 4. Data Preparation

Pada tahap ini, data disiapkan agar sesuai untuk pemodelan *machine learning*.

1.  **Mendefinisikan Masalah Klasifikasi Biner:**
    * Kolom `quality` (skala 3-8) diubah menjadi masalah klasifikasi biner.
    * Anggur dengan `quality` < 6 dikategorikan sebagai **'Buruk' (0)**.
    * Anggur dengan `quality` >= 6 dikategorikan sebagai **'Baik' (1)**.
    * Kolom baru `quality_label` dibuat untuk ini.

2.  **Pemisahan Fitur (X) dan Target (y):**
    * Fitur (X) mencakup semua kolom karakteristik fisikokimia kecuali `quality` asli dan `quality_label`.
    * Target (y) adalah kolom `quality_label`.

3.  **Pemisahan Data Training dan Testing:**
    * Data dibagi menjadi 80% data pelatihan dan 20% data pengujian (`test_size=0.20`).
    * `random_state=42` digunakan untuk reproduktibilitas.
    * `stratify=y` digunakan untuk memastikan distribusi kelas target yang seimbang antara set pelatihan dan pengujian.

4.  **Feature Scaling (StandardScaler):**
    * Fitur diskalakan menggunakan `StandardScaler` sehingga memiliki rata-rata 0 dan standar deviasi 1. Ini penting untuk algoritma yang sensitif terhadap skala fitur (seperti Logistic Regression) dan juga bermanfaat untuk model berbasis pohon.

## 5. Modeling

Pada tahap ini, tiga model klasifikasi diimplementasikan, dilatih, dan di-*tune* untuk memprediksi kualitas anggur.

1.  **Fungsi Pelatihan dan Evaluasi Model:**
    Dibuat sebuah fungsi (`train_and_evaluate_model`) untuk melatih dan mengevaluasi model secara konsisten, mencetak akurasi, *confusion matrix*, dan *classification report*. Fungsi ini juga memvisualisasikan *confusion matrix*.

2.  **Logistic Regression:**
    Digunakan sebagai model *baseline*. Model ini dilatih pada data yang sudah diskalakan.

3.  **Random Forest Classifier (Tuned):**
    Model Random Forest di-*tune* menggunakan `GridSearchCV` dengan *hyperparameter* seperti `n_estimators`, `max_features`, `max_depth`, `min_samples_split`, dan `min_samples_leaf` untuk menemukan kombinasi optimal yang meningkatkan akurasi.

4.  **XGBoost Classifier (Tuned):**
    Model XGBoost, dikenal dengan performa tingginya, juga di-*tune* menggunakan `GridSearchCV` dengan *hyperparameter* seperti `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, dan `gamma`.

## 6. Evaluation

### 6.1. Ringkasan Performa Model

Setelah semua model dilatih dan di-*tune*, performa akurasi mereka dirangkum dan dibandingkan:

| Model                 | Akurasi |
| :-------------------- | :------ |
| XGBoost (Tuned)       | 0.834375|
| Random Forest (Tuned) | 0.809375|
| Logistic Regression   | 0.740625|


### 6.2. Visualisasi Perbandingan Akurasi Model

Visualisasi ini secara jelas menunjukkan model mana yang memiliki akurasi tertinggi:

 ![image alt](https://github.com/Nabilafairuzr/dicoding-submission-klasifikasi-buang-anggur-/blob/d58243ed640d1226c13c6699c08910e61ec69bd3/percentage.jpeg?raw=true)


### 6.3. Analisis Model Terbaik: XGBoost (Tuned)

Berdasarkan perbandingan akurasi, **XGBoost (Tuned)** adalah model terbaik yang dipilih untuk menyelesaikan permasalahan ini, karena memberikan akurasi tertinggi pada data pengujian.

#### 6.3.1. Analisis Confusion Matrix dan Classification Report

Detail performa XGBoost (Tuned) dapat dilihat dari Confusion Matrix dan Classification Report yang dihasilkan saat evaluasi model. Metrik `Accuracy` mengukur proporsi prediksi yang benar dari total prediksi.

* **Confusion Matrix:** Menunjukkan jumlah True Positives (TP), True Negatives (TN), False Positives (FP), dan False Negatives (FN). Ini penting untuk memahami jenis kesalahan yang dibuat model.
* **Classification Report:** Menyajikan metrik `Precision` (proporsi positif yang benar), `Recall` (proporsi positif yang teridentifikasi dengan benar), dan `F1-Score` (rata-rata harmonis precision dan recall) untuk setiap kelas. Metrik ini sangat relevan untuk dataset dengan kelas yang mungkin tidak seimbang.

#### 6.3.2. Analisis Feature Importance

Untuk memahami fitur mana yang paling berpengaruh dalam prediksi kualitas anggur oleh model XGBoost, kita dapat melihat `feature importances`:
![image alt](https://github.com/Nabilafairuzr/dicoding-submission-klasifikasi-buang-anggur-/blob/main/image%20feature.jpeg?raw=true)
Dari analisis ini, kita dapat melihat fitur-fitur yang paling berkontribusi terhadap model, memberikan wawasan berharga tentang karakteristik kimiawi anggur yang paling dominan dalam menentukan kualitasnya.

---
