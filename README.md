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

Pada tahap ini, tiga model klasifikasi diimplementasikan, dilatih, dan di-tune untuk memprediksi kualitas anggur. Setiap model dipilih berdasarkan karakteristiknya dalam menangani masalah klasifikasi biner dan kemampuannya untuk mengidentifikasi pola dalam data.

5.1.  **Fungsi Pelatihan dan Evaluasi Model:**
    Untuk memastikan konsistensi dalam pelatihan dan evaluasi model, sebuah fungsi kustom bernama train_and_evaluate_model dibuat. Fungsi ini menerima model dan data pelatihan/pengujian, kemudian melatih model, melakukan prediksi, dan mencetak metrik evaluasi utama (accuracy_score, confusion_matrix, classification_report). Fungsi ini juga mampu memvisualisasikan confusion matrix untuk analisis yang lebih intuitif.

5.2.  **Pelatihan**
**Model 1: Logistic Regression**
a. Pembahasan Cara Kerja: Logistic Regression adalah algoritma klasifikasi linier yang digunakan untuk memprediksi probabilitas suatu kelas, biasanya untuk masalah klasifikasi biner. Meskipun namanya mengandung "Regression," ia bekerja dengan memetakan input ke probabilitas antara 0 dan 1 melalui fungsi sigmoid. Jika probabilitas yang dihasilkan melewati ambang batas tertentu (misalnya, 0.5), maka data tersebut diklasifikasikan ke satu kelas, dan sebaliknya. Ini mencoba menemukan hubungan linier terbaik antara fitur input dan log-odds dari kelas target.
b. Pembahasan Parameter: Model ini dilatih dengan parameter default dari Scikit-learn, dengan beberapa penyesuaian spesifik:
* random_state=42: Parameter ini digunakan untuk memastikan reproduktivitas hasil model, sehingga setiap kali kode dijalankan dengan random_state yang sama, hasil inisialisasi internal model juga akan sama.

* solver='liblinear': Ini adalah algoritma optimasi yang digunakan untuk menemukan koefisien model. liblinear dipilih karena efisien untuk dataset kecil dan cocok untuk model yang mendukung regularisasi L1 dan L2.

c. Kelebihan/Kekurangan:
* Kelebihan: Mudah diinterpretasikan karena hubungannya linier, sangat efisien dan cepat dalam pelatihan, serta memberikan probabilitas keluaran yang berguna.
* Kekurangan: Asumsi hubungan linier antara fitur dan log-odds target dapat membatasi performa pada data yang memiliki hubungan non-linier kompleks. Rentan terhadap outlier dan memerlukan penskalaan fitur.

**Model 2: Random Forest Classifier (Tuned)**
a. Pembahasan Cara Kerja: Random Forest adalah metode ansambel berbasis pohon keputusan yang bekerja dengan membangun banyak pohon keputusan secara independen. Setiap pohon dilatih pada subset data yang di-bootstrapping (pengambilan sampel acak dengan penggantian) dan subset fitur yang dipilih secara acak. Untuk klasifikasi, prediksi akhir ditentukan oleh voting mayoritas dari prediksi masing-masing pohon. Mekanisme ini secara efektif mengurangi masalah overfitting yang sering terjadi pada satu pohon keputusan tunggal dan meningkatkan kemampuan generalisasi model.

b. Pembahasan Parameter (Hasil Tuning): Model Random Forest di-tune menggunakan GridSearchCV untuk menemukan kombinasi hyperparameter terbaik yang meningkatkan akurasi. Parameter grid yang digunakan untuk pencarian adalah:
* n_estimators: [100, 200, 300] (Jumlah pohon dalam hutan)
max_features: ['sqrt', 'log2'] (Jumlah fitur yang dipertimbangkan untuk setiap pemisahan)
* max_depth: [10, 20, None] (Kedalaman maksimum pohon, None berarti simpul diperluas sampai semua daun murni atau sampai semua daun berisi kurang dari min_samples_split sampel)
* min_samples_split: [2, 5] (Jumlah minimum sampel yang dibutuhkan untuk membagi node internal)
* min_samples_leaf: [1, 2] (Jumlah minimum sampel yang dibutuhkan di setiap node daun)
* GridSearchCV dengan cv=5 (5-fold cross-validation) dan scoring='accuracy' digunakan. Model terbaik yang ditemukan dari tuning akan memiliki hyperparameter optimal yang digunakan untuk evaluasi akhir.
* Parameter Terbaik yang Ditemukan (Contoh, sesuaikan dengan output Anda):
{'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}

c. Kelebihan/Kekurangan:
Kelebihan: Sangat efektif dalam mengurangi overfitting, mampu menangani banyak fitur (termasuk fitur non-linier dan interaksi), dan dapat memberikan estimasi feature importance.
Kekurangan: Kurang dapat diinterpretasikan dibandingkan pohon keputusan tunggal karena kompleksitas banyak pohon. Proses pelatihan bisa lebih lambat jika jumlah pohon sangat banyak.

**Model 3: XGBoost Classifier (Tuned)**
a. Pembahasan Cara Kerja: XGBoost (eXtreme Gradient Boosting) adalah implementasi yang sangat efisien dan populer dari algoritma Gradient Boosting. Ia membangun model secara sekuensial, di mana setiap pohon baru (disebut weak learner) berusaha mengoreksi kesalahan yang dibuat oleh gabungan pohon-pohon sebelumnya. XGBoost menambahkan regularisasi (L1 dan L2) untuk mencegah overfitting dan mendukung paralelisasi, membuatnya sangat cepat dan kuat. Ini sering menjadi pilihan utama karena performa tinggi dan kemampuannya menangani berbagai jenis data.
b. Pembahasan Parameter (Hasil Tuning): Model XGBoost di-tune menggunakan GridSearchCV untuk mengoptimalkan performanya. Parameter grid yang digunakan untuk pencarian adalah:
* n_estimators: [100, 200, 300] (Jumlah pohon)
* learning_rate: [0.01, 0.05, 0.1] (Ukuran langkah pembelajaran)
* max_depth: [3, 5, 7] (Kedalaman maksimum pohon)
* subsample: [0.7, 0.8, 0.9] (Rasio sampel data untuk setiap pohon)
* colsample_bytree: [0.7, 0.8, 0.9] (Rasio kolom/fitur untuk setiap pohon)
* gamma: [0, 0.1, 0.2] (Reduksi loss minimum yang diperlukan untuk melakukan pemisahan simpul)
* GridSearchCV dengan cv=5 (5-fold cross-validation) dan scoring='accuracy' digunakan. Model terbaik yang ditemukan dari tuning akan memiliki hyperparameter optimal yang digunakan untuk evaluasi akhir.
* Parameter Terbaik yang Ditemukan (Contoh, sesuaikan dengan output Anda):
{'colsample_bytree': 0.8, 'gamma': 0, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 200, 'subsample': 0.8}

c. Kelebihan/Kekurangan:
Kelebihan: Performa sangat tinggi dan sering menjadi pemenang di berbagai kompetisi machine learning. Efisien dalam komputasi dan mampu menangani missing values secara internal. Memiliki fitur regularisasi yang kuat untuk mencegah overfitting.
Kekurangan: Bisa kompleks untuk di-tune karena banyaknya hyperparameter. Model kurang mudah diinterpretasikan dibandingkan model linier atau pohon tunggal.
## 6. Evaluation

### 6.1. Ringkasan Performa Model

Setelah semua model dilatih dan di-*tune*, performa akurasi mereka dirangkum dan dibandingkan:

| Model                 | Akurasi |
| :-------------------- | :------ |
| XGBoost (Tuned)       | 0.783088|
| Random Forest (Tuned) | 0.772059|
| Logistic Regression   | 0.735294|


### 6.2. Visualisasi Perbandingan Akurasi Model

Visualisasi ini secara jelas menunjukkan model mana yang memiliki akurasi tertinggi:

 ![image alt](https://github.com/Nabilafairuzr/dicoding-submission-klasifikasi-buang-anggur-/blob/main/image%20(1).png?raw=true)


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
