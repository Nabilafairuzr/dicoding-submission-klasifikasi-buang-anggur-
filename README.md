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

## 5. Model Deployment

Pada tahap ini, tiga model klasifikasi diimplementasikan, dilatih, dan dioptimalkan untuk memprediksi kualitas anggur. Pemilihan model didasarkan pada kemampuannya dalam menangani masalah klasifikasi biner dan mengidentifikasi pola yang tersembunyi dalam data.

Untuk menjaga konsistensi dalam proses pelatihan dan evaluasi, digunakan sebuah fungsi kustom bernama `train_and_evaluate_model`. Fungsi ini menerima input berupa model dan data pelatihan/pengujian, kemudian:
- Melatih model dengan data pelatihan
- Melakukan prediksi pada data pengujian
- Mencetak metrik evaluasi seperti `accuracy_score`, `confusion_matrix`, dan `classification_report`
- Menampilkan visualisasi confusion matrix untuk mempermudah analisis performa model

### 5.1 Model 1: Logistic Regression

**1. Pembahasan Cara Kerja**  
Logistic Regression merupakan algoritma klasifikasi linier yang digunakan untuk memodelkan probabilitas suatu kelas dalam masalah klasifikasi biner. Algoritma ini menggunakan fungsi sigmoid untuk mengkonversi kombinasi linier dari fitur input menjadi probabilitas bernilai antara 0 dan 1. Jika nilai probabilitas melebihi ambang batas tertentu (misalnya 0.5), maka data diklasifikasikan ke kelas positif, sebaliknya ke kelas negatif. Model ini bekerja dengan mencari hubungan linier antara fitur dan log-odds dari target.

**2. Pembahasan Parameter**  
Model ini dilatih menggunakan pustaka Scikit-learn dengan parameter sebagai berikut:
- `random_state=42`: Digunakan untuk memastikan hasil yang konsisten saat pelatihan diulang.
- `solver='liblinear'`: Digunakan karena efisien untuk dataset kecil dan mendukung regularisasi L1 maupun L2.

Parameter lainnya dibiarkan sebagai default karena telah sesuai dengan karakteristik data.

**3. Kelebihan/Kekurangan**  
- Kelebihan: Mudah diinterpretasikan, efisien dan cepat dilatih, serta menghasilkan probabilitas yang dapat digunakan untuk analisis lebih lanjut.  
- Kekurangan: Asumsi hubungan linier dapat membatasi performa ketika data memiliki pola non-linier yang kompleks. Rentan terhadap outlier dan memerlukan penskalaan fitur.


### 5.2 Model 2: Random Forest Classifier (Tuned)

**1. Pembahasan Cara Kerja**  
Random Forest adalah algoritma ensambel berbasis pohon keputusan yang terdiri dari banyak pohon yang dilatih secara independen. Setiap pohon dilatih menggunakan teknik bootstrapping (pengambilan sampel acak dengan penggantian) dan subset acak dari fitur. Prediksi akhir diambil berdasarkan voting mayoritas dari seluruh pohon. Pendekatan ini mengurangi risiko overfitting yang umum terjadi pada pohon keputusan tunggal dan meningkatkan generalisasi model.

**2. Pembahasan Parameter**  
Model ini dioptimalkan menggunakan `GridSearchCV` untuk mencari kombinasi parameter terbaik. Parameter grid yang digunakan:
- `n_estimators`: [100, 200, 300] – Jumlah pohon dalam hutan
- `max_features`: ['sqrt', 'log2'] – Jumlah fitur yang dipertimbangkan dalam pemisahan node
- `max_depth`: [10, 20, None] – Kedalaman maksimum pohon
- `min_samples_split`: [2, 5] – Jumlah minimum sampel untuk membagi node internal
- `min_samples_leaf`: [1, 2] – Jumlah minimum sampel pada setiap daun pohon


**3. Kelebihan/Kekurangan**  
- Kelebihan: Sangat efektif dalam mengurangi overfitting, mampu menangani banyak fitur (termasuk fitur non-linier dan interaksi), dan dapat memberikan estimasi feature importance.<br>
- Kekurangan: Kurang dapat diinterpretasikan dibandingkan pohon keputusan tunggal karena kompleksitas banyak pohon. Proses pelatihan bisa lebih lambat jika jumlah pohon sangat banyak.

### **Model 3: XGBoost Classifier (Tuned)**
**1. Pembahasan Cara Kerja**  
XGBoost (eXtreme Gradient Boosting) adalah algoritma boosting yang membangun pohon keputusan secara bertahap untuk memperbaiki kesalahan model sebelumnya. Setiap pohon baru difokuskan untuk meminimalkan sisa kesalahan (residual) dari gabungan pohon sebelumnya. XGBoost mendukung regularisasi L1 dan L2, sehingga mampu mengendalikan kompleksitas model dan mencegah overfitting. Selain itu, XGBoost sangat efisien secara komputasi karena mendukung eksekusi paralel dan memiliki optimasi bawaan untuk kecepatan tinggi serta kemampuan menangani missing value secara internal.

**2. Pembahasan Parameter**  
Model ini dioptimalkan menggunakan `GridSearchCV` untuk mencari kombinasi hyperparameter terbaik. Parameter grid yang digunakan:
- `n_estimators`: [100, 200, 300]
- `learning_rate`: [0.01, 0.05, 0.1]
- `max_depth`: [3, 5, 7]
- `subsample`: [0.7, 0.8, 0.9]
- `colsample_bytree`: [0.7, 0.8, 0.9]
- `gamma`: [0, 0.1, 0.2]

Tuning dilakukan menggunakan `GridSearchCV` dengan `cv=5` (5-fold cross-validation) dan `scoring='accuracy'`. Parameter terbaik yang diperoleh (contoh hasil tuning):

```python
{'colsample_bytree': 0.8, 'gamma': 0, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 200, 'subsample': 0.8}
```

**3. Kelebihan/Kekurangan**
- Kelebihan: Performa sangat tinggi dan sering menjadi pemenang di berbagai kompetisi machine learning. Efisien dalam komputasi dan mampu menangani missing values secara internal. Memiliki fitur regularisasi yang kuat untuk mencegah overfitting.
- Kekurangan: Bisa kompleks untuk di-tune karena banyaknya hyperparameter. Model kurang mudah diinterpretasikan dibandingkan model linier atau pohon tunggal.

## 6. Evaluation

### 6.1. Metrik Evaluasi yang Digunakan

Untuk mengevaluasi performa model, digunakan beberapa metrik yang relevan dengan kasus klasifikasi biner, yaitu:

- **Accuracy**: Proporsi prediksi yang benar dibandingkan total prediksi.
$$\frac{TP + TN}{TP + TN + FP + FN}$$
- **Precision**: Proporsi prediksi positif yang benar-benar positif.
$$\text{Precision} = \frac{TP}{TP + FP}$$
- **Recall**: Proporsi data positif yang berhasil terdeteksi.
$$\text{Recall} = \frac{TP}{TP + FN}$$
- **F1-Score**: Rata-rata harmonis dari precision dan recall, berguna saat data tidak seimbang.
$$F1 Score = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
- **Confusion Matrix**: Untuk melihat distribusi prediksi benar dan salah secara lebih rinci.
$$
\begin{bmatrix}
TN & FP \\
FN & TP \\
\end{bmatrix}
$$

Pemilihan metrik ini penting karena **tujuan bisnis** dalam konteks ini adalah meminimalkan kesalahan dalam mengklasifikasikan kualitas anggur, terutama dalam menghindari anggur berkualitas rendah diklasifikasikan sebagai berkualitas tinggi, yang dapat berdampak buruk pada kepuasan konsumen dan citra merek.



### 6.2. Ringkasan Performa Model

Setelah semua model dilatih dan dituning, berikut hasil evaluasi akurasi pada data pengujian:

| Model                 | Akurasi  |
| :-------------------- | :------- |
| XGBoost (Tuned)       | 0.783088 |
| Random Forest (Tuned) | 0.772059 |
| Logistic Regression   | 0.735294 |

XGBoost menunjukkan performa terbaik dari ketiganya.



### 6.3. Visualisasi Perbandingan Akurasi Model

Visualisasi berikut memberikan gambaran jelas perbandingan akurasi antar model:

![Perbandingan Akurasi Model](https://github.com/Nabilafairuzr/dicoding-submission-klasifikasi-buang-anggur-/blob/d58243ed640d1226c13c6699c08910e61ec69bd3/percentage.jpeg?raw=true)



### 6.4. Analisis Model Terbaik: XGBoost (Tuned)

#### 6.4.1. Confusion Matrix dan Classification Report

Hasil evaluasi model XGBoost menunjukkan:

- Akurasi tinggi dibandingkan model lain
- Precision dan recall seimbang, yang berarti model tidak bias terhadap salah satu kelas
- Confusion Matrix menunjukkan distribusi kesalahan model secara detail

Metrik ini memperkuat keandalan XGBoost dalam memprediksi kualitas anggur secara akurat dan konsisten.

#### 6.4.2. Feature Importance

Untuk memahami kontribusi tiap fitur dalam prediksi, ditampilkan visualisasi feature importance dari model XGBoost:

![Feature Importance](https://github.com/Nabilafairuzr/dicoding-submission-klasifikasi-buang-anggur-/blob/main/image%20feature.jpeg?raw=true)

Hasil ini memberikan **wawasan bisnis penting** mengenai fitur kimiawi mana yang paling menentukan kualitas anggur. Informasi ini dapat dimanfaatkan oleh produsen untuk **mengontrol proses produksi dan meningkatkan kualitas produk**.



### 6.5. Keterkaitan dengan Business Understanding

Model yang dibangun telah menjawab **seluruh problem statement** dan **goals** yang ditetapkan, yaitu:

- Menghasilkan sistem klasifikasi otomatis yang dapat membedakan kualitas anggur secara efisien
- Memberikan dasar pengambilan keputusan berbasis data terkait proses produksi anggur
- Membantu perusahaan mengurangi risiko distribusi produk berkualitas rendah ke konsumen

Dengan model terbaik (XGBoost), prediksi kualitas dapat dilakukan secara **lebih akurat dan cepat**, sehingga memberikan **dampak langsung terhadap kualitas layanan dan kepuasan pelanggan**.
