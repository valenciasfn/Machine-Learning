# **KLASIFIKASI KUALITAS BIJI KEDELAI MENGGUNAKAN VGG-16 DAN RESNET-50**

## 📖 **OVERVIEW PROJECT**
Tujuan dari proyek ini adalah untuk mengembangkan sistem yang dapat mempermudah dan mempercepat identifikasi jenis kualitas benih kedelai untuk meningkatkan efisiensi dan kualitas pemrosesan benih kedelai secara keseluruhan. Klasifikasi secara manual sering kali menghadapi tantangan dalam menyortir benih kedelai secara akurat dan cepat, terutama di berbagai kategori seperti kedelai utuh, berbintik, belum matang, rusak, dan rusak kulit. Tantangan-tantangan ini dapat mengurangi efisiensi dan kualitas yang tidak konsisten jika dilakukan secara manual. Untuk mengatasi hal ini, teknologi pemrosesan gambar dan pembelajaran mendalam memberikan solusi dengan memungkinkan klasifikasi dan penyortiran biji kedelai secara otomatis. Teknologi ini tidak hanya meningkatkan efisiensi pemrosesan, tetapi juga memastikan standar kualitas produk dan mendukung skalabilitas dalam praktik pertanian.

## 📂 **DATASET**
Dalam projek ini, data yang digunakan dari platform Mendeley Data [Soybean Seeds Dataset](https://data.mendeley.com/datasets/v6vzvfszj6/6). Dataset ini terdiri dari lima jenis biji kedelai, yaitu utuh, berbintik, belum matang, pecah, dan rusak kulitnya, dengan masing-masing kategori memiliki lebih dari 1000 citra biji kedelai. Total keseluruhan dataset pada penelitian ini berjumlah 5513 citra. Semua citra berukuran sama, yakni 277 x 277 pixel.
![Citra Soybean Seeds](UAP/assets/Citra%20Soybean%20Seeds.png)

## ⚙ **PREPROCESSING**
Tahap preprocessing dilakukan untuk memperbaiki kualitas data serta mencegah hasil yang kurang optimal untuk meningkatkan kualitas dataset agar sesuai dengan model yang akan digunakan.
Berikut tahapan preprocessing yang dilakukan:
1. **Reshape**: dengan mengubah dimensi gambar menjadi ukuran yang diharapkan oleh model (224 x 224 piksel).
2. **Rescale**: dengan mengubah rentang nilai piksel dari 0 hingga 255 menjadi nilai 0 hingga 1.
3. Splitting Data
   - **Train**: 80% data digunakan untuk proses pelatihan model.
   - **Validation**: 10% data digunakan untuk memantau performa model selama pelatihan berlangsung.
   - **Test**: 10% digunakan untuk menguji kinerja model setelah proses pelatihan selesai.

## 🧠 **MODELS**
### **1. VGG-16**


## 📊 **EVALUASI MODEL**
## 🌐 **DEPLOYMENT WEB**
## **AUTHOR**
[Valencia Sefiana Putri]






