import numpy as np
import tensorflow as tf
from pathlib import Path
import streamlit as st

# Judul aplikasi
st.title("Klasifikasi Kualitas Biji Kedelai")

# Deskripsi aplikasi
st.markdown(
    """
    <div style="text-align: justify; font-size: 18px; line-height: 1.8; color: rgb(0, 0, 0); padding: 20px; background: rgb(245, 245, 245); border: 4px solid rgb(66, 25, 5);
    border-radius: 12px; box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);">
        Aplikasi ini dirancang untuk mengklasifikasikan kualitas biji kedelai ke dalam 5 kategori, antara lain:
        <b style="color: rgb(66, 25, 5);">Broken soybeans</b>, <b style="color: rgb(66, 25, 5);">Immature soybeans</b>, 
        <b style="color: rgb(66, 25, 5);">Intact soybeans</b>, <b style="color: rgb(66, 25, 5);">Skin-damaged soybeans</b>, 
        dan <b style="color: rgb(66, 25, 5);">Spotted soybeans</b>. Dengan model AI, aplikasi ini membantu mengoptimalkan proses penyortiran kedelai secara otomatis.
    </div>
    """,
    unsafe_allow_html=True,
)

# Tambahkan CSS
st.markdown(
    """
    <style>
        body {
            background-color: #000000; /* Hitam sebagai latar belakang utama */
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: #FFFFFF; /* Warna putih untuk teks */
        }
        .stMarkdown b {
            font-weight: bold;
            color: rgb(66, 25, 5); /* Sentuhan cokelat untuk teks tebal */
        }
        .stTitle {
            text-align: center;
            color: #FFFFFF; /* Warna putih untuk judul */
            font-size: 42px;
            font-weight: 700;
            margin-bottom: 40px;
            text-shadow: 3px 3px 8px rgba(0, 0, 0, 0.3);
        }
        .stButton button {
            background-color: rgb(66, 25, 5); /* Warna cokelat untuk tombol */
            color: white;
            border-radius: 8px;
            padding: 12px 24px;
            font-size: 18px;
            border: none;
            cursor: pointer;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            transition: background-color 0.3s ease, transform 0.3s ease;
        }
        .stButton button:hover {
            background-color:rgb(255, 255, 255); /* Warna cokelat lebih terang saat hover */
            transform: translateY(-2px);
        }
        .stButton button:active {
            background-color: rgb(66, 25, 5) /* Warna cokelat lebih gelap saat aktif */
            transform: translateY(2px);
        }
        .stFileUploader {
            background: rgb(240, 240, 240); /* Warna abu terang untuk background uploader */
            padding: 15px;
            border-radius: 10px;
            border: 4px solid rgb(66, 25, 5); /* Garis cokelat untuk uploader */
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
        }
        .stFileUploader label {
            color: rgb(66, 25, 5);
            font-size: 12px; 
            font-weight: bold;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Fungsi untuk memuat dan memproses gambar
def predict(uploaded_image, model_path):
    class_names = [
        "Broken soybeans",
        "Immature soybeans",
        "Intact soybeans",
        "Skin-damaged soybeans",
        "Spotted soybeans",
    ]

    # Muat dan preprocess citra
    img = tf.keras.utils.load_img(uploaded_image, target_size=(224, 224))
    img = tf.keras.utils.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Muat model
    model = tf.keras.models.load_model(model_path)

    # Prediksi
    output = model.predict(img)
    score = tf.nn.softmax(output[0])
    return class_names[np.argmax(score)], 100 * np.max(score)

# Pilihan model untuk prediksi
model_option = st.selectbox(
    "Pilih model untuk prediksi:", 
    ["VGG-16", "ResNet50"]
)

# Tetapkan path model sesuai dengan pilihan
model_paths = {
    "VGG-16": Path(__file__).parent / "model\VGG-16_model.h5",
    "ResNet50": Path(__file__).parent / "model\ResNet50_model.h5"
}
model_path = model_paths[model_option]

# Komponen file uploader untuk beberapa file
uploads = st.file_uploader(
    "Unggah gambar kedelai untuk klasifikasi",
    type=["png", "jpg"],
    accept_multiple_files=True,
)

# Tombol prediksi
if st.button("Prediksi"):
    if uploads:
        st.subheader("Hasil Prediksi:")

        for upload in uploads:
            st.image(upload, caption=f"Citra diunggah: {upload.name}", use_column_width=True)

            with st.spinner(f"Memproses {upload.name}..."):
                try:
                    label, confidence = predict(upload, model_path)
                    st.write(f"**Hasil Klasifikasi:** {label}")
                    st.write(f"**Confidence:** {confidence:.2f}%")
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat memproses {upload.name}: {e}")
    else:
        st.error("Silakan unggah setidaknya satu gambar untuk diprediksi!")
