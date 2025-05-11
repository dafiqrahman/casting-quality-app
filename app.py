import streamlit as st
import tensorflow as tf
import torch
import numpy as np
from PIL import Image
import cv2
import keras
print(tf.__version__)
# Load CNN Model (Klasifikasi)


# @st.cache_resource
# def load_cnn_model():
#     return tf.keras.models.load_model("cnn_residuals (2).h5")

# # Load YOLOv5 Model (Deteksi)


# @st.cache_resource
# def load_yolo_model():
#     return torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)


cnn_model = tf.keras.models.load_model("cnn_residuals (2).h5")
yolo_model = torch.hub.load(
    'ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

# Create tabs for navigation
tab1, tab2 = st.tabs(
    ["Beranda", "Klasifikasi dan deteksi Cacat Logam"])

with tab1:
    # add the title centered "Judul Aplikasi"
    st.markdown(
        "<h1 style='text-align: center;'>Aplikasi Inspeksi Cacat produk Pengecoran Logam</h1>",
        unsafe_allow_html=True)
    # h2
    st.markdown(
        "<h2 style='text-align: center;'>Menggunakan Metode CNN dan Yolo Bi-FPN</h2>",
        unsafe_allow_html=True)
    # Create 3 columns with the middle column wider
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:  # This will center the content
        st.image("undip_logo.png", width=350)  # Tambahkan logo UNDIP

    st.markdown(
        "<h3 style='text-align: center;'>Oleh : Budi Setyawan</h3>",
        unsafe_allow_html=True)
    st.markdown(
        "<h3 style='text-align: center;'>Magister Sistem Informasi - Universitas Diponegoro</h3>",
        unsafe_allow_html=True)

with tab2:
    st.write("## Klasifikasi dan Deteksi Cacat Logam")
    st.write("Gunakan menu ini untuk melakukan klasifikasi guna mengetahui apakah terdapat kecacatan pada logam.")

    uploaded_file = st.file_uploader(
        "Input Gambar", type=["jpg", "png", "jpeg"],
        label_visibility="visible", key="klasifikasi_uploader")

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        col_preview, col_result = st.columns([1, 1])
        predicted_class = None

        with col_preview:
            st.write("### Preview Gambar")
            st.image(image, use_container_width=True)
            # Tombol prediksi di bawah gambar
            if st.button("Tombol Prediksi", key="klasifikasi_button"):
                # Proses prediksi
                img = tf.keras.utils.img_to_array(image)
                img = tf.image.resize(img, (224, 224))
                img_array = tf.expand_dims(img, axis=0)
                prediction = cnn_model.predict(img_array)
                class_names = ["Cacat", "Tidak Cacat"]
                predicted_class = class_names[int(np.round(prediction[0][0]))]
                confidence = (1 - prediction[0][0]) * 100

                # Menampilkan hasil prediksi di sisi kanan
                with col_result:
                    st.write("### Hasil Prediksi:")
                    st.write(f"**Status:** {predicted_class}")
                    col_cacat, col_tidak = st.columns(2)
                    with col_cacat:
                        st.metric("Persentase Logam Cacat",
                                  f"{confidence:.2f}%")
                    with col_tidak:
                        st.metric("Persentase Logam Tidak Cacat",
                                  f"{100-confidence:.2f}%")

                # Jika logam cacat, lanjut ke deteksi
        if predicted_class == "Cacat":
            st.write("---")
            st.write("## Deteksi Cacat Logam")

            col_deteksi1, col_deteksi2 = st.columns(2)
            with col_deteksi1:
                st.write("### Preview Gambar")
                st.image(image, use_container_width=True,
                         caption="Gambar Asli")

            with col_deteksi2:
                st.write("### Gambar Hasil Deteksi")
                image_cv = cv2.cvtColor(
                    np.array(image), cv2.COLOR_RGB2BGR)
                results = yolo_model(image_cv, size=320)
                st.image(results.render()[0],
                         caption="Deteksi Bounding Box",
                         use_container_width=True)
        else:
            st.success("Logam tidak mengalami cacat.")


# with tab3:
#     st.write("## Deteksi Cacat Logam dengan YOLOv5")
#     st.write(
#         "Gunakan menu ini untuk mendeteksi bagian logam yang mengalami kecacatan.")

#     uploaded_file = st.file_uploader(
#         "Unggah Gambar", type=["jpg", "png", "jpeg"],
#         key="deteksi_uploader")

#     if uploaded_file is not None:
#         image = Image.open(uploaded_file)
#         col_yolo, col_yolo2 = st.columns(2)
#         with col_yolo:
#             st.write("### Gambar Asli")
#             st.image(image, caption="Gambar Asli",
#                      use_container_width=True)
#             if st.button("Deteksi Cacat", key="deteksi_button"):
#                 with col_yolo2:

#                     image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

#                     # Perform YOLOv5 inference
#                     results = yolo_model(image_cv, size=320)

#                     st.write("### Gambar Hasil Deteksi")
#                     st.image(results.render()[0],
#                              caption="Hasil Deteksi dengan Bounding Box",
#                              use_container_width=True)
