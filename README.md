# Aplikasi Deteksi dan Klasifikasi Cacat Logam

Aplikasi ini menggunakan model **CNN** untuk klasifikasi dan **YOLOv5** untuk deteksi cacat logam. Dibangun menggunakan **Streamlit**, aplikasi ini memungkinkan pengguna untuk mengunggah gambar dan mendapatkan prediksi apakah logam mengalami kecacatan atau tidak.

## 🚀 Instalasi Lokal

### 1. Clone Repository
```sh
git clone https://github.com/username/repository-name.git
cd repository-name
```

### 2. Buat Virtual Environment (Opsional)
Disarankan untuk menjalankan aplikasi di dalam virtual environment:
```sh
python -m venv venv
source venv/bin/activate  # Untuk macOS/Linux
venv\Scripts\activate  # Untuk Windows
```

### 3. Install Dependencies
```sh
pip install -r requirements.txt
```

### 4. Download Model Files
Pastikan Anda telah memiliki model berikut:
- **model.h5** (CNN untuk klasifikasi)
- **best.pt** (YOLOv5 untuk deteksi)

Tempatkan file model di direktori utama proyek.

### 5. Jalankan Aplikasi
```sh
streamlit run app.py
```
Aplikasi akan berjalan di browser pada `http://localhost:8501`.

## 📌 Fitur Aplikasi
1. **Beranda**: Informasi tentang aplikasi dan pembuatnya.
2. **Klasifikasi Cacat Logam**: Menggunakan model CNN untuk menentukan apakah logam memiliki cacat atau tidak.
3. **Deteksi Cacat Logam**: Menggunakan YOLOv5 untuk mendeteksi bagian cacat pada gambar logam.

## 📜 Lisensi
Aplikasi ini dibuat untuk tujuan penelitian dan edukasi.

---

Silakan hubungi jika ada pertanyaan atau kontribusi! 😊

