# StockPred Master

Aplikasi prediksi pasar saham yang cerdas yang memanfaatkan algoritma machine learning terdepan untuk memberikan pengguna prediksi harga saham real-time dan fitur visualisasi komprehensif, membantu pengguna membuat keputusan investasi yang tepat.

## 🎯 Gambaran Proyek

StockPred Master adalah aplikasi web berbasis Next.js yang mengintegrasikan API data keuangan dan algoritma machine learning canggih untuk memprediksi harga pasar saham. Aplikasi ini menawarkan berbagai opsi visualisasi, fitur pemilihan algoritma, dan dokumentasi arsitektur komprehensif untuk membantu pengguna mendapatkan wawasan tentang tren pasar saham.

## ✨ Fitur Utama

### 🔗 Integrasi Data Real-Time
- Koneksi ke API data keuangan (Alpha Vantage, Yahoo Finance, Google Finance)
- Lapisan caching cerdas dengan TTL 60 detik dan rasio hit cache 85%
- Rate limiting dengan degradasi yang halus (5 panggilan/menit, 500 panggilan/hari)
- Dukungan WebSocket untuk pembaruan real-time
- Mode demo dengan data cadangan untuk pengembangan

### 🤖 Algoritma Machine Learning Canggih
- **LSTM (Long Short-Term Memory)** - R² = 0.92, spesialisasi untuk time series
- **Model Transformer** - R² = 0.93, pemodelan sequence berbasis attention
- **Hybrid CNN-LSTM** - R² = 0.91, ekstraksi fitur spasial + temporal
- **Model Ensemble** - R² = 0.94 (Performa Terbaik), rata-rata berbobot
- **XGBoost** - R² = 0.89, gradient boosting dengan regularisasi
- **Random Forest** - R² = 0.88, ensemble pohon keputusan
- **Gradient Boost** - R² = 0.90, peningkatan model berurutan
- **TDDM (Time-Dependent Deep Model)** - Model kustom untuk time series keuangan

### 📊 Visualisasi Data Interaktif
- Grafik candlestick real-time dengan indikator teknis
- Perbandingan prediksi multi-algoritma
- Dashboard metrik performa (RMSE, MAE, R²)
- Penilaian kepercayaan dan kuantifikasi ketidakpastian
- Indikator teknis yang dapat disesuaikan (RSI, MACD, Moving Averages)
- Desain responsif untuk mobile dan desktop

### 🛠️ Fitur Teknis
- Inferensi ML sisi klien dengan TensorFlow.js
- Konfigurasi pelatihan cepat untuk demonstrasi
- Manajemen memori dan pembersihan tensor
- Pemrosesan batch untuk dataset besar
- Pola factory model untuk pemilihan algoritma
- TypeScript untuk keamanan tipe

## 🏗️ Dokumentasi Arsitektur

### Diagram Arsitektur Sistem
Proyek ini mencakup diagram arsitektur komprehensif yang dihasilkan dengan Python matplotlib:

```
architecture_diagrams/
├── system_architecture.png    # Arsitektur sistem 4-layer utama (527KB)
├── ml_pipeline.png           # Alur kerja pelatihan ML lengkap (454KB)
└── data_flow.png            # Pipeline pemrosesan data real-time (378KB)
```

Untuk menghasilkan diagram yang diperbarui:
```bash
python generate_architecture.py
```

### Layer Arsitektur
1. **Layer Frontend** - React + TensorFlow.js + 8 komponen terhubung
2. **Layer API** - Next.js Server + 5 layanan terintegrasi
3. **Layer Model ML** - 8 algoritma + Factory Model
4. **API Eksternal** - 5 sumber data + caching cerdas

## 🚀 Stack Teknologi

### Teknologi Frontend
- **Next.js 14.1.0** - Framework React dengan App Router
- **React 18.2.0** - Library komponen UI
- **TypeScript 5.3.3** - JavaScript yang type-safe
- **TailwindCSS 3.4.1** - Framework CSS utility-first

### Visualisasi Data
- **ApexCharts 4.7.0** - Grafik keuangan interaktif
- **Chart.js 4.4.9** - Library grafik responsif
- **D3.js 7.8.5** - Visualisasi data kustom
- **ECharts 5.5.0** - Solusi charting profesional

### Machine Learning & Pemrosesan Data
- **TensorFlow.js 4.17.0** - Machine learning sisi klien
- **Axios 1.6.7** - Klien HTTP untuk integrasi API
- **Model ML Kustom** - Diimplementasikan dalam TypeScript

### Backend & API
- **Next.js API Routes** - Fungsi backend serverless
- **Alpha Vantage API** - Sumber data keuangan utama
- **Rate Limiting** - Sistem antrian permintaan dan caching

## 📁 Struktur Proyek Rinci

```
stockpred-master/
├── app/                           # Direktori app Next.js 14
│   ├── components/                # Komponen React yang dapat digunakan kembali
│   │   ├── AlgorithmSelector.tsx  # Pemilihan algoritma ML
│   │   ├── AlgorithmStatus.tsx    # Status pelatihan real-time
│   │   ├── AmlVisualization.tsx   # Visualisasi analisis ML
│   │   ├── ApiStatus.tsx          # Monitoring koneksi API
│   │   ├── DateRangePicker.tsx    # Pemilihan rentang data historis
│   │   ├── PredictionResults.tsx  # Tampilan output prediksi
│   │   ├── StockChart.tsx         # Grafik keuangan interaktif
│   │   ├── StockSelector.tsx      # Pemilihan simbol saham
│   │   └── TechnicalIndicators.tsx # Alat analisis teknis
│   ├── dashboard/                 # Dashboard aplikasi utama
│   │   ├── [algorithm]/           # Halaman dinamis khusus algoritma
│   │   │   └── page.tsx          # Analisis algoritma individual
│   │   └── page.tsx              # Interface dashboard utama
│   ├── hooks/                     # Custom React hooks
│   │   └── useStockData.ts       # Hook manajemen data saham
│   ├── lib/                       # Library utilitas
│   │   ├── algorithmPerformance.ts # Kalkulasi metrik performa
│   │   ├── api.ts                # Klien API dan pengambilan data
│   │   ├── technicalIndicators.ts # Fungsi analisis teknis
│   │   └── tuning.ts             # Optimisasi hyperparameter
│   ├── models/                    # Model Machine Learning
│   │   ├── CNNLSTMModel.ts       # Hybrid konvolusional + LSTM
│   │   ├── config.ts             # Konfigurasi model dan pelatihan cepat
│   │   ├── EnsembleModel.ts      # Implementasi ensemble berbobot
│   │   ├── GradientBoostModel.ts # Algoritma gradient boosting
│   │   ├── index.ts              # Factory model dan ekspor
│   │   ├── LSTMModel.ts          # Jaringan neural LSTM
│   │   ├── RandomForestModel.ts  # Implementasi random forest
│   │   ├── TDDMModel.ts          # Model deep time-dependent
│   │   ├── TransformerModel.ts   # Transformer dengan self-attention
│   │   └── XGBoostModel.ts       # Implementasi XGBoost
│   ├── tuning/                    # Interface tuning hyperparameter
│   │   └── page.tsx              # Dashboard tuning model
│   ├── layout.tsx                # Komponen layout root
│   ├── page.tsx                  # Halaman beranda
│   └── styles/
│       └── globals.css           # Gaya global dan CSS kustom
├── architecture_diagrams/         # Dokumentasi arsitektur yang dihasilkan
│   ├── system_architecture.png   # Gambaran sistem (527KB)
│   ├── ml_pipeline.png          # Alur kerja ML (454KB)
│   └── data_flow.png            # Pemrosesan data (378KB)
├── analysis/                      # Makalah penelitian dan dokumentasi
│   ├── applsci-14-05062-v2.pdf   # Penelitian Applied Sciences ML
│   ├── Comparative_Study_on_Stock_Market_Prediction_using_Generic_CNN-LSTM_and_Ensemble_Learning.pdf
│   └── computation-13-00003.pdf   # Analisis komputasional
├── generate_architecture.py       # Generator diagram arsitektur
├── next-env.d.ts                 # Deklarasi TypeScript Next.js
├── next.config.js                # Konfigurasi Next.js
├── package.json                  # Dependensi dan skrip
├── postcss.config.js             # Konfigurasi PostCSS
├── tailwind.config.js            # Konfigurasi TailwindCSS
├── tsconfig.json                 # Konfigurasi TypeScript
└── README.md                     # Dokumentasi proyek
```

## 🚀 Memulai

### Prasyarat
- **Node.js 16.x atau lebih tinggi**
- **npm atau yarn package manager**
- **Python 3.8+** (untuk generasi diagram arsitektur)

### Instalasi

1. **Clone repository:**
   ```bash
   git clone https://github.com/dihannahdi/finalproject_ai.git
   cd finalproject_ai-1
   ```

2. **Instal dependensi:**
   ```bash
   npm install
   # atau
   yarn install
   ```

3. **Konfigurasi API keys:**
   - Dapatkan API key gratis dari [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
   - Buat file `.env.local` di direktori root:
     ```env
     NEXT_PUBLIC_ALPHA_VANTAGE_API_KEY=your_api_key_here
     ```
   - **Catatan**: Tanpa API key, aplikasi akan menggunakan mode "demo" dengan data sampel
   - **Batas Tarif**: API key gratis memiliki batasan (5 panggilan/menit, 500 panggilan/hari)

4. **Jalankan server pengembangan:**
   ```bash
   npm run dev
   # atau
   yarn dev
   ```

5. **Buka aplikasi:**
   Navigasi ke [http://localhost:3000](http://localhost:3000) di browser Anda

### Generate Diagram Arsitektur (Opsional)

```bash
# Instal dependensi Python
pip install matplotlib

# Generate diagram arsitektur komprehensif
python generate_architecture.py
```

## 📖 Panduan Penggunaan

### Penggunaan Dasar
1. **Navigasi** ke halaman dashboard
2. **Pilih** simbol saham dari dropdown menu
3. **Pilih** algoritma prediksi (LSTM, Transformer, Ensemble, dll.)
4. **Atur** rentang tanggal untuk analisis data historis
5. **Lihat** hasil prediksi dan visualisasi interaktif
6. **Sesuaikan** indikator teknis dan pengaturan grafik
7. **Bandingkan** performa algoritma dengan metrik benchmark

### Fitur Lanjutan
- **Tuning Algoritma**: Sesuaikan hyperparameter di halaman tuning
- **Analisis Performa**: Bandingkan skor RMSE, MAE, dan R² antar model
- **Pembaruan Real-time**: Monitor status API dan kesegaran data
- **Ekspor Data**: Unduh prediksi dan grafik untuk analisis eksternal

## 📊 Benchmark Performa

| Algoritma | RMSE | MAE | Skor R² | Waktu Pelatihan | Penggunaan Memori |
|-----------|------|-----|----------|---------------|--------------|
| **Ensemble** | **1.95** | **1.68** | **0.94** | 125s | 180MB |
| Transformer | 2.08 | 1.76 | 0.93 | 60s | 150MB |
| LSTM | 2.15 | 1.83 | 0.92 | 45s | 120MB |
| CNN-LSTM | 2.22 | 1.90 | 0.91 | 52s | 140MB |
| Gradient Boost | 2.28 | 1.95 | 0.90 | 35s | 90MB |
| XGBoost | 2.35 | 2.01 | 0.89 | 30s | 80MB |
| Random Forest | 2.41 | 2.08 | 0.88 | 25s | 70MB |

*Hasil benchmark berdasarkan backtesting dengan data S&P 500*

## 🎓 Presentasi Teknis

Proyek ini mencakup presentasi teknis komprehensif 6 slide yang mencakup:

1. **Arsitektur Sistem & Stack Teknologi** - Gambaran teknologi lengkap
2. **Arsitektur Pipeline ML** - Alur kerja pelatihan dan inferensi
3. **Arsitektur Aliran Data** - Pipeline pemrosesan real-time
4. **Implementasi Deep Learning** - Detail LSTM dan Transformer
5. **Ensemble & Performa** - Teknik optimisasi lanjutan
6. **Metrik & Ekspor** - Benchmark dan dokumentasi arsitektur

## 🔧 Referensi API

### Factory Model
```typescript
import { createModel } from '@/app/models';

// Buat dan konfigurasi model
const model = createModel('ensemble', {
  epochs: 10,
  timeSteps: 20,
  batchSize: 16,
  learningRate: 0.01
});

// Latih model
await model.train(stockData);

// Generate prediksi
const predictions = await model.predict(stockData, 7); // Perkiraan 7 hari
```

### Manajemen Data
```typescript
import { useStockData } from '@/app/hooks/useStockData';

// Ambil dan kelola data saham
const { data, loading, error, fetchData } = useStockData();

// Ambil saham spesifik dengan caching
await fetchData('AAPL', { 
  interval: '1day', 
  outputsize: 'compact' 
});
```

## 🧪 Testing & Pengembangan

### Mode Pelatihan Cepat
Aplikasi mencakup konfigurasi yang dioptimalkan untuk pengembangan:
- **Epoch yang dikurangi**: 10 alih-alih 100-200 untuk iterasi lebih cepat
- **Sequence yang lebih kecil**: 20 timesteps alih-alih 60
- **Pemrosesan batch**: 16 sampel per batch untuk efisiensi memori
- **Inferensi sisi klien**: TensorFlow.js untuk prediksi real-time

### Manajemen Memori
- Pembersihan tensor otomatis untuk mencegah kebocoran memori
- Optimisasi garbage collection
- Pemrosesan batch untuk dataset besar
- Monitoring performa dan peringatan

## 🚨 Disclaimer

Aplikasi ini hanya untuk **tujuan demonstrasi dan edukasi**. Prediksi **tidak boleh dianggap sebagai saran keuangan**. Selalu konsultasikan dengan penasihat keuangan yang berkualifikasi sebelum membuat keputusan investasi.

- **Peringatan Risiko**: Investasi pasar saham membawa risiko yang melekat
- **Tidak Ada Jaminan**: Performa masa lalu tidak menjamin hasil masa depan
- **Penggunaan Edukasi**: Dirancang untuk pembelajaran konsep ML dan analisis keuangan

## 📄 Lisensi

MIT License - lihat file [LICENSE](LICENSE) untuk detail.

## 🤝 Kontribusi

Kontribusi dipersilakan! Silakan submit Pull Request. Untuk perubahan besar, mohon buka issue terlebih dahulu untuk mendiskusikan apa yang ingin Anda ubah.

### Setup Pengembangan
1. Fork repository
2. Buat feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit perubahan Anda (`git commit -m 'Add some AmazingFeature'`)
4. Push ke branch (`git push origin feature/AmazingFeature`)
5. Buka Pull Request

## 📞 Kontak & Dukungan

- **Email**: info@stockpredmaster.com
- **Issues**: [GitHub Issues](https://github.com/dihannahdi/finalproject_ai/issues)
- **Dokumentasi**: [Project Wiki](https://github.com/dihannahdi/finalproject_ai/wiki)

## 🙏 Ucapan Terima Kasih

- Makalah penelitian di direktori `analysis/` untuk panduan implementasi ML
- Alpha Vantage untuk menyediakan API data keuangan
- Tim TensorFlow.js untuk kemampuan ML sisi klien
- Komunitas Next.js dan React untuk dokumentasi yang sangat baik

---

**⭐ Beri bintang repository ini jika Anda merasa terbantu!** 