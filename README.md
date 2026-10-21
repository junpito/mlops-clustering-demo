# Demo MLOps: user baru memicu clustering

Demo mandiri hasil adaptasi proyek BMLP clustering dan contoh Workflow-CI/SMSML.
Satu baris dataset mewakili **satu AccountID**, diperlakukan sebagai user demo.
GitHub Actions menjalankan eksperimen KMeans saat snapshot user berubah; MLflow
mencatat eksperimen dan artefak. Tidak ada deployment production otomatis.

## Jalankan lokal

Gunakan Python 3.12. Dari folder ini:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python train.py --output results/baseline-demo
python view_runs.py
```

Di Linux/macOS: gunakan `python3.12 -m venv .venv` dan `source .venv/bin/activate`.
MLflow UI ada di http://127.0.0.1:5000, experiment **user-clustering-demo**.
Tutup UI dengan Ctrl+C saat selesai.

`train.py` menolak output directory yang sudah ada agar hasil lama tidak
tercampur. Untuk percobaan berikutnya pilih nama output baru. Semua run tetap
tercatat di `mlruns/`.

## Demo pertambahan user

1. Mulai dari `data/users.csv`: **296 user**.
2. Simulasikan batch baru:

   ```powershell
   python append_users.py
   ```

   Menambahkan **99 user**, menjadi **395**. Pengulangan perintah menambahkan 0
   user. ID lama dengan fitur berbeda ditolak: demo ini khusus penambahan user.

3. Secara lokal:

   ```powershell
   python train.py --output results/after-new-users
   ```

4. Di repository GitHub, commit/push perubahan **data/users.csv** ke `main`:

   ```bash
   git add data/users.csv
   git commit -m "demo: add 99 users"
   git push origin main
   ```

   Workflow berjalan otomatis. Jika melalui pull request, PR menjalankan tes;
   training berjalan setelah perubahan digabungkan ke `main`.

**Menambah baris lokal saja tidak memicu GitHub Actions.** Pemicu demo adalah
push perubahan snapshot. Database/event ETL bukan bagian dari contoh ini.
Workflow juga dapat dipicu manual melalui **Actions → User data - clustering
experiments → Run workflow**.

## Memasang di GitHub

Salin **isi folder demo ini ke root repository tersendiri**, termasuk
`.github/workflows/clustering.yml` dan `.gitignore`; jangan masukkan `.venv`,
`mlruns`, `results`, atau raw transaksi. Workflow pada subfolder `outputs/`
repository Manirent tidak akan ditemukan oleh GitHub Actions.

Gunakan branch `main`. Tidak diperlukan secrets, Docker Hub, DagsHub, atau
tracking server. Saat menyalin, pertahankan file atribusi. Workflow belum
dianggap teruji di GitHub sampai benar-benar dijalankan pada repository tujuan.

## Apa yang dilatih dan dibandingkan?

- Lima fitur perilaku numerik: jumlah transaksi, rata-rata nilai transaksi,
  durasi transaksi, percobaan login, dan saldo akun.
- `StandardScaler + KMeans` disimpan bersama dalam satu pipeline.
- Baseline tetap: k=3, seed=42, dilatih ulang secara deterministik pada 296 user
  awal. Ini baseline demo yang dibekukan, bukan model production/champion.
- Kandidat: k=2,3,4 dengan seed 42 dan 43, dilatih pada snapshot user saat ini.
- Pemilihan kandidat memakai **49 user validasi**; **50 user test** hanya untuk
  pemeriksaan akhir kandidat terpilih.
- Silhouette semua model dihitung pada ruang fitur yang sama, yaitu scaler
  yang di-fit pada baseline. Ini mencegah perubahan scaler membuat skor
  perbandingan memakai definisi jarak berbeda.
- Stabilitas: adjusted Rand index (ARI) antar-seed pada user validasi.
- PCA hanya untuk gambar 2D, bukan preprocessing model.

Gate teknis demo, ditetapkan sebelum melihat hasil:

- test silhouette >= 0.05;
- penurunan silhouette dari baseline <= 0.02;
- semua cluster muncul pada test, masing-masing >= 5%;
- stabilitas seed pada validasi (ARI) >= 0.60;
- model tersimpan harus dapat dimuat ulang dengan prediksi identik.

Gagal gate: status `REJECTED`, exit code 2, workflow merah; artefak dan alasan
tetap diunggah. Lolos: `READY_FOR_INTEGRATION`. Kedua hasil **tidak otomatis
mengganti model production**. Threshold ini contoh mekanisme, bukan standar
bisnis. Cluster ID tidak stabil lintas retraining; jangan menganggap cluster 0
pada dua model sebagai segmen bisnis yang sama.

Set test yang dipakai berulang adalah benchmark demo, bukan bukti generalisasi
untuk semua data masa depan. Evaluasi bisnis memerlukan pemeriksaan profil
cluster; riwayat transaksi yang dipakai di sini bukan simulasi waktu produksi.

## Hasil yang dilihat saat presentasi

1. Jumlah user berubah 296 → 395 pada commit dataset.
2. GitHub Actions memulai workflow, menjalankan tes lalu eksperimen.
3. Summary menampilkan kandidat terpilih, perbandingan baseline, dan keputusan gate.
4. Unduh artifact `clustering-<run-id>-<attempt>` (retensi 14 hari).
5. Ekstrak **seluruh archive** ke folder baru, install requirements, lalu:

   ```powershell
   python view_runs.py
   ```

   Script memperbaiki URI file-store dari runner Linux ke lokasi hasil ekstraksi
   sebelum membuka UI. Ini hanya untuk archive tracking lokal demo, bukan remote
   tracking server.

6. Buka parent run `users-296` atau `users-395`; expand child runs. Parameter,
   metrik, model setiap kandidat, dan `handoff/` tersedia di UI.

Setiap artifact GitHub berisi snapshot tracking run tersebut. Untuk membandingkan
dua ukuran dataset dalam satu UI, jalankan dua demo lokal pada folder `mlruns`
yang sama; artifact antar-job GitHub tidak otomatis digabungkan.

## Isi paket serah terima

`results/<nama-run>/`:

- `model/`: pipeline sklearn, signature, input example, dan dependensi.
- `report.json`: status gate, kandidat, hash dataset, run ID, dan versi kode.
- `summary.md`: ringkasan pemeriksaan.
- `cluster_profiles.csv`: statistik per cluster pada skala asli.
- `assignments.csv`: user dan cluster untuk snapshot training.
- `clusters.png`: visualisasi PCA.
- `input_output_contract.json`: kontrak integrasi.

Contoh assignment tanpa retraining:

```powershell
python predict.py --model results/baseline-demo/model --data data/incoming_users.csv --output results/new-user-assignments.csv
```

Gunakan artefak tepercaya dari demo ini. User ID hanya untuk menghubungkan hasil;
tidak menjadi fitur model. Input harus memiliki semua kolom kontrak, numerik
finite/nonnegatif, dan `transaction_count` bilangan bulat positif.

## Data dan asal proyek

Lihat [ATTRIBUTION.md](ATTRIBUTION.md) dan `data/manifest.json`.
Snapshot sudah disiapkan, sehingga training tidak mengakses Google Sheets.
`prepare_data.py` dapat membangun ulang snapshot ke directory **baru** dari CSV
sumber; download sumber hanya diperlukan untuk meregenerasi dataset.

Rencana dan bukti verifikasi ada di `docs/`.
