# SMS Phishing Detection - Guide How To Run The Program

## Software Requirements
1. XAMPP (for MySQL database)
2. Python 3.x (recommended 3.8+)
3. Pip (Python package manager)

---

## Steps

1. **Download the necessary library using:**
   ```
   pip install -r requirements.txt
   ```

2. **Start MySQL from XAMPP:**
   - Buka XAMPP Control Panel.
   - Klik **Start** pada MySQL.

3. **Import the database:**
   - Buka **phpMyAdmin** (biasanya di http://localhost/phpmyadmin).
   - Buat database baru dengan nama `smstriclass`.
   - Import file `data.sql` yang ada di folder `Back-end` ke database tersebut.

4. **Set up environment variables (opsional, jika ingin custom DB user/password):**
   - Edit file `.env` di folder `Back-end` jika ingin mengubah konfigurasi database (DB_USER, DB_PASS, dll).
   - Contoh isi `.env`:
     ```
     DB_USER=root
     DB_PASS=
     DB_HOST=localhost
     DB_PORT=3306
     DB_NAME=smstriclass
     ```

5. **Run the backend server:**
   - Buka terminal/cmd, arahkan ke folder `Back-end`.
   - Jalankan:
     ```
     python app.py
     ```
   - Server akan berjalan di `http://localhost:5001`

6. **(Opsional) Buka front-end:**
   - Buka file `index.html` atau `admin_page.html` di folder `front-end` menggunakan browser, atau letakkan di folder `htdocs` XAMPP jika ingin diakses via http://localhost/sms/front-end/index.html

7. **Test API:**
   - Gunakan Postman, curl, atau front-end untuk mencoba endpoint seperti `/predict`, `/admin-login`, dll.

---

**Catatan:**
- Pastikan MySQL sudah aktif sebelum menjalankan backend.
- Jika ada error koneksi database, cek konfigurasi di `.env` dan pastikan database sudah di-import.
- Untuk training ulang model, gunakan endpoint atau script yang sudah disediakan. 
