from flask import Flask, request, jsonify, session
from flask_cors import CORS
import joblib
import pandas as pd
import os
from datetime import timedelta, datetime
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
import re
import html
import csv
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from collections import defaultdict
import time

# ============================================================================
# 1. PREPARASI DATA DAN MODEL
# ============================================================================

# Load Indonesian stop words untuk preprocessing
def load_indonesian_stop_words():
    try:
        with open('stop_word_indonesia', 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print("Warning: stop_word_indonesia file not found, using default Indonesian stop words")
        return ['yang', 'dan', 'atau', 'dengan', 'untuk', 'dari', 'ke', 'di', 'pada', 'oleh']

indonesian_stop_words = load_indonesian_stop_words()

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = 'very-secret-key'
app.permanent_session_lifetime = timedelta(days=1)  # Session lasts for 1 day

# CORS Configuration
CORS(app, 
     supports_credentials=True,
     resources={r"/*": {
         "origins": ["http://localhost:8080", "http://localhost:8080/sms/front-end"],
         "methods": ["GET", "POST", "OPTIONS"],
         "allow_headers": ["Content-Type", "Authorization"],
         "expose_headers": ["Content-Type", "Authorization"],
         "supports_credentials": True
     }})

# ============================================================================
# 2. INPUT SANITIZATION (Keamanan)
# ============================================================================

# Fungsi sanitasi untuk membersihkan input dari user
def sanitize_text(text):
    if not text:
        return None
    # Remove any HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Escape HTML special characters
    text = html.escape(text)
    # Remove any remaining script tags
    text = re.sub(r'<script.*?>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
    return text

def sanitize_email(email):
    if not email:
        return None
    # Basic email validation
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_pattern, email):
        return None
    return sanitize_text(email)

def sanitize_category(category):
    if not category:
        return None
    # Only allow specific categories 
    allowed_categories = ['promo', 'normal', 'penipuan']
    category = sanitize_text(category)
    return category if category in allowed_categories else None

# ============================================================================
# 3. DATABASE CONFIGURATION
# ============================================================================

# Database configuration   
user = os.getenv('DB_USER', 'root')
pwd = os.getenv('DB_PASS', '')
host = os.getenv('DB_HOST', 'localhost')
port = os.getenv('DB_PORT', '3306')
db_name = os.getenv('DB_NAME', 'smstriclass')

app.config['SQLALCHEMY_DATABASE_URI'] = f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{db_name}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize DB
db = SQLAlchemy(app)

def wib_now():
    return datetime.utcnow() + timedelta(hours=7)

# Database Models
class Contribution(db.Model):
    __tablename__ = 'contributions'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=True)
    email = db.Column(db.String(100), nullable=True)
    sms_text = db.Column(db.Text, nullable=False)
    suggested_category = db.Column(db.String(50), nullable=False)
    status = db.Column(db.String(10), nullable=False, default='pending')
    created_at = db.Column(db.DateTime, default=wib_now)

class DatasetEntry(db.Model):
    __tablename__ = 'dataset_entries'
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    label = db.Column(db.String(50), nullable=False)
    created_at = db.Column(db.DateTime, default=wib_now)

class Admin(db.Model):
    __tablename__ = 'admin_tbl'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), nullable=False, unique=True)
    password = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# ============================================================================
# 4. MODEL LOADING (Load model yang sudah dilatih)
# ============================================================================

# Load model dan vectorizer yang sudah dilatih sebelumnya
def load_models():
    try:
        model = joblib.load('./best_phishing_model2.pkl')
        vectorizer = joblib.load('./best_tfidf_vectorizer2.pkl')
        return model, vectorizer
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None

model, vectorizer = load_models()

# ============================================================================
# 5. RATE LIMITING (Anti DDoS ringan)
# ============================================================================

# Rate limiting manual untuk mencegah brute force attack
request_history = defaultdict(list)

def check_rate_limit(ip, limit_seconds=1):
    """Check if IP has exceeded rate limit"""
    current_time = time.time()
    # Remove old requests
    request_history[ip] = [req_time for req_time in request_history[ip] 
                          if current_time - req_time < limit_seconds]
    
    # Check if too many requests
    if len(request_history[ip]) >= 1:  # Max 1 request per second
        return False
    
    # Add current request
    request_history[ip].append(current_time)
    return True

# ============================================================================
# 6. KLASIFIKASI SMS (Proses utama)
# ============================================================================

# Endpoint untuk klasifikasi SMS - Input → Preprocessing → Vectorization → Prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Input: Terima teks SMS dari user
        data = request.get_json()
        text = data.get('text')
        if not text:
            return jsonify({'error': 'No text provided'}), 400
            
        # 2. Preprocessing: Sanitasi input
        text = sanitize_text(text)
        if not text:
            return jsonify({'error': 'Invalid text provided'}), 400
            
        # 3. Model Check: Pastikan model tersedia
        if model is None or vectorizer is None:
            return jsonify({'error': 'Model not available. Please train the model first.'}), 500
            
        # 4. Vectorization: Ubah teks menjadi vektor numerik menggunakan TF-IDF
        vector = vectorizer.transform([text])
        
        # 5. Prediction: Model memprediksi kategori SMS
        prediction = model.predict(vector)
        
        # 6. Output: Kembalikan hasil klasifikasi
        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================================
# 7. MODEL TRAINING (Proses training ulang model)
# ============================================================================

# Train machine learning model dengan dataset yang diperbarui
def train_machine_learning_model():
    """Train machine learning model with updated dataset"""
    try:
        # 1. Load dataset dari CSV
        csv_path = os.path.join(os.path.dirname(__file__), 'DatasetPesanPalsu2.csv')
        if not os.path.exists(csv_path):
            return False, "CSV file not found"
            
        df = pd.read_csv(csv_path)
        if len(df) == 0:
            return False, "Dataset is empty"
            
        # 2. Validasi kolom yang diperlukan
        if 'TEXT' not in df.columns or 'LABEL' not in df.columns:
            return False, "CSV file missing required columns (TEXT, LABEL)"
            
        # 3. Preprocessing: Lowercase semua teks
        df['TEXT'] = df['TEXT'].str.lower()
        
        X = df['TEXT']
        y = df['LABEL']
        
        # 4. Validasi jumlah sampel
        min_samples_per_class = y.value_counts().min()
        if min_samples_per_class < 2:
            return False, "Not enough samples per class for training"
            
        # 5. Split data: Training (80%) dan Testing (20%)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # 6. Feature extraction: TF-IDF Vectorization
        tfidf = TfidfVectorizer(stop_words=indonesian_stop_words, max_features=5000)
        X_train_tfidf = tfidf.fit_transform(X_train)
        X_test_tfidf = tfidf.transform(X_test)
        
        # 7. Model selection: Test 3 algoritma berbeda
        models = {
            "Naive Bayes": MultinomialNB(),
            "SVM": SVC(),
            "Random Forest": RandomForestClassifier()
        }
        
        # 8. Train dan evaluasi model
        best_model = None
        best_accuracy = 0
        best_model_name = ""
        
        for model_name, model in models.items():
            model.fit(X_train_tfidf, y_train)
            y_pred = model.predict(X_test_tfidf)
            accuracy = accuracy_score(y_test, y_pred)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_model_name = model_name
        
        # 9. Save model terbaik
        model_path = os.path.join(os.path.dirname(__file__), 'best_phishing_model2.pkl')
        vectorizer_path = os.path.join(os.path.dirname(__file__), 'best_tfidf_vectorizer2.pkl')
        
        joblib.dump(best_model, model_path)
        joblib.dump(tfidf, vectorizer_path)
        
        return True, "Model trained successfully!"
        
    except Exception as e:
        return False, f"Training error: {str(e)}"

# ============================================================================
# 8. ADMIN AUTHENTICATION
# ============================================================================

# Admin login menggunakan session Flask
@app.route('/admin-login', methods=['POST'])
def admin_login():
    # Manual rate limiting
    client_ip = request.remote_addr
    print(f"[DEBUG] Admin login attempt from IP: {client_ip}")
    
    if not check_rate_limit(client_ip, 1):  # 1 request per second
        print(f"[RATE LIMITED] IP: {client_ip}")
        return jsonify({'success': False, 'message': 'Rate limit exceeded. Please wait 1 second.'}), 429
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'message': 'No data provided'}), 400
            
        # Sanitasi credentials
        username = sanitize_text(data.get('username'))
        password = sanitize_text(data.get('password'))
        
        # Validasi input
        if not username or not password:
            return jsonify({'success': False, 'message': 'Username and password are required'}), 400
            
        admin = Admin.query.filter_by(username=username).first()
        if admin and admin.password == password:
            session.permanent = True
            session['admin_logged_in'] = True
            return jsonify({'success': True})
            
        return jsonify({'success': False, 'message': 'Invalid credentials'}), 401
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/check-session', methods=['GET'])
def check_session():
    try:
        if session.get('admin_logged_in'):
            return jsonify({'logged_in': True})
        return jsonify({'logged_in': False})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/logout', methods=['POST'])
def logout():
    try:
        session.clear()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================================
# 9. DATA MANAGEMENT (CRUD operations)
# ============================================================================

def to_wib(dt):
    return dt.isoformat() if dt else None

# Ambil semua kontribusi yang belum disetujui
@app.route('/get-all-contribution', methods=['GET'])
def get_all_contribution():
    try:
        # Check admin session
        if not session.get('admin_logged_in'):
            return jsonify({'error': 'Unauthorized'}), 401
            
        # Only show contributions that are not approved
        contributions = Contribution.query.filter(Contribution.status != 'approved').order_by(Contribution.created_at.desc()).all()
        result = [{
            'id': c.id,
            'created_at': to_wib(c.created_at),
            'sms_text': c.sms_text,
            'name': c.name,
            'email': c.email,
            'suggested_category': c.suggested_category,
            'status': c.status
        } for c in contributions]
        
        return jsonify({'data': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Ambil semua data dataset
@app.route('/get-all-dataset', methods=['GET'])
def get_all_dataset():
    try:
        # Check admin session
        if not session.get('admin_logged_in'):
            return jsonify({'error': 'Unauthorized'}), 401
            
        entries = DatasetEntry.query.order_by(DatasetEntry.created_at.desc()).all()
        result = [{
            'created_at': to_wib(e.created_at),
            'text': e.text,
            'label': e.label
        } for e in entries]
        
        return jsonify({'data': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Update status kontribusi (approve/reject)
@app.route('/update-status-contribution', methods=['POST'])
def update_status_contribution():
    try:
        # Check admin session
        if not session.get('admin_logged_in'):
            return jsonify({'error': 'Unauthorized'}), 401
            
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        contribution_id = data.get('id')
        new_status = sanitize_text(data.get('status'))
        
        # Validasi input
        if not contribution_id or not new_status:
            return jsonify({'error': 'ID and status are required'}), 400
            
        if new_status not in ['pending', 'approved', 'rejected']:
            return jsonify({'error': 'Invalid status'}), 400
            
        contribution = Contribution.query.get(contribution_id)
        if not contribution:
            return jsonify({'error': 'Contribution not found'}), 404
            
        if new_status == 'approved':
            # Pindahkan ke dataset_entries jika belum ada
            existing = DatasetEntry.query.filter_by(text=contribution.sms_text).first()
            if not existing:
                dataset_entry = DatasetEntry(
                    text=contribution.sms_text,
                    label=contribution.suggested_category
                )
                db.session.add(dataset_entry)
            # Hapus dari tabel contributions
            db.session.delete(contribution)
        else:
            contribution.status = new_status

        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Tambahkan data ke dataset dan train ulang model
@app.route('/add-to-dataset', methods=['POST'])
def add_to_dataset():
    try:
        # Check admin session
        if not session.get('admin_logged_in'):
            return jsonify({'error': 'Unauthorized'}), 401
            
        # Get all approved contributions
        approved_contributions = Contribution.query.filter_by(status='approved').all()

        for contrib in approved_contributions:
            # Check if entry already exists
            existing = DatasetEntry.query.filter_by(text=contrib.sms_text).first()
            if not existing:
                dataset_entry = DatasetEntry(
                    text=contrib.sms_text,
                    label=contrib.suggested_category
                )
                db.session.add(dataset_entry)
            # Hapus dari tabel contributions
            db.session.delete(contrib)

        db.session.commit()

        # Update CSV setelah data dipindahkan ke dataset_entries
        csv_path = os.path.join(os.path.dirname(__file__), 'DatasetPesanPalsu2.csv')
        existing_rows = set()
        if os.path.exists(csv_path):
            with open(csv_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                next(reader, None)  # skip header
                for row in reader:
                    if len(row) >= 2:
                        existing_rows.add((row[0], row[1]))

        # Ambil data dari database yang belum ada di CSV
        new_rows = []
        for entry in DatasetEntry.query.all():
            if (entry.text, entry.label) not in existing_rows:
                new_rows.append([entry.text, entry.label])

        # Append data baru ke CSV
        with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            if os.stat(csv_path).st_size == 0:
                writer.writerow(['TEXT', 'LABEL'])
            for row in new_rows:
                writer.writerow(row)

        # Train new machine learning model
        training_success, training_message = train_machine_learning_model()
        
        if training_success:
            # Reload the models after training
            global model, vectorizer
            model, vectorizer = load_models()
            return jsonify({'success': True, 'message': f'Dataset updated and {training_message}'})
        else:
            return jsonify({'success': False, 'error': f'Dataset updated but training failed: {training_message}'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Submit kontribusi dari user
@app.route('/submit-contribution', methods=['POST'])
def submit_contribution():
    try:
        data = request.get_json()
        print("Received data:", data)
        
        if not data:
            print("No data received in request")
            return jsonify({'error': 'No data provided'}), 400
            
        # Sanitasi semua input
        sms_text = sanitize_text(data.get('sms_text') or data.get('text'))
        category = sanitize_category(data.get('category'))
        name = sanitize_text(data.get('name'))
        email = sanitize_email(data.get('email'))
        
        print("Extracted fields:", {
            'sms_text': sms_text,
            'category': category,
            'name': name,
            'email': email
        })
        
        # Validasi field yang diperlukan
        if not sms_text:
            print("Missing or invalid SMS text")
            return jsonify({'error': 'Valid SMS text is required'}), 400
            
        if not category:
            print("Missing or invalid category")
            return jsonify({'error': 'Valid category (Spam or Ham) is required'}), 400
            
        contribution = Contribution(
            sms_text=sms_text,
            suggested_category=category,
            name=name,
            email=email
        )
        db.session.add(contribution)
        db.session.commit()
        
        print("Successfully saved contribution")
        return jsonify({'success': True, 'message': 'Contribution submitted'})
    except Exception as e:
        print("Submission error:", str(e))
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

# ============================================================================
# 10. APPLICATION STARTUP
# ============================================================================

if __name__ == '__main__':
    try:
        # Ensure tables are created
        with app.app_context():
            db.create_all()
        app.run(debug=True, port=5001)
    except Exception as e:
        print("Error during startup:", e)
