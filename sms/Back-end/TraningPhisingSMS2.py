import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load Indonesian stop words from file
def load_indonesian_stop_words():
    try:
        with open('stop_word_indonesia', 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print("Warning: stop_word_indonesia file not found, using default Indonesian stop words")
        return ['yang', 'dan', 'atau', 'dengan', 'untuk', 'dari', 'ke', 'di', 'pada', 'oleh']

indonesian_stop_words = load_indonesian_stop_words()

# Load and preprocess data
df = pd.read_csv('./DatasetPesanPalsu2.csv')
df['TEXT'] = df['TEXT'].str.lower()

# Print dataset statistics
print("\nDataset Statistics:")
print(df['LABEL'].value_counts())
print("\nTotal samples:", len(df))

X = df['TEXT']
y = df['LABEL']

# Check if we have enough samples for stratification
min_samples_per_class = y.value_counts().min()
print(f"\nMinimum samples in a class: {min_samples_per_class}")

try:
    if min_samples_per_class < 2:
        print("\nWarning: Not enough samples for stratification. Using simple random split.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        print("\nUsing stratified split.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Feature extraction
    tfidf = TfidfVectorizer(stop_words=indonesian_stop_words, max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Define models
    models = {
        "Naive Bayes": MultinomialNB(),
        "SVM": SVC(),
        "Random Forest": RandomForestClassifier()
    }

    # Train and evaluate models
    best_model = None
    best_accuracy = 0
    results = {}

    print("\nTraining Models:")
    for model_name, model in models.items():
        print(f"\nMelatih {model_name}...")
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        results[model_name] = {"Accuracy": accuracy}

        for label in ['promo', 'normal', 'penipuan']:
            if label in report:
                results[model_name].update({
                    f"Precision ({label})": report[label]['precision'],
                    f"Recall ({label})": report[label]['recall'],
                    f"F1 ({label})": report[label]['f1-score']
                })
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = model_name

    # Save best model
    print(f"\nModel terbaik: {best_model_name} dengan akurasi {best_accuracy:.4f}")
    joblib.dump(best_model, 'best_phishing_model2.pkl')
    joblib.dump(tfidf, 'best_tfidf_vectorizer2.pkl')

    # Print results
    results_df = pd.DataFrame(results).T
    print("\nPerbandingan Model:")
    print(results_df)

except Exception as e:
    print(f"\nError during training: {str(e)}")
    raise
