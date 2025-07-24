from sklearn.metrics import classification_report, accuracy_score
from tabulate import tabulate
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Load stop words Indonesia dari file eksternal
with open('stop_word_indonesia', encoding='utf-8') as f:
    indonesian_stop_words = [line.strip() for line in f if line.strip()]

# Load dataset
file_path = 'DatasetPesanPalsu2.csv'
df = pd.read_csv(file_path)
df['TEXT'] = df['TEXT'].astype(str).str.lower()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['TEXT'], df['LABEL'], test_size=0.2, random_state=42, stratify=df['LABEL']
)

# TF-IDF
tfidf = TfidfVectorizer(stop_words=indonesian_stop_words, max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Define models
models = {
    'Naive Bayes': MultinomialNB(),
    'SVM': SVC(kernel='linear', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

# Train and evaluate all models
results = {}
for name, model in models.items():
    print(f"\n>>> Training {name}...")
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=2, output_dict=True)
    results[name] = {'accuracy': accuracy, 'report': report, 'predictions': y_pred}

# Display detailed results for each model
for name, result in results.items():
    print(f"\n>>> Model: {name}")
    print(f"Akurasi: {result['accuracy']:.4f}")
    
    report = result['report']
    labels = [label for label in report if label not in ('accuracy', 'macro avg', 'weighted avg')]
    table = []
    for label in labels:
        row = [
            label,
            f"{report[label]['precision']:.2f}",
            f"{report[label]['recall']:.2f}",
            f"{report[label]['f1-score']:.2f}",
            int(report[label]['support'])
        ]
        table.append(row)

    table.append(['accuracy', '', '', f"{report['accuracy']:.2f}", int(np.sum([report[l]['support'] for l in labels]))])
    for avg in ['macro avg', 'weighted avg']:
        row = [
            avg,
            f"{report[avg]['precision']:.2f}",
            f"{report[avg]['recall']:.2f}",
            f"{report[avg]['f1-score']:.2f}",
            int(report[avg]['support'])
        ]
        table.append(row)

    print(tabulate(table, headers=['', 'precision', 'recall', 'f1-score', 'support'], tablefmt='grid'))

# Comparison table of all models
print("\n" + "="*60)
print("PERBANDINGAN HASIL KETIGA MODEL")
print("="*60)

comparison_table = []
for name, result in results.items():
    report = result['report']
    row = [
        name,
        f"{result['accuracy']:.4f}",
        f"{report['macro avg']['precision']:.4f}",
        f"{report['macro avg']['recall']:.4f}",
        f"{report['macro avg']['f1-score']:.4f}"
    ]
    comparison_table.append(row)

print(tabulate(comparison_table, 
               headers=['Model', 'Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1'], 
               tablefmt='grid'))

# Find best model
best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
print(f"\nMODEL TERBAIK: {best_model[0]} dengan akurasi {best_model[1]['accuracy']:.4f}")
