# 💼 Fake Job Posting Detection using NLP + Machine Learning

## 📌 Project Overview

This project detects whether a job posting is **Real or Fake** using Natural Language Processing (NLP) and Machine Learning techniques.

The system analyzes job descriptions and predicts fraudulent job postings using **TF-IDF vectorization** and a **Linear Support Vector Machine (SVM)** model.

A Flask web application allows users to enter a job posting and receive a prediction in real time.

---

## 🚀 Features

- Fake job detection using NLP
- Text preprocessing (cleaning, stopword removal, lemmatization)
- TF-IDF feature extraction
- Machine Learning classification using Linear SVM
- Flask web application for prediction
- Streamlit interface (optional)
- Model saved using Joblib

---

## 📊 Dataset

- **Dataset:** Fake Job Postings Dataset (Kaggle)
- **Target Column:** `fraudulent`

### Features Used

- title
- location
- department
- company_profile
- description
- requirements
- benefits
- employment_type
- required_experience
- required_education
- industry
- function

---

## 🧠 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Flask
- Streamlit
- Joblib

---

## 🏗️ Project Workflow

```text
Data Collection
        ↓
Data Cleaning
        ↓
Handling Missing Values
        ↓
Text Preprocessing (NLP)
        ↓
TF-IDF Vectorization
        ↓
Train-Test Split
        ↓
Model Training (Linear SVM)
        ↓
Model Evaluation
        ↓
Save Model (.pkl)
        ↓
Flask Web Application
```

---

## ⚙️ Model Performance

After comparing multiple machine learning models, **Linear SVM** achieved the best performance.

| Metric | Value |
|--------|-------|
| Model | Linear SVM |
| Accuracy | **98.76%** |
| F1 Score | **0.8625** |
| Recall | **0.7976** |
| Feature Extraction | TF-IDF |

---

## 🧪 Example Prediction

### Input

```text
Work from home job.
Earn $5000 per week.
No experience required.
Apply immediately.
```

### Output

```text
⚠️ Fake Job Posting
```

---

## 💾 Model Saving

```python
import joblib

joblib.dump(best_svm, "model.pkl")
joblib.dump(tfidf, "tfidf.pkl")
```

---

## 🌐 Running the Flask Application

### Run the application

```bash
python app.py
```

### Open in your browser

```text
http://127.0.0.1:5000/
```

### Steps

1. Paste a job description.
2. Click **Predict**.
3. View whether the posting is **Real** or **Fake**.

---

## 📁 Project Structure

```text
Fake-Job-Detection/
│
├── app.py
├── train.py
├── model.pkl
├── tfidf.pkl
├── dataset.csv
├── templates/
│   └── index.html
├── static/
│   └── style.css
├── README.md
```

---

## 📌 Future Improvements

- Implement BERT or Transformer models
- Deploy the application using Render or Railway
- Improve recall for fake job detection
- Add URL-based job verification
- Enhance the user interface

---

## 👩‍💻 Author

**MUHSINA.K**

Data Science & AI Enthusiast

---
## 📸 Application Screenshot

![streamlit App](images/app.png)
## ⭐ Support

If you found this project useful, please consider giving it a ⭐ on GitHub.
