# Architecture Documentation
## AI Content Trust & Spam Detection System

This document is the final, consolidated architecture reference for
the project — suitable for academic submission or as a walkthrough
guide in a technical interview.

---

## 1. Overview

A desktop application (PySide6) that analyzes text, PDFs, images, and
videos for spam, phishing, clickbait, toxic content, and fake reviews
using classical NLP/ML (Naive Bayes, Logistic Regression, SVM), and
displays a confidence score, risk level, credibility score, and a
plain-English explanation — fully offline after the models are trained.

## 2. High-Level Architecture

Two layers: UI and Core Logic. No controller layer, no interfaces/
plugin architecture — the UI calls `core/` functions directly.

```
┌───────────────────────────────────────────┐
│              UI Layer (PySide6)             │
│  Home | Analyze | History | Reports | Settings │
└───────────────────┬───────────────────────┘
                    │ direct function calls
┌───────────────────▼───────────────────────┐
│                Core Logic Layer             │
│  input_handler.py   → extracts text from     │
│                        text/pdf/image/video  │
│  preprocessing.py   → cleans & tokenizes text │
│  ml_engine.py       → loads models, predicts, │
│                        scores, explains       │
│  database.py        → all SQLite operations   │
│  report_generator.py → PDF/CSV export         │
└───────────────────┬───────────────────────┘
                    │
┌───────────────────▼───────────────────────┐
│    Saved Model Files (.pkl) + SQLite DB      │
└───────────────────────────────────────────┘
```

## 3. Folder Structure

```
ai-content-trust-detector/
├── main.py                     # Entry point
├── app.spec                    # PyInstaller packaging config
├── requirements.txt
├── README.md
├── app_data.db                 # created at runtime (not in version control)
│
├── ui/
│   ├── main_window.py          # 5-tab navigation shell
│   ├── analyze_screen.py       # input + results display
│   ├── history_screen.py       # past analyses table
│   ├── reports_screen.py       # PDF/CSV export
│   ├── settings_screen.py      # theme preference
│   └── theme.py                # light/dark stylesheets
│
├── core/
│   ├── input_handler.py        # text/pdf/image/video -> plain text
│   ├── preprocessing.py        # clean_and_tokenize()
│   ├── ml_engine.py            # MLEngine class - predict/explain
│   ├── database.py             # all SQLite reads/writes
│   └── report_generator.py     # export_csv() / export_pdf()
│
├── models/                     # trained artifacts (.pkl)
│   ├── spam_model.pkl / spam_vectorizer.pkl
│   ├── phishing_model.pkl / phishing_vectorizer.pkl
│   ├── clickbait_model.pkl / clickbait_vectorizer.pkl
│   ├── toxic_model.pkl / toxic_vectorizer.pkl
│   └── fake_review_model.pkl / fake_review_vectorizer.pkl
│
├── training/
│   ├── train_models.py         # trains & compares NB/LR/SVM, saves best
│   ├── dataset_loader.py       # loads standardized per-category CSVs
│   └── datasets/
│       ├── DATASETS.md         # download + conversion guide per category
│       ├── spam/data.csv
│       ├── phishing/data.csv
│       ├── clickbait/data.csv
│       ├── toxic/data.csv
│       └── fake_review/data.csv
│
├── docs/
│   └── ARCHITECTURE.md         # this file
│
└── tests/
    ├── test_input_handler.py
    ├── test_preprocessing.py
    ├── test_dataset_loader.py
    ├── test_train_models.py
    ├── test_ml_engine.py
    ├── test_database.py
    ├── test_report_generator.py
    └── test_theme.py
```

## 4. Data Flow

```
User Input (text typed, or file: pdf/image/video)
        │
        ▼
 input_handler.py  ──►  plain text
        │
        ▼
 ml_engine.py (calls preprocessing.py internally)
        │       cleans text → TF-IDF vectors → per-category prediction
        ▼
 Prediction Result: confidence + risk level + credibility score +
                     highlighted keywords + explanation
        │
        ├──► UI (Analyze Screen) shows results immediately
        │
        └──► database.py  ──►  SQLite (analysis_history table)
                                        │
                                        ▼
                              report_generator.py ──► PDF/CSV
```

## 5. Module Interaction

```
main.py
  └── ui.main_window.MainWindow
        ├── ui.analyze_screen.AnalyzeScreen
        │      ├── core.input_handler.extract_text()
        │      ├── core.ml_engine.MLEngine.predict()   (cleans + predicts)
        │      └── core.database.save_analysis()
        │
        ├── ui.history_screen.HistoryScreen
        │      └── core.database.get_history() / delete_analysis()
        │
        ├── ui.reports_screen.ReportsScreen
        │      ├── core.database.get_history()
        │      └── core.report_generator.export_pdf() / export_csv()
        │
        └── ui.settings_screen.SettingsScreen
               └── core.database.get_setting() / save_setting()
                   ui.theme.apply_theme()
```

## 6. Database Design (SQLite — `app_data.db`)

```sql
CREATE TABLE analysis_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    input_type TEXT NOT NULL,        -- 'text' | 'pdf' | 'image' | 'video'
    analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    prediction TEXT NOT NULL,        -- e.g. 'Spam, Clickbait' or 'Safe'
    confidence REAL NOT NULL,        -- 0-100, highest among flagged categories
    credibility_score REAL NOT NULL, -- 0-100
    risk_level TEXT NOT NULL         -- 'Low' | 'Medium' | 'High'
);

CREATE TABLE settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
```

## 7. Machine Learning Pipeline

For each of the 5 categories (spam, phishing, clickbait, toxic, fake_review):
1. Load the category's dataset (`training/dataset_loader.py`)
2. Clean every text sample with `core/preprocessing.py`'s
   `clean_and_tokenize()` — the exact same function used at prediction
   time, which keeps training and inference consistent
3. Split 80/20 train/test (stratified)
4. Vectorize with TF-IDF (fit only on training data)
5. Train and compare Naive Bayes, Logistic Regression, and SVM on
   Accuracy / Precision / Recall / F1 / Confusion Matrix
6. Save only the best model (by F1) + its vectorizer to `models/`

**Explainability:** rather than a separate library (SHAP/LIME), the
ML Engine reads each model's own learned feature weights (`coef_` for
Logistic Regression and calibrated SVM, `feature_log_prob_` for Naive
Bayes) and reports which of the input's own words had the highest
weight toward the "flagged" class — the basis for the highlighted
words and explanation sentence.

**Risk level thresholds:** confidence < 50% → Low, 50–80% → Medium,
> 80% → High.

**Credibility score:** 100 minus the average confidence across all
flagged categories (100 if nothing was flagged).

## 8. Team Module Breakdown

| Module | Owner | Files | Responsibility |
|---|---|---|---|
| A — Input & Preprocessing | Student 1 | `core/input_handler.py`, `core/preprocessing.py` | Extract text from any input type; clean/tokenize it |
| B — ML Engine | Student 2 | `training/train_models.py`, `core/ml_engine.py` | Train/compare/save models; predict, score, explain at runtime |
| C — Database & Reports | Student 3 | `core/database.py`, `core/report_generator.py` | SQLite persistence; PDF/CSV export |
| D — User Interface | Student 4 | `ui/*.py` | All 5 screens; wires user actions to Modules A/B/C |

## 9. SDLC

Simple iterative model: 2-week iterations, each ending in a working,
demoable state, with a short integration check after each iteration.
No formal Scrum ceremonies — appropriately lightweight for a 4-person,
one-semester team project.

## 10. Development Roadmap (Milestones — all completed)

1. Setup & Skeleton — project structure, database schema, empty 5-tab UI
2. Input Handling — text/PDF/image/video → plain text
3. Preprocessing + Dataset Collection — `clean_and_tokenize()`, standardized dataset format
4. Model Training & Comparison — NB/LR/SVM trained and compared per category
5. ML Engine & Explainability — predictions, confidence, risk, credibility, explanations
6. Full UI Integration — Analyze/History/Reports fully wired end-to-end
7. Settings, Polish & Testing — theme preference, broader test coverage, bug fixes
8. Documentation & Final Demo Prep — this document, final README, packaging

## 11. Testing Summary

39 automated tests across 8 test files (`pytest tests/ -v`), covering:
- Input extraction (text/PDF/OCR/error handling)
- Preprocessing (cleaning, tokenizing, lemmatization)
- Dataset loading and label validation
- Model training/evaluation logic
- ML Engine logic (risk thresholds, credibility scoring, explanations)
  using small hand-built models, independent of current dataset size
- Database CRUD (history + settings)
- Report generation (CSV + PDF)
- Theme application

## 12. Known Limitations

- **Training datasets are currently small synthetic samples**, not the
  full real public datasets. The code and pipeline are fully working
  and tested; accuracy will improve substantially once real datasets
  (see `training/datasets/DATASETS.md`) replace the placeholders — no
  code changes required.
- Video speech-to-text requires a manually downloaded offline Vosk
  language model (not bundled, due to its size).
- The app classifies based on learned linguistic patterns only — it
  does not verify the real-world factual truth of any content.
