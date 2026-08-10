# AI Content Trust & Spam Detection System

A desktop application that analyzes text, PDFs, images, and videos for
**spam, phishing, clickbait, toxic content, and fake reviews** using
classical NLP and machine learning (Naive Bayes, Logistic Regression,
SVM). For each analysis, it shows a confidence score, risk level
(Low/Medium/High), an overall credibility score, and a plain-English
explanation of *why* content was flagged — including the specific
words that influenced the decision.

Built with **PySide6**, fully **offline** after the models are
trained, with results stored locally in **SQLite**.

> The app classifies content based on learned linguistic patterns from
> training data — it does not verify the real-world factual truth of
> any content.

---

## Features

- **4 input types:** paste plain text, or upload a PDF, image (OCR),
  or video (offline speech-to-text)
- **5 detection categories:** Spam, Phishing, Clickbait, Toxic
  Content, Fake Reviews
- **Explainable results:** confidence %, risk level, credibility
  score, highlighted influencing words, and a plain-English
  explanation per flagged category
- **History:** every analysis is saved locally and browsable/deletable
- **Reports:** export your full history as PDF or CSV
- **Light/Dark theme**, remembered between sessions
- **Fully offline** at runtime — no data ever leaves your machine

## Screens

| Tab | Purpose |
|---|---|
| Home | Welcome screen |
| Analyze | Paste text or upload a file, run analysis, view results |
| History | Browse and delete past analyses |
| Reports | Export history as PDF or CSV |
| Settings | Switch between Light and Dark theme |

---

## Getting Started

### 1. Install dependencies
```
pip install -r requirements.txt
```

### 2. Install Tesseract OCR (for image analysis)
This is a separate program, not just a Python package:
- **Windows:** install from https://github.com/UB-Mannheim/tesseract/wiki, then add its install folder to your system PATH
- **Mac:** `brew install tesseract`
- **Linux:** `sudo apt install tesseract-ocr`

### 3. (Optional) Enable video analysis
Video speech-to-text uses an offline Vosk model (not included, due to
size). Download a small English model from
https://alphacephei.com/vosk/models, unzip it, and place it at:
`models/vosk-model-small-en-us`. Until then, video files will show a
clear message instead of crashing — text/PDF/image analysis work fine
without this step.

### 4. Run the app
```
python main.py
```

---

## Training Your Own Models

The app ships with trained models (`models/*.pkl`) built from small
**synthetic sample datasets** included in `training/datasets/`, so
everything is runnable and testable out of the box. For real-world
accuracy, replace these with real public datasets:

1. Follow `training/datasets/DATASETS.md` to download and convert each
   of the 5 categories' datasets into the standard `text,label` CSV
   format.
2. Re-run training:
   ```
   python training/train_models.py
   ```
   This prints an Accuracy/Precision/Recall/F1 comparison table per
   category and saves the best-performing model.

No application code needs to change — the app always loads whatever
is currently in `models/`.

---

## Running Tests

```
pytest tests/ -v
```
**39 tests** across 8 files, covering input extraction, preprocessing,
dataset loading, model training logic, the ML engine (predictions,
risk scoring, explanations), database operations, report generation,
and theming.

---

## Packaging as a Standalone Executable

To build a distributable executable that runs without Python installed:
```
pip install pyinstaller
pyinstaller app.spec
```
The finished executable appears in `dist/`. Build on the same OS
you're targeting (PyInstaller doesn't cross-compile) — e.g. run this
on Windows to get a `.exe`. The build bundles the trained models and
NLTK's data files, so the packaged app runs fully offline, including
on a machine with no internet access at all.

---

## Project Structure

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full
architecture, folder structure, data flow diagrams, database schema,
and team module breakdown.

## Known Limitations

- Included datasets are small synthetic samples for pipeline testing —
  see "Training Your Own Models" above to use real data.
- Video analysis requires a manually downloaded Vosk model.
- This is a pattern-detection tool, not a fact-checker.

## License / Academic Use

Built as a university NLP group project. Public datasets used for
training are subject to their own individual licenses — see
`training/datasets/DATASETS.md` for sources.
