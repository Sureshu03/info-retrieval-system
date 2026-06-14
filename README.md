# End-to-End Information Retrieval System (Streamlit)

This project implements the complete IR workflow from the assignment in `Task.txt`:
- Dataset upload/view
- Preprocessing + inverted index
- Phrase querying (Biword vs Positional)
- Dictionary search benchmark (BST vs B-Tree)
- Tolerant retrieval (wildcard, spelling/edit distance, k-gram, phonetic)
- Inference and discussion panel

## Project Structure
- `app.py` → Streamlit application
- `requirements.txt` → dependencies
- `data/docs/*.txt` → bundled sample dataset
- `data/relevance_judgments.csv` → sample relevance labels for evaluation
- `Report.md` → implementation report + inferences
- `Demo_Evidence.md` → demo evidence checklist

## Setup
1. Create virtual environment (recommended)
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Run app:
   - `streamlit run app.py`

## Notes
- You can use bundled dataset or upload your own `.txt` files from the UI.
- Capture screenshots of each section in the app for final report submission.
