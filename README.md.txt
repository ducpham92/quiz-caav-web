# Quiz CAAV Web (Streamlit)

## Chạy local
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
set QUIZ_MODE=streamlit   # PowerShell: $env:QUIZ_MODE="streamlit"
streamlit run app.py

## CSV format
- File name: {CAT}_Module{N}.csv (e.g., B1_Module1.csv)
- Columns: Question, Option A, Option B, (Option C ...), Correct Answer
- Encoding: UTF-8 BOM
