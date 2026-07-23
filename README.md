# CyberForge Backend API

This directory contains the Python Flask backend API and the intent-matching search engine for the CyberForge chatbot.

## File Structure
- `app.py`: Flask API server, session memory manager, and natural intent matching engine.
- `DataSet.xlsx`: Excel sheet containing the source query mappings, intents, and keywords.
- `dataset.json`: JSON export of the intent dataset used directly by `app.py`.
- `build_dataset.py`: Script to regenerate `dataset.json` from `DataSet.xlsx`.
- `requirements.txt`: Python dependencies.
- `start.bat`: Starts the Flask backend server locally on port `6161`.
- `responses/`: Directory containing text file responses for matched intents.

## Local Development Setup

### 1. Install Dependencies
Ensure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Start Backend Server
Run the Flask server:
```bash
python app.py
```
*(Or double-click `start.bat` on Windows)*.

The backend API will be available at [http://localhost:6161](http://localhost:6161).

## Updating the Dataset
If you modify `DataSet.xlsx` to add or update questions and keywords:
1. Ensure all `File Name` entries have a corresponding `.txt` file inside the `responses/` folder.
2. Regenerate the JSON dataset by running:
   ```bash
   python build_dataset.py
   ```
