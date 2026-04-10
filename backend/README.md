# Backend for Autism FER Pipeline

This backend serves as the Python inference server for your `.pth` PyTorch model.

## What belongs here

- `main.py` — FastAPI server
- `requirements.txt` — Python dependencies
- `multiscale_vit_best.pth` — FER model checkpoint
- `eye_best.pkl` — eye tracking model checkpoint

## How to run

1. Create a Python virtual environment:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place your model files in this folder as:
   - `multiscale_vit_best.pth`
   - `eye_best.pkl`
4. Start the server:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

## API endpoint

- `POST /predict`
- form field: `file`
- response:
  - `top_emotion`
  - `confidence`
  - `emotions` (probability distribution)

## Frontend integration

From Next.js, send a `FormData` request to `http://localhost:8000/predict`.

Example:
```ts
const formData = new FormData();
formData.append('file', imageFile);
const response = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  body: formData,
});
const data = await response.json();
```
