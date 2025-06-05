from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
import joblib, json, os
import pandas as pd

# ─────────────────────────────────────────────
# Load model + feature list
# ─────────────────────────────────────────────
MODEL_DIR = os.getenv("MODEL_DIR", "saved_model")
model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
with open(os.path.join(MODEL_DIR, "features.json")) as f:
    FEATURES = json.load(f)  # ["careerLengthYears", "supplementRatio", "highestRank", "gender"]

app = FastAPI(title="Military Pension Amount Predictor")

# ─────────────────────────────────────────────
# Pydantic input schema
# ─────────────────────────────────────────────
class FeatureInput(BaseModel):
    careerLengthYears: int = Field(..., ge=10, le=40)
    supplementRatio:   float = Field(..., ge=0,  le=1)
    highestRank:      int = Field(..., ge=0,  le=15)
    gender:           int = Field(..., ge=0,  le=1)

# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve interactive HTML form."""
    # Build <option> tags for career length
    options = "\n".join(f"<option value='{y}'>{y}</option>" for y in range(10, 41))

    html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Military Pension Amount Predictor</title>
  <style>
    body{font-family:Arial,Helvetica,sans-serif;margin:40px;background:#f5f7fa;color:#333;}
    h1{color:#0b7285;}
    form{display:flex;flex-direction:column;gap:1rem;max-width:420px;}
    label{display:flex;flex-direction:column;font-weight:600;}
    input[type=range]{width:100%;}
    fieldset{border:none;display:flex;gap:1rem;align-items:center;}
    legend{font-weight:600;margin-bottom:0.25rem;}
    button{padding:0.5rem 1rem;border:none;border-radius:6px;background:#0b7285;color:white;font-size:1rem;cursor:pointer;}
    button:hover{background:#086071;}
    #result{margin-top:1.5rem;font-size:1.25rem;font-weight:700;}
  </style>
</head>
<body>
  <h1>Military Pension Amount Predictor</h1>
  <form id="predictForm">
    <label>Career Length (years)
      <select id="careerLengthYears" name="careerLengthYears">{{OPTIONS}}</select>
    </label>

    <label>Supplement Ratio: <span id="suppVal">0.50</span>
      <input type="range" id="supplementRatio" name="supplementRatio" min="0" max="1" step="0.01" value="0.50">
    </label>

    <label>Highest Rank: <span id="rankVal">1</span>
      <input type="range" id="highestRank" name="highestRank" min="0" max="15" step="1" value="1">
    </label>

    <fieldset>
      <legend>Gender</legend>
      <label><input type="radio" name="gender" value="0" checked> 👨</label>
      <label><input type="radio" name="gender" value="1"> 👩</label>
    </fieldset>

    <button type="submit">Predict</button>
  </form>

  <div id="result"></div>

  <script>
    const form      = document.getElementById('predictForm');
    const supp      = document.getElementById('supplementRatio');
    const suppVal   = document.getElementById('suppVal');
    const rank      = document.getElementById('highestRank');
    const rankVal   = document.getElementById('rankVal');

    supp.addEventListener('input', () => suppVal.textContent = supp.value);
    rank.addEventListener('input', () => rankVal.textContent   = rank.value);

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const data = {
        careerLengthYears: parseInt(document.getElementById('careerLengthYears').value),
        supplementRatio:   parseFloat(document.getElementById('supplementRatio').value),
        highestRank:       parseInt(document.getElementById('highestRank').value),
        gender:            parseInt(document.querySelector('input[name="gender"]:checked').value)
      };

      const res  = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
      });
      const json = await res.json();
      document.getElementById('result').textContent =
        'Predicted nominal amount: \u20AC ' +
        json.predicted_nominal.toLocaleString(undefined, {
          minimumFractionDigits: 2,
          maximumFractionDigits: 2
        });
    });
  </script>
</body>
</html>
"""
    html = html_template.replace("{{OPTIONS}}", options)
    return HTMLResponse(html)

@app.post("/predict")
async def predict(data: FeatureInput):
    df = pd.DataFrame([data.dict()])[FEATURES]
    pred = model.predict(df)[0]
    return JSONResponse({"predicted_nominal": float(pred)})
