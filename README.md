# Churn Analyzer

Upload a customer CSV in the browser. A Python (Flask) backend cleans it, runs EDA, trains four models (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting), scores every customer and returns:

1. Cleaned dataset (download)
2. EDA summary and insights
3. Model building steps
4. Churn scoring and risk levels (High >= 0.70, Medium 0.40-0.70, Low < 0.40)
5. EDA charts
6. Model comparison table (+ ROC curves)
7. Model evaluation (confusion matrix, feature importance)

```
churn-prediction-app/
  docs/        frontend (index.html, style.css, script.js, config.js)  -> GitHub Pages
  backend/     Python API (app.py, pipeline.py, requirements.txt)
  sample_data/ Telco-Customer-Churn.csv
  render.yaml  one-click backend deploy on Render
```

## Run on your computer
```
cd backend
python -m pip install -r requirements.txt
python app.py
```
Then open `docs/index.html` in a browser (or run `python -m http.server 8080` inside `docs` and open http://localhost:8080). Click "Use the Telco sample".

## Put it on GitHub
GitHub Pages hosts only static files, so it serves the frontend. The Python backend must run somewhere else.

1. Create a GitHub repo and push this folder.
2. Repo Settings -> Pages -> Deploy from branch -> `main` and folder `/docs`.
3. Deploy the backend free on Render: New -> Blueprint -> pick the repo (it reads `render.yaml`). Copy the URL it gives you.
4. Edit `docs/config.js` and set `window.CHURN_API_BASE` to that URL. Commit and push.

The free Render plan sleeps after 15 minutes idle; the first request can take about a minute. For a viva, run the backend locally and keep `config.js` on `http://localhost:5000`.

## Your CSV needs
One row per customer, a column with exactly two values for churn (Yes/No, 1/0, True/False), 50 to 50,000 rows, up to 25 MB.
