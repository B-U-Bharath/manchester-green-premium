# Quantifying the Green Premium

Does an EPC rating actually move a house price in Greater Manchester, once you account for location, size, and property type? That's the question behind this project. MSc Data Science & Analytics dissertation, Maynooth University. 363,250 real property transactions, three UK government datasets, an XGBoost/Random Forest ensemble, and a live web app.

**Live app:** https://manchesterhouseprices.streamlit.app

## The question

UK homes get rated A (most efficient) to G (least efficient) on their EPC certificate. The common assumption is that a better rating means a higher sale price. I wanted to test that against real transaction data instead of taking it on faith, and the answer turned out to be more complicated than a single headline number.

## Data

| Source | Role |
|---|---|
| HM Land Registry Price Paid Data (2018–2026) | Sale prices |
| EPC Domestic Certificates (DLUHC) | Energy ratings |
| ONS National Statistics Postcode Lookup | Coordinates |

None of these three datasets share a common ID, so I linked them with a three-tier cascade match: postcode + house number first, then postcode + street name, then postcode alone as a last resort. That got a 66.8% match rate across 369,635 initial records, cleaned down to 363,250 usable rows.

## Pipeline

| Phase | Notebook | What it does |
|---|---|---|
| 1–2 | [`Phase1_2_DataEngineering.ipynb`](Phase1_2_DataEngineering.ipynb) | Merges the three datasets, deduplicates, builds 19 model features (spatial lag price, distance to city centre and nearest station, property age bands, tenure) |
| 3 | [`Phase3_EDA.ipynb`](Phase3_EDA.ipynb) | Exploratory analysis: price trends, spatial distribution, Moran's I spatial autocorrelation (I = 0.4226, p = 0.001) |
| 4 | [`Phase4_Modelling_Final.ipynb`](Phase4_Modelling_Final.ipynb) | Ridge, Random Forest, and Optuna-tuned XGBoost, trained on a price-index-adjusted target so the model isn't just picking up market-wide price growth over the sample period |
| 5 | [`Phase5_GreenPremiumAnalysis.ipynb`](Phase5_GreenPremiumAnalysis.ipynb) | Isolates the EPC effect itself, with bootstrapped confidence intervals, alpha sensitivity checks, and a breakdown by property type |
| 6 | [`app.py`](app.py), [`pages/`](pages), [`utils.py`](utils.py) | The deployed Streamlit app: price predictor, Green Premium map, model performance dashboard |

## Results

Held-out test set, 72,337 transactions, reported in real sale-date £ terms:

| Model | MAPE | R² |
|---|---|---|
| Ridge | 24.10% | 0.665 |
| Random Forest | 19.46% | 0.761 |
| XGBoost (Optuna-tuned) | 18.80% | 0.776 |
| Ensemble (XGBoost + Random Forest) | 18.73% | 0.777 |

On the Green Premium question itself: a simple linear model shows almost nothing, close to a zero effect. The signal only shows up once you let the model capture non-linearity. The XGBoost-based analysis found a small but genuine positive effect moving from an EPC D to a B, and it's concentrated in terraced homes and flats. Detached and semi-detached properties go the other way, a small negative effect, confirmed with bootstrapped confidence intervals and checked against a "wasted space" hypothesis to rule out an obvious confound. Across the 354 postcode sectors in the analysis, the effect ranges from about +18.7% down to -18.4% depending on where you are. That spread is the real finding. A single national average would have hidden it.

## Notes on how this was built

Every number in the pipeline comes from real government data, nothing synthetic. I found and fixed a reproducibility bug in the original Optuna search (the sampler wasn't seeded, so results shifted on every rerun). I also tested a 1–3km spatial grid as an alternative to postcode-based geography, it underperformed, and I kept that as a documented negative result rather than quietly dropping it. The deployed app approximates spatial lag price from training-data medians at serving time, which is a train/inference mismatch I've noted rather than hidden.

## Running it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tech stack

Python, pandas, scikit-learn, XGBoost, Optuna, SHAP, Streamlit, Folium
