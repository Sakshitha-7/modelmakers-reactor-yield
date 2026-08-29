"""
Reactor Yield Predictor - Backend
==================================
Serves predictions from the ModelMakers delta-learning hybrid model
(Arrhenius PFR physics core + ExtraTrees residual correction).

On startup: loads train_dataset.csv, fits the physics kinetics parameters
and trains the residual model once, then keeps them in memory.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from sklearn.ensemble import ExtraTreesRegressor

R = 8.314

FEATURES = [
    "flow_rate_L_min",
    "concentration_mol_L",
    "inlet_temperature_K",
    "length_m",
    "jacket_temperature_K",
]


def pfr_yield(row, p):
    lnA1, Ea1, lnA2, Ea2, UA = p
    Tin, Tjk = row.inlet_temperature_K, row.jacket_temperature_K
    tau = row.length_m / row.flow_rate_L_min

    def odes(t, s):
        cA, cB, T = s
        k1 = np.exp(lnA1) * np.exp(-Ea1 / (R * T))
        k2 = np.exp(lnA2) * np.exp(-Ea2 / (R * T))
        dT = UA * (Tjk - T)
        return [-k1 * cA, k1 * cA - k2 * cB, dT]

    sol = solve_ivp(odes, [0, tau], [1.0, 0.0, Tin],
                     rtol=1e-6, atol=1e-9, method="LSODA")
    return float(np.clip(sol.y[1, -1] * 100, 0, 100))


def physics_predict(df, p):
    return np.array([pfr_yield(r, p) for _, r in df.iterrows()])


def fit_kinetics(df, y):
    p0 = [11.87, 33900, 39.94, 141742, 3.06]
    bounds = ([0, 1e3, 0, 1e3, 0.0], [40, 2e5, 40, 2e5, 20.0])
    resid = lambda p: physics_predict(df, p) - y
    return least_squares(resid, p0, bounds=bounds, max_nfev=120).x


def engineer(df):
    d = df.copy()
    Tin, Tjk = d.inlet_temperature_K, d.jacket_temperature_K
    d["tau_proxy"] = d.length_m / d.flow_rate_L_min
    d["T_mean"] = (Tin + Tjk) / 2
    d["dT"] = Tjk - Tin
    d["sel"] = np.exp(-8000 / Tjk) / (np.exp(-12000 / Tjk) + 1e-9)
    d["tau_T"] = d.tau_proxy * d.T_mean
    d["log_flow"] = np.log(d.flow_rate_L_min)
    d["T_sq"] = d.T_mean ** 2
    d["overtemp"] = d.T_mean * d.tau_proxy
    d["Da_c"] = d.tau_proxy * np.exp(-8000 / Tjk) * d.concentration_mol_L
    return d


def new_et():
    return ExtraTreesRegressor(n_estimators=1000, min_samples_leaf=1,
                                max_features=0.7, random_state=42, n_jobs=-1)


print("Loading training data and fitting model (delta-learning hybrid)...")
tr = pd.read_csv("../data/train_dataset.csv")
y = tr["overall_yield"].values

KINETICS_PARAMS = fit_kinetics(tr, y)
ph_tr = physics_predict(tr, KINETICS_PARAMS)

X_COLS = list(engineer(tr).drop(columns=["overall_yield"]).columns)
Xtr = engineer(tr)[X_COLS].copy()
Xtr["phys"] = ph_tr

RESIDUAL_MODEL = new_et()
RESIDUAL_MODEL.fit(Xtr, y - ph_tr)

print(f"Model ready. Fitted kinetics [lnA1, Ea1, lnA2, Ea2, UA]: {np.round(KINETICS_PARAMS, 2)}")


def predict_one(inputs: dict) -> float:
    df = pd.DataFrame([{
        "flow_rate_L_min": inputs["flow_rate_L_min"],
        "concentration_mol_L": inputs["concentration_mol_L"],
        "inlet_temperature_K": inputs["inlet_temperature_K"],
        "length_m": inputs["length_m"],
        "jacket_temperature_K": inputs["jacket_temperature_K"],
    }])
    ph = physics_predict(df, KINETICS_PARAMS)
    Xq = engineer(df)[X_COLS].copy()
    Xq["phys"] = ph
    pred = np.clip(ph + RESIDUAL_MODEL.predict(Xq), 0, 100)
    return float(pred[0])


app = Flask(__name__, static_folder="..", static_url_path="")
CORS(app)


@app.route("/")
def home():
    return app.send_static_file("app/index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}

    missing = [f for f in FEATURES if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    parsed = {}
    for f in FEATURES:
        try:
            parsed[f] = float(data[f])
        except (TypeError, ValueError):
            return jsonify({"error": f"'{f}' must be a number"}), 400

    if parsed["flow_rate_L_min"] <= 0:
        return jsonify({"error": "flow_rate_L_min must be greater than 0"}), 400

    try:
        yield_pct = predict_one(parsed)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    return jsonify({"overall_yield": round(yield_pct, 3)})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
