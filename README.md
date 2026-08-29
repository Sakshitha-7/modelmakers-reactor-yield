# Reactor Yield Predictor

Physics-informed ML web app for predicting reactor yield in a non-isothermal packed-bed reactor. The solution combines an Arrhenius-based PFR kinetics model with an ExtraTrees residual correction, producing fast predictions suitable for interactive use in a live demo.
# live demo https://modelmakerz.netlify.app/
## Problem

The reactor system is a series reaction network of the form A → B → C in a non-isothermal plug-flow reactor. The yield of the intermediate product B depends on the interaction between residence time, thermal profile, and operating conditions such as feed rate, concentration, reactor length, and jacket temperature.

A mechanistic model based on the PFR mass and energy balances can represent the underlying chemistry well, but it is computationally expensive to evaluate repeatedly during interactive optimization or live demonstration. A purely black-box surrogate can be fast, but it often struggles to capture the nonlinear process physics and operating envelope consistently.

## Approach

This project uses a delta-learning hybrid strategy:

1. The mechanistic core solves the non-isothermal PFR ODE system using Arrhenius kinetics.
2. The physics-based prediction is then passed into an ExtraTrees regressor that learns the residual between the true yield and the kinetic estimate.
3. The final prediction is:
   
   overall_yield ≈ physics_prediction + residual_correction

This structure keeps the model physically grounded while still capturing systematic deviations due to unmodeled effects and nonlinear interactions. In practice, the hybrid model gives a much more reliable fit than a pure data-driven model over the same operating range, while retaining the response speed required for real-time inference.

## Results

The official hackathon leaderboard result for this model was:

- Rank: 36 out of 384 teams
- RMSE: 12.9029
- MAE: 5.9045
- R²: 0.8719

These values reflect the final submission-quality model developed for the Fugacity Predictive Modeling Optimization Challenge at IIT Kharagpur, under the team name ModelMakers.

## Web Demo

The repository includes a lightweight Flask app that serves a single-page web interface. Users enter five operating conditions:

- flow rate (L/min)
- concentration (mol/L)
- inlet temperature (K)
- reactor length (m)
- jacket temperature (K)

The app returns the predicted overall yield in real time.

### Run locally

```bash
pip install -r requirements.txt
python app/app.py
```

Then open:

```text
http://127.0.0.1:5000
```

The frontend is served directly by Flask, so no separate build step is required.

## Tech Stack

- Python
- Flask
- scikit-learn
- SciPy
- pandas
- NumPy
- HTML/CSS/JavaScript

## Project Structure

```text
reactor-yield-predictor/
├── app/
│   ├── app.py
│   └── index.html
├── data/
│   └── train_dataset.csv
├── .gitignore
├── requirements.txt
├── README.md
└── .git/
```

## Notes

The dataset in `data/train_dataset.csv` is the training set used to fit both the kinetics parameters and the residual-learning model. The app loads this file on startup, fits the model once, and then serves predictions from memory for the live demo.

This project was built for the Fugacity ML hackathon, with the goal of combining reactor engineering knowledge with machine learning to create a practical and interpretable predictive system.
