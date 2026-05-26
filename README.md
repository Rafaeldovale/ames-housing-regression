# Ames Housing Price Prediction Engine 🏠 Graph-Backed Regression Pipeline

A comprehensive end-to-end Machine Learning engineered pipeline designed to predict residential property prices in Ames, Iowa. This project transitions systematically from mathematical baseline modeling into high-performance tree ensembles, culminating in a serialized, production-ready scoring artifact.

## 🚀 Architectural Blueprint & Workflow

The repository is structured following strict production-ready data science standards:

```text
property-price-prediction/
├── data/
│   └── raw/
│       └── train.csv
├── notebooks/
│   └── property_price_prediction.ipynb
├── champion_house_predictor.pkl
├── venv/
├── README.md
└── requirements.txt


📊 The Model Benchmarking Ecosystem (Tournament)

To satisfy rigorous software engineering and financial risk standards, the dataset was exposed to an automated benchmarking tournament crossing 6 distinct algorithmic families (parametric, non-parametric, distance-based, and boosting ensembles) across the entire target token matrix.

The Final Leaderboard Placar

The models were evaluated globally on the test split, sorted by their variance explanation power ($R^2$ Score):

| Rank | Model Hierarchy | MAE ($) | RMSE ($) | $R^2$ Score | Status / Architectural Takeaway |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **🥇 1** | Gradient Boosting | 16,227.67 | 25,755.65 | 0.9135 | Production Champion. Sequential boosting minimizes residual chain errors. |
| **🥈 2** | Random Forest | 20,904.75 | 33,168.42 | 0.8566 | Stable. Bootstrap aggregation dampens single-tree variance splits. |
| **🥉 3** | Decision Tree | 27,511.28 | 39,390.78 | 0.7977 | Baseline Tree. Susceptible to step-function limits and overfitting. |
| **4** | K-Nearest Neighbors (KNN) | 29,705.85 | 48,941.00 | 0.6877 | Distance Failure. Constrained by localized geometric dimensions. |
| **5** | Support Vector Regressor (SVR) | 59,556.71 | 88,653.05 | -0.0246 | Scalability Collapse. Strict requirement for advanced feature scaling. |
| **6** | Linear Regression | 26,477.83 | 98,644.59 | -0.2686 | Baseline Floor. High RMSE reveals extreme vulnerability to outlier variance. |

🔬 Core Engineering Insights & Validation

* **The Power of Sequential Learning:** Gradient Boosting outperformed classical models by driving the global Mean Absolute Error (MAE) down to $16,227.67, explaining 91.35% of the dataset variance.
* **Catastrophic Outlier Penalization:** While Linear Regression presented a moderate MAE, its massive RMSE ($98,644.59) acted as a structural red flag, revealing catastrophic mispredictions on luxury edge cases due to its rigid parametric assumptions.
* **Spatial Convergence:** Convergence was validated visually via Actual vs. Predicted Spatial Matrices. The Gradient Boosting prediction tokens tightly locked onto the $45^\circ$ identity line across all valuation spectrums.

📦 Production Serialization & Local Deployment

The winning Gradient Boosting model was compiled alongside its explicit metadata schema into a decoupled binary deployment artifact: `champion_house_predictor.pkl`.

⚡ Quick Start: Spin Up and Execute the Environment

To replicate the experimental pipeline, configure the virtual environment, and run live inferences, execute the following commands in your terminal:

**1. Clone the repository and navigate into the workspace:**
```bash
cd property-price-prediction


**2. Initialize and activate the isolated virtual environment (venv):

# Windows (Git Bash / Command Prompt)
python -m venv venv
source venv/Scripts/activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate

**3. Install the explicit production dependencies:

pip install -r requirements.txt


**4. Execute Production Inference Server Simulation:

Inside your notebook or python execution environment, load the serialized .pkl artifact to score real-time payloads with zero retraining overhead:

import pickle

# Load the binary architecture back into server memory
with open('champion_house_predictor.pkl', 'rb') as file:
    loaded_artifact = pickle.load(file)

deployed_model = loaded_artifact['model']
deployed_features = loaded_artifact['features']

# Run instant inference on fresh real estate arrays
# predictions = deployed_model.predict(new_house_payload[deployed_features])

Developed by Rafael Bezerra do Vale as a milestone portfolio component in Advanced Applied Machine Learning and Architectural AI Engineering.