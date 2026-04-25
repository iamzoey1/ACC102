# Influencer Marketing Market Power Analyzer

## 1. Problem & User
This project quantifies influencer market power through user stickiness, demand elasticity, and consumer welfare to help brand managers and MCN analysts identify optimal KOL investment targets beyond vanity metrics.

## 2. Data
- **Source:** Kaggle (`influencer_marketing_roi_dataset.csv`)
- **Access Date:** April 2026
- **Size:** 150,000 campaigns (2023–2025)
- **Key Fields:** `estimated_reach` (audience scale), `engagements` (interaction quality), `product_sales` (monetization)
- **Note:** The dataset is loaded via relative path in both the Jupyter Notebook and the Streamlit app.

## 3. Methods
- **Data preprocessing:** Pandas loading, field mapping to economic concepts (`fans`, `interactions`, `revenue`), deterministic noise injection (`np.random.seed(42)`), and invalid-record filtering.
- **Feature engineering:** NumPy vectorized computation of attention conversion rate, ARPU, user stickiness (logistic function), and a weighted market-power composite.
- **Analysis & visualization:** Static EDA with Matplotlib/Seaborn in Jupyter Notebook; interactive decision-support tool built with Streamlit and Plotly.

## 4. Key Findings
- ~35% of campaigns exhibit "Winner-Takes-All" market power, indicating an oligopolistic core.
- Follower count does not guarantee pricing power; engagement quality dominates reach quantity.
- Competitive markets show demand elasticity (~-0.8) roughly 3× higher than winner-takes-all segments (~-0.25), requiring segmented pricing strategies.
- A strong negative correlation (-0.85) between market power and consumer welfare confirms the power-welfare trade-off.
- User stickiness exhibits a threshold effect around 0.10 attention conversion rate.

## 5. How to run
Prerequisites: Python 3.9+; dependencies listed in requirements.txt.

Setup:
1. Clone this repository:
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
2. Install dependencies:

    pip install -r requirements.txt

Run the Jupyter Notebook:
Open notebook.ipynb and execute cells sequentially from top to bottom to reproduce the full static analysis.

Run the Streamlit App (Track 4 Product):

    streamlit run app.py  

The app will start locally at http://localhost:8501.

## 6. Product link / Demo
- Product Link (GitHub Repository): https://github.com/<your-username>/<your-repo-name>
- Demo Video: [Link to be updated upon upload — accessible to marker during marking period]

## 7. Limitations & next steps
- Cross-sectional data limits causal inference; panel data and fixed-effects models would enable dynamic analysis.
- Market-structure thresholds (0.40, 0.70) are normative; future work should apply endogenous breakpoint tests and platform-specific calibration.
- Fixed model weights (35%-35%-30%) assume homogeneity; AHP or Bayesian hierarchical models could improve heterogeneity handling.
- Bidirectional causality between fans and interactions may bias OLS estimates; instrumental variables or SEM are recommended for future research.
