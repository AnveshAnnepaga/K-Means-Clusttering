# 🧠 Wholesale Customer Segmentation

## Business Problem
All customers were treated the same, leading to:
- Inefficient inventory planning
- Poor marketing strategies
- Missed upselling opportunities

## Solution
Customers are grouped using **KMeans clustering** based on purchasing behavior.

---

## Features Used
- Fresh
- Milk
- Grocery
- Frozen
- Detergents_Paper
- Delicassen

These features directly represent customer spending habits.

---

## Steps Performed
1. Data loading and inspection
2. Feature selection (spending-related only)
3. Feature scaling (StandardScaler)
4. KMeans clustering
5. Cluster profiling
6. Visualization
7. Business insight generation

---

## Optimal K Selection
Elbow method or business interpretability suggests **K = 3 to 5** is reasonable.

---

## Business Strategies
- **High bulk buyers** → Volume discounts & priority inventory
- **Retail-focused buyers** → Grocery & detergent promotions
- **Low spenders** → Upselling bundles

---

## Limitations
- KMeans assumes spherical clusters
- Sensitive to outliers
- Requires predefined K

---

## Run the App
```bash
pip install -r requirements.txt
streamlit run app.py
