---
title: Aadhaar Insights Dashboard
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.35.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

# Aadhaar Insights Dashboard

An interactive dashboard for analyzing Aadhaar enrollment and update statistics across India.

## 🚀 Features

- Interactive map visualization of Aadhaar data
- State-wise and district-wise analysis
- Time-series trends and insights
- Demographic breakdowns
- **Advanced Demographics**: Age Group (0-5, 5-17, 18+) and Gender analysis
- **Hyper-local Insights**: Pincode-level drill-down for top performing areas
- **Update Type Analysis**: Biometric vs Demographic comparisons
- **Predictive Analytics**: 3-month forecasting using simple moving averages
- **Impact Metrics**: Digital Inclusion Index (DII) based on service reach

## 🛠️ Running Locally

1. Clone the repository
   ```bash
   git clone https://github.com/shrusti205/aadhaar-insights.git
   cd aadhaar-insights
   ```

2. Create and activate a virtual environment
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Linux/Mac
   # source venv/bin/activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application
   ```bash
   streamlit run app.py
   ```

## 🏆 Evaluation Parameters

All applications will be rated on the parameters listed below:

- **Data Analysis & Insights**:
  - Depth, accuracy, and relevance of univariate/bivariate/trivariate analysis
  - Ability to extract meaningful findings from the data

- **Creativity & Originality**:
  - Uniqueness of the problem statement or solution, innovative use of datasets

- **Technical Implementation**:
  - Code quality, reproducibility, and rigour of approach, appropriate methods, tooling, and documentation

- **Visualisation & Presentation**:
  - Clarity and effectiveness of data visualisations, Quality of written report or slides

- **Impact & Applicability**:
  - Potential for social/administrative benefit, Practicality and feasibility of insights/solutions