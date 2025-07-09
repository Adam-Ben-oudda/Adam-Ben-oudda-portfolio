# 📈 Entreprise KPI Dashboard

**Version:** 1.0
**Author:** Adam Ben oudda 
**Project Type:** Enterprise-grade backend dashboard system for SaaS business KPI tracking
**Status:** MVP Complete

---

## 🔧 Overview

The **Entreprise KPI Dashboard** is a full-scale, AI-assisted performance monitoring engine built to handle critical metrics across large-scale SaaS systems.

It features:

* Real-time KPI ingestion and trend tracking
* Forecasting using Holt-Winters models
* Anomaly detection and alerting
* Personalized AI recommendations
* Multi-user session support with configurable dashboards
* Modular and extensible architecture (plug-and-play KPIs, models, alerts)

Built entirely in **Python**, this system requires **no frontend** and runs as a back-end service, returning structured JSON widgets that can power any UI (Streamlit, Dash, React, etc.).

---

## 🌐 Core Features

| Feature                       | Description                                                                        |
| ----------------------------- | ---------------------------------------------------------------------------------- |
| ✅ **KPI Registry**            | Fully pluggable KPI definitions with units, weight, directionality, and thresholds |
| ⚙️ **Forecast Engine**        | Holt-Winters smoothing models for short-term and long-term prediction              |
| 🚨 **Alert Manager**          | Rule-based alert system with severity levels, real-time notifications              |
| 🧠 **AI Recommender**         | Rule-based insights for trends, anomalies, correlations, and optimization          |
| 📅 **Session Manager**        | Multi-user profiles, refresh intervals, theme preferences, and stateful layouts    |
| 🔄 **Scheduled Data Refresh** | Asynchronous auto-ingestion of KPI data for continuous updates                     |
| 🔎 **Validator Engine**       | Cleans data, removes outliers, and autocorrects based on z-score logic             |

---

## 📊 Architecture Overview

```
+------------------------------+
|     EnterpriseKPIDashboard  |
+------------------------------+
       |    |     |     |      
       |    |     |     |      
       v    v     v     v      
    [Validator] [Forecaster] [AlertManager] [AIRecommender]     
         |           |            |                |
       [DataStore]   +------------+----------------+
             |
       [VisualizationEngine]
             |
           [Output JSON]
```

---

## 🌐 KPI Definitions

Each KPI is registered using the `KPI` dataclass:

```python
KPI(
    id="churn_rate",
    name="Churn Rate",
    unit="%",
    higher_is_better=False,
    min_value=0,
    max_value=100,
    weight=0.25,
    anomaly_threshold=2.5
)
```

---

## 🧰 AI Recommendation Engine

The engine provides contextual suggestions based on:

* **Trend analysis** (e.g. "Revenue increasing 21.4% this week")
* **Anomaly alerts** (e.g. sudden spike in churn)
* **Filter usage behavior** (e.g. recommend saving frequent filters)
* **Correlated KPI patterns**

All recommendations are structured and prioritized.

---

## 🚀 Getting Started

### 1. 📁 Clone the Repository

```bash
git clone https://github.com/yourusername/Entreprise_KPI_Dashboard.git
cd Entreprise_KPI_Dashboard
```

### 2. ♻️ Run the Application

```bash
python business_kpi_dashboard.py
```

System will auto-ingest sample data, analyze it, and run in the background.

---

## 📊 Sample Output

```json
{
  "type": "kpi",
  "data": {
    "title": "Churn Rate",
    "value": 4.87,
    "unit": "%",
    "trend": -2.1,
    "comparison": null,
    "color": "#00b894"
  }
}
```

---

## 📏 Future Enhancements

* Frontend interface (Streamlit or React)
* OAuth2 + JWT authentication
* Streaming data support
* WebSocket / API layer
* Time-series database integration
* AutoML selection for forecasts

---

## 👤 Author

**Adam Ben oudda**
Enterprise Data Scientist & AI Engineer
[GitHub](https://github.com/Adam-Ben-oudda) • [LinkedIn](Coming soon ...)

---

## 📄 License

This project is licensed under the **MIT License**.
