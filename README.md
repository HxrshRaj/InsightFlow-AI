# 🚀 InsightFlow AI

<div align="center">

### Enterprise Data Engineering & AI Analytics Platform

**Transform raw datasets into actionable business insights using AI-powered analytics.**

🌐 **Live Demo:** https://insightflow-ai-stazd6bjfvnu5uyjlvqbmv.streamlit.app/

💻 **GitHub Repository:** https://github.com/HxrshRaj/InsightFlow-AI

</div>

---

## 📖 Overview

InsightFlow AI is a modular data analytics platform designed to automate the complete workflow from raw CSV datasets to business-ready insights.

The application enables users to upload datasets, assess data quality, automatically clean the data, explore interactive dashboards, generate AI-powered business insights using Groq LLM, and export reports—all through a simple Streamlit interface.

The project follows a modular architecture, making it easy to extend with additional enterprise-grade capabilities in future versions.

---

# ✨ Features

## 📁 Dataset Upload

* Upload CSV datasets
* Instant dataset preview
* Automatic dataset profiling
* Memory usage analysis
* Data type detection

---

## ✅ Data Validation

* Missing value detection
* Duplicate record detection
* Constant column identification
* Column profiling
* Data Quality Score
* Validation recommendations

---

## 🧹 Automatic Data Cleaning

* Remove duplicate rows
* Fill missing numeric values
* Fill missing categorical values
* Optimize data types
* Before vs After comparison
* Download cleaned dataset

---

## 📊 Interactive Analytics Dashboard

* Executive KPI cards
* Summary statistics
* Correlation analysis
* Histograms
* Scatter plots
* Box plots
* Category distribution charts
* Outlier detection

---

## 🤖 AI Business Analyst

Powered by the Groq API.

Automatically generates:

* Executive Summary
* Business Insights
* Potential Risks
* Actionable Recommendations

---

## 📄 Report Export

* PDF Report
* Markdown Report
* Cleaned CSV Download

---

# 🛠️ Technology Stack

| Category          | Technologies     |
| ----------------- | ---------------- |
| Language          | Python           |
| Frontend          | Streamlit        |
| Data Processing   | Pandas, NumPy    |
| Visualization     | Plotly           |
| Machine Learning  | Scikit-learn     |
| AI                | Groq API (Llama) |
| Report Generation | FPDF2            |
| Configuration     | python-dotenv    |

---

# 🏗️ Project Architecture

```text
                CSV Upload
                     │
                     ▼
          Dataset Validation Engine
                     │
                     ▼
        Automatic Data Cleaning
                     │
                     ▼
        Analytics & Visualization
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
 Interactive Dashboard     AI Business Analyst
          │                     │
          └──────────┬──────────┘
                     ▼
              Report Generation
```

---

# 📂 Project Structure

```text
InsightFlow-AI/
│
├── app.py
├── requirements.txt
├── README.md
│
├── backend/
│   ├── analytics.py
│   ├── validation.py
│   ├── cleaning.py
│   ├── ai.py
│   ├── report.py
│   └── utils.py
│
├── sample_data/
├── screenshots/
└── .gitignore
```

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/HxrshRaj/InsightFlow-AI.git
cd InsightFlow-AI
```

Create a virtual environment

```bash
python -m venv venv
```

Activate

Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Create a `.env` file

```env
GROQ_API_KEY=YOUR_GROQ_API_KEY
```

Run the application

```bash
streamlit run app.py
```

---

# 🌐 Live Application

👉 https://insightflow-ai-stazd6bjfvnu5uyjlvqbmv.streamlit.app/

---

# 📸 Screenshots

Add the following screenshots after deployment:

* Home Page
* Dataset Upload
* Data Validation
* Automatic Cleaning
* Analytics Dashboard
* AI Business Insights
* Report Export

---

# 🎯 Future Enhancements

* User Authentication
* SQLite-based Dataset History
* Multi-file Data Processing
* REST API Integration
* Docker Support
* Cloud Storage Integration
* Scheduled ETL Pipelines
* Apache Spark Integration
* Apache Airflow Workflow Automation

---

# 💼 Resume Highlights

* Built an AI-powered data analytics platform using Python, Streamlit, Pandas, Plotly, and Scikit-learn to automate dataset validation, preprocessing, visualization, and reporting.

* Developed modular data engineering components for data quality assessment, automated cleaning, interactive analytics, and report generation.

* Integrated Groq LLM to generate executive summaries, business insights, risk analysis, and actionable recommendations from structured datasets.

* Designed a scalable architecture that supports future enterprise integrations while maintaining a clean, modular, and deployable MVP.

---

# 🤝 Contributing

Contributions, feature suggestions, and improvements are welcome.

Feel free to fork the repository and submit a pull request.

---

# 📄 License

This project is licensed under the MIT License.

---

<div align="center">

### ⭐ If you found this project useful, consider giving it a star!

Built with ❤️ using Python, Streamlit, Plotly, Pandas, and Groq AI.

</div>
