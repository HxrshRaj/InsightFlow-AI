# InsightFlow AI

A production-ready Streamlit application for transforming raw CSV data into actionable insights through automated analytics and AI-powered reporting.

## Features

- **Dark Modern UI** — Professional interface with responsive layout
- **CSV Upload** — Drag-and-drop file ingestion with validation
- **Data Cleaning** — Automated handling of missing values and duplicates
- **Analytics Engine** — Statistical summaries and trend detection
- **AI Insights** — Natural-language explanations of your data
- **Report Generation** — Exportable analysis reports

## Project Structure

```
InsightFlow-AI/
├── app.py                  # Streamlit application entry point
├── requirements.txt        # Python dependencies
├── README.md
├── backend/
│   ├── __init__.py
│   ├── analytics.py        # Statistical analysis engine
│   ├── validation.py       # Data validation
│   ├── cleaning.py         # Data preprocessing
│   ├── ai.py               # AI-powered insights
│   ├── report.py           # Report generation
│   └── utils.py            # Shared utilities
├── sample_data/            # Sample CSV datasets
├── screenshots/            # Application screenshots
└── .streamlit/
    └── config.toml         # Streamlit theme configuration
```

## Getting Started

### Prerequisites

- Python 3.10 or higher
- pip

### Installation

```bash
git clone <repository-url>
cd InsightFlow-AI
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Development

This project follows a modular architecture. Backend logic lives in the `backend/` package and is imported by the Streamlit frontend in `app.py`. Each module has a single responsibility:

| Module | Purpose |
|---|---|
| `utils.py` | CSV loading, formatting helpers |
| `validation.py` | Dataset schema and quality checks |
| `cleaning.py` | Preprocessing and normalization |
| `analytics.py` | Statistical summaries and metrics |
| `ai.py` | LLM-powered insight generation |
| `report.py` | Markdown and export report building |

## License

MIT
