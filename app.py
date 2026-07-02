"""InsightFlow AI — Streamlit application entry point."""

from __future__ import annotations

import streamlit as st

from backend.utils import (
    DatasetSummary,
    get_dataframe_preview,
    get_dataset_summary,
    load_csv,
)

APP_TITLE = "InsightFlow AI"
APP_SUBTITLE = "Transform raw data into actionable insights with intelligent analytics"
APP_ICON = "📊"

NAV_ITEMS = {
    "Upload Dataset": "upload",
    "Home": "home",
    "Analytics": "analytics",
    "AI Insights": "ai_insights",
    "Reports": "reports",
    "Settings": "settings",
}

NAV_ICONS = {
    "Upload Dataset": "📁",
    "Home": "🏠",
    "Analytics": "📈",
    "AI Insights": "🤖",
    "Reports": "📄",
    "Settings": "⚙️",
}

DEFAULT_PAGE = "Upload Dataset"


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

            html, body, [class*="css"] {
                font-family: 'Inter', sans-serif;
            }

            .block-container {
                padding-top: 2rem;
                padding-bottom: 3rem;
                max-width: 1200px;
            }

            .hero-container {
                text-align: center;
                padding: 3rem 1rem 2rem;
            }

            .hero-title {
                font-size: 3rem;
                font-weight: 700;
                background: linear-gradient(135deg, #6366F1 0%, #A78BFA 50%, #38BDF8 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                margin-bottom: 0.5rem;
                line-height: 1.2;
            }

            .hero-subtitle {
                font-size: 1.15rem;
                color: #94A3B8;
                max-width: 600px;
                margin: 0 auto 2rem;
                line-height: 1.6;
            }

            .feature-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
                gap: 1.25rem;
                margin: 2rem 0;
            }

            .feature-card {
                background: linear-gradient(145deg, #1E293B 0%, #0F172A 100%);
                border: 1px solid #334155;
                border-radius: 12px;
                padding: 1.5rem;
                transition: border-color 0.2s ease, transform 0.2s ease;
            }

            .feature-card:hover {
                border-color: #6366F1;
                transform: translateY(-2px);
            }

            .feature-icon {
                font-size: 1.75rem;
                margin-bottom: 0.75rem;
            }

            .feature-title {
                font-size: 1rem;
                font-weight: 600;
                color: #F1F5F9;
                margin-bottom: 0.4rem;
            }

            .feature-desc {
                font-size: 0.875rem;
                color: #94A3B8;
                line-height: 1.5;
            }

            .upload-zone {
                background: #1E293B;
                border: 2px dashed #475569;
                border-radius: 16px;
                padding: 2.5rem 2rem;
                text-align: center;
                margin: 1.5rem 0;
                transition: border-color 0.2s ease;
            }

            .upload-zone:hover {
                border-color: #6366F1;
            }

            .upload-icon {
                font-size: 2.5rem;
                margin-bottom: 0.75rem;
            }

            .upload-title {
                font-size: 1.1rem;
                font-weight: 600;
                color: #F1F5F9;
                margin-bottom: 0.4rem;
            }

            .upload-hint {
                font-size: 0.85rem;
                color: #64748B;
            }

            .stat-row {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
                gap: 1rem;
                margin: 1.5rem 0;
            }

            .stat-card {
                background: #1E293B;
                border: 1px solid #334155;
                border-radius: 10px;
                padding: 1.25rem;
                text-align: center;
            }

            .stat-value {
                font-size: 1.75rem;
                font-weight: 700;
                color: #6366F1;
            }

            .stat-label {
                font-size: 0.8rem;
                color: #94A3B8;
                margin-top: 0.25rem;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }

            .section-header {
                font-size: 1.4rem;
                font-weight: 600;
                color: #F1F5F9;
                margin: 2rem 0 1rem;
                padding-bottom: 0.5rem;
                border-bottom: 1px solid #334155;
            }

            .sidebar-brand {
                font-size: 1.1rem;
                font-weight: 700;
                color: #6366F1;
                padding: 0.5rem 0 1rem;
                border-bottom: 1px solid #334155;
                margin-bottom: 1rem;
            }

            div[data-testid="stSidebar"] {
                background-color: #0F172A;
                border-right: 1px solid #1E293B;
            }

            div[data-testid="stSidebar"] .stRadio label {
                font-size: 0.9rem;
            }

            .footer {
                text-align: center;
                color: #475569;
                font-size: 0.8rem;
                padding: 2rem 0 0;
                border-top: 1px solid #1E293B;
                margin-top: 3rem;
            }

            .empty-state {
                text-align: center;
                padding: 4rem 2rem;
                margin: 2rem 0;
                background: linear-gradient(145deg, #1E293B 0%, #0F172A 100%);
                border: 1px dashed #475569;
                border-radius: 16px;
            }

            .empty-state-icon {
                font-size: 2.5rem;
                margin-bottom: 1rem;
                opacity: 0.6;
            }

            .empty-state-title {
                font-size: 1.25rem;
                font-weight: 600;
                color: #CBD5E1;
                margin-bottom: 0.5rem;
            }

            .empty-state-desc {
                font-size: 0.9rem;
                color: #64748B;
            }

            div[data-testid="stMetric"] {
                background: #1E293B;
                border: 1px solid #334155;
                border-radius: 10px;
                padding: 0.75rem 1rem;
            }

            div[data-testid="stMetric"] label {
                color: #94A3B8 !important;
                font-size: 0.75rem !important;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }

            div[data-testid="stMetric"] [data-testid="stMetricValue"] {
                color: #6366F1 !important;
                font-weight: 700;
            }

            .overview-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 0.75rem;
                margin: 1rem 0 1.5rem;
            }

            .overview-item {
                background: #1E293B;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 0.875rem 1rem;
            }

            .overview-label {
                font-size: 0.75rem;
                color: #64748B;
                text-transform: uppercase;
                letter-spacing: 0.04em;
                margin-bottom: 0.25rem;
            }

            .overview-value {
                font-size: 1.1rem;
                font-weight: 600;
                color: #F1F5F9;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def configure_page() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )


def init_session_state() -> None:
    if "dataframe" not in st.session_state:
        st.session_state.dataframe = None
    if "uploaded_filename" not in st.session_state:
        st.session_state.uploaded_filename = None


def render_sidebar() -> str:
    st.sidebar.markdown(
        f'<div class="sidebar-brand">{APP_ICON} {APP_TITLE}</div>',
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("##### Navigation")
    nav_options = list(NAV_ITEMS.keys())
    default_index = nav_options.index(DEFAULT_PAGE)
    selected = st.sidebar.radio(
        label="Navigation",
        options=nav_options,
        index=default_index,
        format_func=lambda item: f"{NAV_ICONS[item]}  {item}",
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("##### About")
    st.sidebar.caption(
        "InsightFlow AI helps teams explore, clean, and understand "
        "their data through automated analytics and AI-powered insights."
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("v0.1.0 · Foundation Release")

    return NAV_ITEMS[selected]


def render_hero() -> None:
    st.markdown(
        f"""
        <div class="hero-container">
            <div class="hero-title">{APP_TITLE}</div>
            <div class="hero-subtitle">{APP_SUBTITLE}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_feature_cards() -> None:
    features = [
        ("📁", "Smart Upload", "Drag and drop CSV files with instant validation and preview."),
        ("🧹", "Auto Cleaning", "Detect and resolve missing values, duplicates, and outliers."),
        ("📈", "Deep Analytics", "Statistical summaries, correlations, and trend detection."),
        ("🤖", "AI Insights", "Natural-language explanations powered by large language models."),
    ]
    cards_html = '<div class="feature-grid">'
    for icon, title, desc in features:
        cards_html += f"""
            <div class="feature-card">
                <div class="feature-icon">{icon}</div>
                <div class="feature-title">{title}</div>
                <div class="feature-desc">{desc}</div>
            </div>
        """
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)


def render_home_page() -> None:
    render_hero()
    render_feature_cards()


def render_empty_state() -> None:
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-state-icon">📂</div>
            <div class="empty-state-title">Upload a CSV dataset to begin analysis.</div>
            <div class="empty-state-desc">Drag and drop a file above or click to browse.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_cards(summary: DatasetSummary) -> None:
    metrics = [
        ("Rows", f"{summary.row_count:,}"),
        ("Columns", f"{summary.column_count:,}"),
        ("Memory", f"{summary.memory_mb:.2f} MB"),
        ("Missing Values", f"{summary.missing_values:,}"),
        ("Duplicate Rows", f"{summary.duplicate_rows:,}"),
        ("Numeric Features", f"{summary.numeric_count:,}"),
    ]
    cols = st.columns(6)
    for col, (label, value) in zip(cols, metrics):
        with col:
            st.metric(label=label, value=value)


def render_dataset_overview(summary: DatasetSummary) -> None:
    st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="overview-grid">
            <div class="overview-item">
                <div class="overview-label">Rows</div>
                <div class="overview-value">{summary.row_count:,}</div>
            </div>
            <div class="overview-item">
                <div class="overview-label">Columns</div>
                <div class="overview-value">{summary.column_count:,}</div>
            </div>
            <div class="overview-item">
                <div class="overview-label">Memory Usage</div>
                <div class="overview-value">{summary.memory_mb:.2f} MB</div>
            </div>
            <div class="overview-item">
                <div class="overview-label">Numeric Columns</div>
                <div class="overview-value">{summary.numeric_count:,}</div>
            </div>
            <div class="overview-item">
                <div class="overview-label">Categorical Columns</div>
                <div class="overview-value">{summary.categorical_count:,}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_column_details(summary: DatasetSummary) -> None:
    st.markdown('<div class="section-header">Column Details</div>', unsafe_allow_html=True)
    st.dataframe(
        summary.column_info,
        use_container_width=True,
        hide_index=True,
        height=min(44 * len(summary.column_info) + 38, 400),
    )


def render_data_preview(df) -> None:
    st.markdown('<div class="section-header">Data Preview</div>', unsafe_allow_html=True)
    preview = get_dataframe_preview(df, rows=10)
    st.dataframe(
        preview,
        use_container_width=True,
        hide_index=True,
        height=400,
    )


def handle_file_upload() -> None:
    uploaded_file = st.file_uploader(
        label="Upload CSV",
        type=["csv"],
        accept_multiple_files=False,
        help="Upload a comma-separated values file to begin analysis.",
    )

    if uploaded_file is None:
        return

    try:
        st.session_state.dataframe = load_csv(uploaded_file)
        st.session_state.uploaded_filename = uploaded_file.name
    except ValueError as exc:
        st.error(str(exc))


def render_upload_page() -> None:
    st.markdown("## Upload Dataset")
    st.caption("Import a CSV file to explore your data structure and quality metrics.")

    st.markdown(
        """
        <div class="upload-zone">
            <div class="upload-icon">📂</div>
            <div class="upload-title">Drop your CSV file here</div>
            <div class="upload-hint">Supports .csv files only</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    handle_file_upload()

    df = st.session_state.dataframe
    if df is None:
        render_empty_state()
        return

    filename = st.session_state.uploaded_filename or "dataset.csv"
    st.success(f"Loaded **{filename}** successfully.")

    summary = get_dataset_summary(df)

    render_kpi_cards(summary)
    render_dataset_overview(summary)
    render_column_details(summary)
    render_data_preview(df)


def render_placeholder_page(title: str, description: str) -> None:
    st.markdown(f"## {title}")
    st.markdown(description)
    st.info("This section is under development and will be available soon.")


def render_page(page: str) -> None:
    pages = {
        "home": lambda: render_home_page(),
        "upload": lambda: render_upload_page(),
        "analytics": lambda: render_placeholder_page(
            "Analytics Dashboard",
            "Explore statistical summaries, distributions, and correlations across your data.",
        ),
        "ai_insights": lambda: render_placeholder_page(
            "AI Insights",
            "Receive natural-language summaries and actionable recommendations from your data.",
        ),
        "reports": lambda: render_placeholder_page(
            "Reports",
            "Generate and export professional analysis reports in multiple formats.",
        ),
        "settings": lambda: render_placeholder_page(
            "Settings",
            "Configure API keys, display preferences, and export options.",
        ),
    }
    pages[page]()


def render_footer() -> None:
    st.markdown(
        f'<div class="footer">© 2026 {APP_TITLE} · Built with Streamlit</div>',
        unsafe_allow_html=True,
    )


def main() -> None:
    configure_page()
    init_session_state()
    inject_custom_css()

    current_page = render_sidebar()
    render_page(current_page)
    render_footer()


if __name__ == "__main__":
    main()
