"""InsightFlow AI — Streamlit application entry point."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from backend.ai import create_dataset_context, generate_ai_insights
from backend.analytics import (
    AnalyticsEngine,
    categorical_columns,
    create_boxplot,
    create_correlation_heatmap,
    create_histogram,
    create_scatter,
    detect_outliers,
    numeric_columns,
    summary_statistics,
    top_categories,
)
from backend.cleaning import clean_dataset
from backend.report import ReportGenerator, generate_markdown_report, generate_pdf_report, generate_summary_table
from backend.utils import (
    DatasetSummary,
    calculate_quality_score,
    get_dataframe_metrics,
    get_dataframe_preview,
    get_dataset_summary,
    load_csv,
)
from backend.validation import DataValidator

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
    """Initialize the Streamlit session state used across the app."""
    if "dataframe" not in st.session_state:
        st.session_state.dataframe = None
    if "df" not in st.session_state:
        st.session_state.df = None
    if "uploaded_filename" not in st.session_state:
        st.session_state.uploaded_filename = None
    if "cleaned_df" not in st.session_state:
        st.session_state.cleaned_df = None
    if "cleaning_summary" not in st.session_state:
        st.session_state.cleaning_summary = None
    if "cleaning_log" not in st.session_state:
        st.session_state.cleaning_log = []
    if "ai_report" not in st.session_state:
        st.session_state.ai_report = ""


def render_sidebar() -> str:
    """Render the sidebar navigation for the application."""
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
    """Render the landing hero section on the home page."""
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
    """Render the feature cards shown on the home page."""
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


def get_current_dataframe() -> pd.DataFrame | None:
    """Return the active dataframe from the current session state."""
    df = st.session_state.get("df")
    if df is None:
        df = st.session_state.get("dataframe")
        if df is not None:
            st.session_state.df = df
    return df


def render_data_validation(df) -> None:
    st.markdown('<div class="section-header">Data Validation</div>', unsafe_allow_html=True)
    validator = DataValidator()
    result = validator.validate(df)

    if result.is_valid:
        st.success("The dataset passed the basic validation checks.")
    else:
        st.warning("The dataset has validation issues that may affect analysis.")

    if result.errors:
        with st.expander("Validation errors", expanded=True):
            for error in result.errors:
                st.write(f"• {error}")

    if result.warnings:
        with st.expander("Validation warnings", expanded=False):
            for warning in result.warnings:
                st.write(f"• {warning}")


def render_cleaning_summary_cards(summary: dict[str, object]) -> None:
    """Render KPI cards for the cleaning summary."""
    metrics = [
        ("Rows Before", f"{summary['rows_before']:,}"),
        ("Rows After", f"{summary['rows_after']:,}"),
        ("Duplicates Removed", f"{summary['duplicates_removed']:,}"),
        ("Missing Values Fixed", f"{summary['missing_values_fixed']:,}"),
        ("Memory Saved", f"{summary['memory_saved_mb']:.2f} MB"),
        ("Columns Optimized", f"{summary['columns_optimized']:,}"),
    ]
    cols = st.columns(6)
    for col, (label, value) in zip(cols, metrics):
        with col:
            st.metric(label=label, value=value)


def render_cleaning_section(df) -> None:
    st.markdown('<div class="section-header">Automatic Data Cleaning</div>', unsafe_allow_html=True)

    if df is None:
        st.info("Upload a CSV dataset to begin cleaning.")
        return

    if st.button("Run Automatic Cleaning", use_container_width=True):
        try:
            cleaned_df, cleaning_summary, cleaning_log = clean_dataset(df)
            st.session_state.cleaned_df = cleaned_df
            st.session_state.cleaning_summary = cleaning_summary
            st.session_state.cleaning_log = cleaning_log
            st.success("Automatic cleaning completed successfully.")
        except Exception as exc:  # pragma: no cover - defensive UI handling
            st.error(f"Cleaning could not be completed: {exc}")
            return

    cleaned_df = st.session_state.get("cleaned_df")
    cleaning_summary = st.session_state.get("cleaning_summary")
    cleaning_log = st.session_state.get("cleaning_log", [])

    if cleaned_df is None or cleaning_summary is None:
        st.info("No cleaned dataset generated yet. Run the cleaning pipeline to prepare a cleaned copy.")
        return

    render_cleaning_summary_cards(cleaning_summary)

    before_summary = get_dataset_summary(df)
    after_summary = get_dataset_summary(cleaned_df)

    comparison_df = {
        "Metric": ["Rows", "Columns", "Memory", "Missing Values", "Duplicate Rows", "Quality Score"],
        "Before": [
            f"{before_summary.row_count:,}",
            f"{before_summary.column_count:,}",
            f"{before_summary.memory_mb:.2f} MB",
            f"{before_summary.missing_values:,}",
            f"{before_summary.duplicate_rows:,}",
            f"{get_dataframe_metrics(df)['quality_score']:.1f}",
        ],
        "After": [
            f"{after_summary.row_count:,}",
            f"{after_summary.column_count:,}",
            f"{after_summary.memory_mb:.2f} MB",
            f"{after_summary.missing_values:,}",
            f"{after_summary.duplicate_rows:,}",
            f"{get_dataframe_metrics(cleaned_df)['quality_score']:.1f}",
        ],
    }

    st.markdown("##### Before vs After")
    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True,
        height=320,
    )

    st.markdown("##### Cleaning Log")
    for entry in cleaning_log:
        st.write(f"{entry}")

    st.download_button(
        label="Download Cleaned CSV",
        data=cleaned_df.to_csv(index=False).encode("utf-8"),
        file_name="cleaned_dataset.csv",
        mime="text/csv",
        use_container_width=True,
    )


def render_data_preview(df: pd.DataFrame) -> None:
    """Render a preview table for the current dataframe."""
    st.markdown('<div class="section-header">Data Preview</div>', unsafe_allow_html=True)
    preview = get_dataframe_preview(df, rows=10)
    st.dataframe(
        preview,
        use_container_width=True,
        hide_index=True,
        height=400,
    )


def handle_file_upload() -> None:
    """Handle CSV uploads and update the session state."""
    uploaded_file = st.file_uploader(
        label="Upload CSV",
        type=["csv"],
        accept_multiple_files=False,
        help="Upload a comma-separated values file to begin analysis.",
    )

    if uploaded_file is None:
        return

    try:
        loaded_df = load_csv(uploaded_file)
        st.session_state.dataframe = loaded_df
        st.session_state.df = loaded_df
        st.session_state.uploaded_filename = uploaded_file.name
        st.session_state.cleaned_df = None
        st.session_state.cleaning_summary = None
        st.session_state.cleaning_log = []
    except ValueError as exc:
        st.error(str(exc))


def render_upload_page() -> None:
    """Render the upload and validation workflow page."""
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

    df = get_current_dataframe()
    if df is None:
        render_empty_state()
        return

    filename = st.session_state.uploaded_filename or "dataset.csv"
    st.success(f"Loaded **{filename}** successfully.")

    summary = get_dataset_summary(df)
    st.session_state.last_summary = get_dataframe_metrics(df)

    render_kpi_cards(summary)
    render_dataset_overview(summary)
    render_column_details(summary)
    render_data_validation(df)
    render_cleaning_section(df)
    render_data_preview(df)


def render_placeholder_page(title: str, description: str) -> None:
    """Render a simple placeholder page for in-development sections."""
    st.markdown(f"## {title}")
    st.markdown(description)
    st.info("This section is under development and will be available soon.")


def render_reports_page() -> None:
    """Render the reports page and provide export downloads."""
    st.markdown("## Reports")
    st.caption("Download a professional PDF or Markdown report for the current analysis.")

    df = st.session_state.get("cleaned_df")
    if df is None:
        df = st.session_state.get("dataframe")

    if df is None:
        st.info("Upload and analyze a dataset before generating reports.")
        return

    report_generator = ReportGenerator()
    metadata = report_generator.build_metadata(
        title="InsightFlow AI Enterprise Data Analytics Report",
        row_count=len(df),
        column_count=len(df.columns),
    )

    cleaning_summary = st.session_state.get("cleaning_summary") or {}
    ai_report = st.session_state.get("ai_report", "") or "AI business insights are unavailable."

    metrics = get_dataframe_metrics(df)
    summary = {
        "rows": metrics["row_count"],
        "columns": metrics["column_count"],
        "memory": f"{metrics['memory_mb']:.2f} MB",
        "missing_values": metrics["missing_values"],
        "duplicates": metrics["duplicate_rows"],
        "quality_score": metrics["quality_score"],
    }

    sections = {
        "Executive Summary": "A professional data analysis workflow was completed for the uploaded dataset.",
        "Dataset Overview": f"Rows: {summary['rows']}\nColumns: {summary['columns']}\nMemory: {summary['memory']}\nMissing Values: {summary['missing_values']}\nDuplicates: {summary['duplicates']}",
        "Data Quality": "The dataset quality was assessed and summarized for review.",
        "Cleaning Summary": (
            f"Rows Removed: {cleaning_summary.get('duplicates_removed', 0)}\n"
            f"Missing Values Fixed: {cleaning_summary.get('missing_values_fixed', 0)}\n"
            f"Memory Saved: {cleaning_summary.get('memory_saved_mb', 0):.2f} MB"
        ),
        "Analytics Summary": "Summary statistics and visualization insights are available in the analytics dashboard.",
        "AI Business Insights": ai_report,
        "Report Generated By": "InsightFlow AI",
    }

    summary_table = generate_summary_table(summary)
    markdown_report = generate_markdown_report(metadata, sections, summary_table=summary_table)
    pdf_bytes = generate_pdf_report(metadata, sections, summary_table=summary_table)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            label="Download PDF Report",
            data=pdf_bytes,
            file_name="insightflow_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            label="Download Markdown Report",
            data=markdown_report.encode("utf-8"),
            file_name="report.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with col3:
        st.download_button(
            label="Download Cleaned CSV",
            data=(st.session_state.get("cleaned_df") if st.session_state.get("cleaned_df") is not None else df).to_csv(index=False).encode("utf-8"),
            file_name="cleaned_dataset.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.success("Reports are ready for download.")


def render_ai_business_analyst() -> None:
    """Render the AI business insights page."""
    st.markdown("## AI Business Analyst")
    st.caption("Generate concise, business-focused insights from your uploaded dataset.")

    df = st.session_state.get("cleaned_df")
    if df is None:
        df = st.session_state.get("dataframe")

    if df is None:
        st.info("Upload a dataset first to generate AI insights.")
        return

    if st.button("Generate AI Insights", use_container_width=True):
        with st.spinner("Generating AI business insights..."):
            context = create_dataset_context(df)
            report = generate_ai_insights(context)
            st.session_state.ai_report = report

    report = st.session_state.get("ai_report", "")
    if not report:
        st.info("No AI report generated yet. Click the button above to create one.")
        return

    st.markdown("### Report")
    st.download_button(
        label="Download AI Report",
        data=report.encode("utf-8"),
        file_name="business_insights.md",
        mime="text/markdown",
        use_container_width=True,
    )

    sections = [section.strip() for section in report.split("#") if section.strip()]
    for section in sections:
        if not section:
            continue
        title, _, body = section.partition("\n")
        if not body.strip():
            continue
        with st.container():
            st.markdown(f"### {title.strip()}")
            st.write(body.strip())


def render_analytics_dashboard() -> None:
    """Render the analytics dashboard with interactive charts and summaries."""
    st.markdown("## Analytics Dashboard")
    st.caption("Explore your dataset with interactive statistics, visualizations, and outlier analysis.")

    df = st.session_state.get("cleaned_df")
    if df is None:
        df = st.session_state.get("dataframe")

    if df is None:
        st.info("Upload a dataset to explore analytics.")
        return

    engine = AnalyticsEngine()
    summary = engine.summarize(df)
    st.session_state.last_analytics_summary = get_dataframe_metrics(df)

    cols = st.columns(6)
    metrics = [
        ("Rows", f"{summary.row_count:,}"),
        ("Columns", f"{summary.column_count:,}"),
        ("Numeric Features", f"{len(summary.numeric_columns):,}"),
        ("Categorical Features", f"{len(summary.categorical_columns):,}"),
        ("Missing Values", f"{sum(summary.missing_values.values()):,}"),
        ("Memory Usage", f"{df.memory_usage(deep=True).sum() / (1024 ** 2):.2f} MB"),
    ]
    for col, (label, value) in zip(cols, metrics):
        with col:
            st.metric(label=label, value=value)

    with st.expander("Summary Statistics", expanded=False):
        stats_df = summary_statistics(df)
        if stats_df.empty:
            st.info("No summary statistics available for this dataset.")
        else:
            st.dataframe(stats_df, use_container_width=True, hide_index=True)

    numeric_cols = numeric_columns(df)
    categorical_cols = categorical_columns(df)

    st.sidebar.markdown("### Analytics Controls")
    selected_numeric = st.sidebar.selectbox(
        "Select Numeric Column",
        options=numeric_cols if numeric_cols else ["No numeric columns"],
        index=0,
    )
    selected_second_numeric = st.sidebar.selectbox(
        "Select Second Numeric Column",
        options=numeric_cols if numeric_cols else ["No numeric columns"],
        index=1 if len(numeric_cols) > 1 else 0,
    )
    selected_categorical = st.sidebar.selectbox(
        "Select Categorical Column",
        options=categorical_cols if categorical_cols else ["No categorical columns"],
        index=0,
    )

    if selected_numeric == "No numeric columns" or selected_second_numeric == "No numeric columns":
        st.info("This dataset does not contain enough numeric columns for the interactive plots.")
        return

    if len(numeric_cols) >= 1:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_histogram(df, selected_numeric), use_container_width=True)
        with col2:
            st.plotly_chart(create_boxplot(df, selected_numeric), use_container_width=True)

    if len(numeric_cols) >= 2:
        st.plotly_chart(create_scatter(df, selected_numeric, selected_second_numeric), use_container_width=True)
    else:
        st.info("Add another numeric column to enable scatter plotting.")

    if len(numeric_cols) >= 2:
        st.plotly_chart(create_correlation_heatmap(df), use_container_width=True)
    else:
        st.info("At least two numeric columns are required for the correlation heatmap.")

    if selected_categorical != "No categorical columns":
        st.plotly_chart(
            px.bar(
                top_categories(df, selected_categorical),
                x=selected_categorical,
                y="Count",
                template="plotly_dark",
                title=f"Top Categories: {selected_categorical}",
            ).update_layout(margin=dict(l=20, r=20, t=40, b=20)),
            use_container_width=True,
        )
    else:
        st.info("No categorical columns are available for category analysis.")

    st.markdown("### Outlier Detection")
    outlier_df = detect_outliers(df)
    if outlier_df.empty:
        st.info("Outlier detection is not available for this dataset.")
    else:
        st.dataframe(outlier_df, use_container_width=True, hide_index=True)


def render_page(page: str) -> None:
    """Render the currently selected page."""
    pages = {
        "home": lambda: render_home_page(),
        "upload": lambda: render_upload_page(),
        "analytics": lambda: render_analytics_dashboard(),
        "ai_insights": lambda: render_ai_business_analyst(),
        "reports": lambda: render_reports_page(),
        "settings": lambda: render_placeholder_page(
            "Settings",
            "Configure API keys, display preferences, and export options.",
        ),
    }
    pages[page]()


def render_footer() -> None:
    """Render the shared footer for the application."""
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
