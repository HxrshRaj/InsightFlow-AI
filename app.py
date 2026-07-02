"""InsightFlow AI — Streamlit application entry point."""

from __future__ import annotations

import streamlit as st

from backend.utils import format_file_size

APP_TITLE = "InsightFlow AI"
APP_SUBTITLE = "Transform raw data into actionable insights with intelligent analytics"
APP_ICON = "📊"

NAV_ITEMS = {
    "Home": "home",
    "Upload Data": "upload",
    "Analytics": "analytics",
    "AI Insights": "ai_insights",
    "Reports": "reports",
    "Settings": "settings",
}

NAV_ICONS = {
    "Home": "🏠",
    "Upload Data": "📁",
    "Analytics": "📈",
    "AI Insights": "🤖",
    "Reports": "📄",
    "Settings": "⚙️",
}


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


def render_sidebar() -> str:
    st.sidebar.markdown(
        f'<div class="sidebar-brand">{APP_ICON} {APP_TITLE}</div>',
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("##### Navigation")
    selected = st.sidebar.radio(
        label="Navigation",
        options=list(NAV_ITEMS.keys()),
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


def render_upload_section() -> None:
    st.markdown('<div class="section-header">Upload Your Data</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="upload-zone">
            <div class="upload-icon">📂</div>
            <div class="upload-title">Drop your CSV file here</div>
            <div class="upload-hint">Supports .csv files up to 200 MB</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        label="Choose a CSV file",
        type=["csv"],
        help="Upload a comma-separated values file to begin analysis.",
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        file_size = format_file_size(uploaded_file.size)
        st.success(f"**{uploaded_file.name}** uploaded successfully ({file_size})")
        st.info("Data processing will be available in the next release.")
    else:
        st.markdown(
            """
            <div class="stat-row">
                <div class="stat-card">
                    <div class="stat-value">—</div>
                    <div class="stat-label">Rows</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">—</div>
                    <div class="stat-label">Columns</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">—</div>
                    <div class="stat-label">Missing</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">—</div>
                    <div class="stat-label">Quality Score</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_home_page() -> None:
    render_hero()
    render_feature_cards()
    render_upload_section()


def render_placeholder_page(title: str, description: str) -> None:
    st.markdown(f"## {title}")
    st.markdown(description)
    st.info("This section is under development and will be available soon.")


def render_page(page: str) -> None:
    pages = {
        "home": lambda: render_home_page(),
        "upload": lambda: render_placeholder_page(
            "Upload Data",
            "Import CSV files, validate schema, and preview your dataset before analysis.",
        ),
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
    inject_custom_css()

    current_page = render_sidebar()
    render_page(current_page)
    render_footer()


if __name__ == "__main__":
    main()
