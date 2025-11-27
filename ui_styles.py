import streamlit as st

def apply_custom_styles():
    st.markdown("""
    <style>
    /* ===== GLOBAL DARK THEME ===== */
    .main, .block-container {
        background-color: #0f172a !important;  /* Deep dark navy */
        color: #e2e8f0 !important;              /* Light text */
        padding-top: 1rem !important;
    }

    /* Remove streamlit default padding & white lines  */
    div[data-testid="stVerticalBlock"] > div {
        padding: 0 !important;
        margin: 0 !important;
    }

    /* Fix sidebar background */
    section[data-testid="stSidebar"] {
        background-color: #1e293b !important;
    }

    /* ===== CARD STYLING ===== */
    .professional-card {
        background: #1e293b !important;     /* Darker card */
        padding: 1.5rem;
        border-radius: 10px;
        border: none !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.4);
        margin-bottom: 1.2rem;
    }

    .emergency-card {
        background: #7f1d1d !important;   /* Dark red */
        border-left: 6px solid #dc2626 !important;
    }

    .status-card {
        background: #1e293b !important;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #334155 !important;
        margin-bottom: 1.2rem;
        text-align: center;
    }

    .status-waiting {
        border-color: #d97706 !important;
        background: #451a03 !important;
    }

    .status-approved {
        border-color: #059669 !important;
        background: #064e3b !important;
    }

    /* ===== HEADERS =====*/
    .section-header {
        color: #f8fafc !important;
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 0.8rem;
        border-bottom: 2px solid #334155;  /* MAKE IT DARK */
        padding-bottom: 0.3rem;
    }

    /* ===== FORM INPUTS ===== */
    .stTextInput input,
    .stNumberInput input,
    .stSelectbox select,
    .stTextArea textarea {
        background: #111827 !important;
        color: white !important;
        border: 1px solid #334155 !important;
        border-radius: 6px;
    }

    .stTextInput input:focus,
    .stNumberInput input:focus,
    .stSelectbox select:focus,
    .stTextArea textarea:focus {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.2) !important;
    }

    /* ===== BUTTONS ===== */
    .stButton button {
        background: linear-gradient(90deg, #2563eb, #1e40af);
        color: white !important;
        border-radius: 8px;
        border: none;
        font-weight: 500;
        padding: 0.6rem 1rem;
        transition: 0.2s ease-in-out;
    }
    .stButton button:hover {
        transform: scale(1.03);
        background: linear-gradient(90deg, #1e40af, #2563eb);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }

    /* ===== SIDEBAR ===== */
    .css-1d391kg, .css-1lcbmhc, .css-1outwn7 {
        background-color: #1e293b !important;
    }

    /* ===== METRIC CARDS ===== */
    [data-testid="metric-container"] {
        background: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 8px;
        padding: 1rem;
    }

    /* ===== CONVERSATION BOXES ===== */
    .chat-box {
        background: #1e293b;
        border-left: 4px solid #2563eb;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        color: #e2e8f0;
    }
    .chat-emergency {
        background: #2d1112;
        border-left: 4px solid #dc2626;
    }

    /* ===== CHECKBOX STYLING ===== */
    .stCheckbox label {
        color: #e2e8f0 !important;
    }

    /* ===== RADIO BUTTONS ===== */
    .stRadio label {
        color: #e2e8f0 !important;
    }

    /* ===== EXPANDER STYLING ===== */
    .streamlit-expanderHeader {
        background: #1e293b !important;
        color: #e2e8f0 !important;
        border: 1px solid #334155 !important;
    }

    /* ===== SCROLL BAR ===== */
    ::-webkit-scrollbar {
        width: 6px;
    }
    ::-webkit-scrollbar-thumb {
        background: #475569;
        border-radius: 10px;
    }

    /* ===== TEMPERATURE CONVERTER ===== */
    .temp-converter {
        background: #111827 !important;
        padding: 0.75rem;
        border-radius: 6px;
        border: 1px solid #334155 !important;
        margin: 0.5rem 0;
        color: #94a3b8;
    }

    /* ===== RISK INDICATORS ===== */
    .risk-low { border-left: 4px solid #059669 !important; }
    .risk-moderate { border-left: 4px solid #d97706 !important; }
    .risk-high { border-left: 4px solid #dc2626 !important; }
    .risk-very-high { border-left: 4px solid #7f1d1d !important; }
    </style>
    """, unsafe_allow_html=True)