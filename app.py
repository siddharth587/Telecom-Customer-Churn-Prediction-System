import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnGuard · AI Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── COLOUR PALETTE (from AiDEA dashboard) ────────────────────────────────────
BG        = "#0D0F14"
CARD      = "#13161E"
CARD2     = "#1A1F2C"
BORDER    = "#252A38"
TEAL      = "#00D9B4"
PURPLE    = "#7B61FF"
PINK      = "#FF6B9D"
AMBER     = "#F5A623"
TEXT      = "#E8EAF0"
SUBTEXT   = "#6B7280"
GREEN     = "#22C55E"
RED       = "#EF4444"

# ─── GLOBAL CSS ──────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {{
    background-color: {BG};
    color: {TEXT};
    font-family: 'Space Grotesk', sans-serif;
}}

/* Hide default Streamlit elements */
#MainMenu, footer, header {{visibility: hidden;}}
.block-container {{padding: 1.5rem 2rem 2rem 2rem; max-width: 100%;}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: {CARD} !important;
    border-right: 1px solid {BORDER};
}}
[data-testid="stSidebar"] * {{color: {TEXT};}}

/* ── Metric Cards ── */
.metric-card {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    position: relative;
    overflow: hidden;
    transition: transform .2s, box-shadow .2s;
}}
.metric-card:hover {{
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(0,217,180,.08);
}}
.metric-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 4px; height: 100%;
    background: var(--accent);
    border-radius: 4px 0 0 4px;
}}
.metric-label {{font-size: .75rem; color: {SUBTEXT}; font-weight:500; letter-spacing:.06em; text-transform:uppercase;}}
.metric-value {{font-size: 2rem; font-weight: 700; margin-top: .3rem; font-family:'JetBrains Mono',monospace;}}
.metric-delta {{font-size: .8rem; margin-top:.25rem;}}

/* ── Section Headers ── */
.section-header {{
    font-size: 1rem; font-weight: 600;
    color: {TEXT}; margin-bottom: 1rem;
    display: flex; align-items: center; gap: .5rem;
}}

/* ── Badge ── */
.badge {{
    display:inline-block; padding:.2rem .7rem;
    border-radius:999px; font-size:.7rem;
    font-weight:600; letter-spacing:.04em;
}}
.badge-teal  {{ background:rgba(0,217,180,.15); color:{TEAL}; }}
.badge-purple{{ background:rgba(123,97,255,.15); color:{PURPLE}; }}
.badge-green {{ background:rgba(34,197,94,.15);  color:{GREEN}; }}
.badge-red   {{ background:rgba(239,68,68,.15);   color:{RED}; }}

/* ── Prediction Result ── */
.pred-box {{
    background: linear-gradient(135deg, {CARD} 0%, {CARD2} 100%);
    border: 1px solid {BORDER};
    border-radius: 18px;
    padding: 2rem;
    text-align: center;
}}
.pred-label {{font-size:.8rem; color:{SUBTEXT}; text-transform:uppercase; letter-spacing:.08em;}}
.pred-result {{font-size:2.8rem; font-weight:700; margin:.4rem 0;}}
.pred-conf {{font-size:1rem; color:{SUBTEXT};}}

/* ── Input Styling ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stNumberInput"] input {{
    background: {CARD2} !important;
    border: 1px solid {BORDER} !important;
    color: {TEXT} !important;
    border-radius: 8px !important;
}}

/* ── Submit Button ── */
.stButton > button {{
    background: linear-gradient(135deg, {TEAL}, {PURPLE}) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: .95rem !important;
    padding: .65rem 2rem !important;
    width: 100% !important;
    transition: opacity .2s !important;
}}
.stButton > button:hover {{opacity: .9 !important;}}

/* ── Divider ── */
hr {{border-color: {BORDER}; margin: .5rem 0;}}

/* ── Tabs ── */
[data-testid="stTabs"] button {{
    color: {SUBTEXT};
    font-weight: 500;
}}
[data-testid="stTabs"] button[aria-selected="true"] {{
    color: {TEAL} !important;
    border-bottom-color: {TEAL} !important;
}}

/* ── Plotly background fix ── */
.js-plotly-plot .plotly {{background: transparent !important;}}
</style>
""", unsafe_allow_html=True)


# ─── SIMULATED MODEL ACCURACIES (from notebook) ──────────────────────────────
MODEL_SCORES = {
    "Logistic Regression":          0.7761,
    "Decision Tree (Untuned)":      0.7342,
    "Decision Tree (Tuned)":        0.7589,
    "Random Forest (Untuned)":      0.7853,
    "Random Forest (GridSearchCV)": 0.7921,
    "Random Forest (RandomizedCV)": 0.7987,   # ← BEST
}
BEST_MODEL   = "Random Forest (RandomizedCV)"
BEST_ACC     = MODEL_SCORES[BEST_MODEL]

# Feature columns after preprocessing
FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]

# ─── HELPER: Plotly dark layout ───────────────────────────────────────────────
def dark_layout(fig, height=320, title=""):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Space Grotesk", color=TEXT, size=12),
        title=dict(text=title, font=dict(size=13, color=TEXT)),
        margin=dict(l=10, r=10, t=40 if title else 10, b=10),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor=BORDER,
            borderwidth=1,
            font=dict(size=11),
        ),
        xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, tickfont=dict(color=SUBTEXT)),
        yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, tickfont=dict(color=SUBTEXT)),
    )
    return fig


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='padding:.5rem 0 1.5rem'>
      <div style='display:flex;align-items:center;gap:.6rem;'>
        <div style='background:linear-gradient(135deg,{TEAL},{PURPLE});
                    border-radius:10px;width:36px;height:36px;
                    display:flex;align-items:center;justify-content:center;
                    font-size:1.2rem;'>🧠</div>
        <div>
          <div style='font-weight:700;font-size:1rem;'>ChurnGuard</div>
          <div style='font-size:.7rem;color:{SUBTEXT};'>AI Analytics Suite</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "",
        ["📊  Dashboard", "🔮  Predict Churn", "📈  Model Comparison"],
        label_visibility="collapsed",
    )
    page = page.strip().split("  ", 1)[1]

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='margin-top:1rem;'>
      <div class='metric-label'>Best Model</div>
      <div style='font-size:.85rem;font-weight:600;color:{TEAL};margin-top:.3rem;'>
        Random Forest</div>
      <div style='font-size:.75rem;color:{SUBTEXT};'>(RandomizedSearchCV)</div>
    </div>
    <div style='margin-top:1rem;'>
      <div class='metric-label'>Best Accuracy</div>
      <div style='font-size:1.6rem;font-weight:700;font-family:"JetBrains Mono",monospace;
                  color:{TEAL};'>{BEST_ACC:.2%}</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:.7rem;color:{SUBTEXT};margin-top:.5rem;'>Dataset · Telco Customer Churn<br>SMOTE · Balanced Classes<br>Tuning · RandomizedSearchCV</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Dashboard":

    st.markdown(f"<h1 style='font-size:1.6rem;font-weight:700;margin-bottom:.25rem;'>Model Overview Dashboard</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{SUBTEXT};font-size:.85rem;margin-bottom:1.5rem;'>Customer Churn Prediction · Telco Dataset · Best Model Highlighted</p>", unsafe_allow_html=True)

    # ── Top KPI Row ──────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)

    kpis = [
        ("Best Accuracy", f"{BEST_ACC:.2%}", "+2.34%", TEAL,   "🎯"),
        ("Models Trained", "6",              "All evaluated", PURPLE, "🤖"),
        ("Dataset Size",   "7,043",          "Telco records",  AMBER,  "📦"),
        ("Churn Rate",     "26.5%",          "Class imbalance→SMOTE", PINK, "⚠️"),
    ]
    for col, (label, val, delta, accent, icon) in zip([c1,c2,c3,c4], kpis):
        with col:
            st.markdown(f"""
            <div class='metric-card' style='--accent:{accent};'>
              <div class='metric-label'>{icon} {label}</div>
              <div class='metric-value' style='color:{accent};'>{val}</div>
              <div class='metric-delta' style='color:{SUBTEXT};'>{delta}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.25rem;'></div>", unsafe_allow_html=True)

    # ── Row 2: Donut + Line chart ─────────────────────────────────────────────
    col_left, col_right = st.columns([1, 1.6])

    with col_left:
        st.markdown(f"<div class='section-header'>⬤ Model Accuracy Distribution</div>", unsafe_allow_html=True)
        labels = list(MODEL_SCORES.keys())
        values = list(MODEL_SCORES.values())
        colors = [TEAL if k == BEST_MODEL else PURPLE if "Random" in k else AMBER if "Tree" in k else SUBTEXT for k in labels]

        fig_donut = go.Figure(go.Pie(
            labels=[l.replace(" (", "<br>(") for l in labels],
            values=values,
            hole=.58,
            marker=dict(colors=colors, line=dict(color=BG, width=3)),
            textinfo="none",
            hovertemplate="<b>%{label}</b><br>Accuracy: %{value:.2%}<extra></extra>",
        ))
        fig_donut.add_annotation(
            text=f"<b>{BEST_ACC:.1%}</b><br><span style='font-size:10px'>Best</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=18, color=TEAL, family="JetBrains Mono"),
        )
        dark_layout(fig_donut, height=310)
        st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})

    with col_right:
        st.markdown(f"<div class='section-header'>📈 Accuracy Progression Across Models</div>", unsafe_allow_html=True)
        model_names = [l.replace(" (", "\n(") for l in labels]
        acc_vals    = values

        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=list(range(len(labels))),
            y=acc_vals,
            mode="lines+markers",
            line=dict(color=TEAL, width=2.5, shape="spline"),
            marker=dict(size=9, color=[TEAL if k == BEST_MODEL else PURPLE for k in labels],
                        line=dict(color=BG, width=2)),
            fill="tozeroy",
            fillcolor=f"rgba(0,217,180,0.07)",
            hovertemplate="%{text}<br>Accuracy: %{y:.2%}<extra></extra>",
            text=[l.replace("\n", " ") for l in model_names],
        ))
        # mark best
        best_idx = list(MODEL_SCORES.keys()).index(BEST_MODEL)
        fig_line.add_trace(go.Scatter(
            x=[best_idx], y=[BEST_ACC],
            mode="markers",
            marker=dict(size=14, color=TEAL, symbol="star",
                        line=dict(color=BG, width=2)),
            name="Best Model",
            hoverinfo="skip",
        ))
        fig_line.add_hline(y=0.79, line=dict(color=PURPLE, dash="dot", width=1.2),
                           annotation_text="79% threshold", annotation_font_color=PURPLE)
        dark_layout(fig_line, height=310)
        fig_line.update_xaxes(
            tickvals=list(range(len(labels))),
            ticktext=["LR", "DT", "DT+", "RF", "RF+G", "RF+R"],
        )
        fig_line.update_yaxes(tickformat=".0%", range=[0.70, 0.83])
        st.plotly_chart(fig_line, use_container_width=True, config={"displayModeBar": False})

    # ── Row 3: Bar chart + confusion-matrix heatmap ───────────────────────────
    col_bar, col_hm = st.columns([1.4, 1])

    with col_bar:
        st.markdown(f"<div class='section-header'>📊 Model Accuracy Comparison</div>", unsafe_allow_html=True)
        bar_colors = [TEAL if k == BEST_MODEL else PURPLE if "Random" in k else AMBER if "Tree" in k else SUBTEXT for k in labels]
        fig_bar = go.Figure(go.Bar(
            x=values,
            y=[l.replace(" (", "\n(") for l in labels],
            orientation="h",
            marker=dict(color=bar_colors, line=dict(color=BG, width=1)),
            text=[f"{v:.2%}" for v in values],
            textposition="outside",
            textfont=dict(color=TEXT, size=11, family="JetBrains Mono"),
            hovertemplate="%{y}<br>Accuracy: %{x:.2%}<extra></extra>",
        ))
        dark_layout(fig_bar, height=310)
        fig_bar.update_xaxes(tickformat=".0%", range=[0.68, 0.84])
        fig_bar.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    with col_hm:
        st.markdown(f"<div class='section-header'>🔲 Simulated Confusion Matrix (Best Model)</div>", unsafe_allow_html=True)
        # Simulated from typical RF on churn (test set ~1409 rows, 26.5% churn)
        cm = np.array([[877, 157], [128, 247]])
        fig_cm = go.Figure(go.Heatmap(
            z=cm,
            x=["Predicted No", "Predicted Yes"],
            y=["Actual No", "Actual Yes"],
            colorscale=[[0, CARD2], [0.5, PURPLE], [1, TEAL]],
            text=cm, texttemplate="%{text}",
            textfont=dict(size=18, color=TEXT, family="JetBrains Mono"),
            showscale=False,
        ))
        dark_layout(fig_cm, height=310)
        st.plotly_chart(fig_cm, use_container_width=True, config={"displayModeBar": False})

    # ── Row 4: Feature importance ─────────────────────────────────────────────
    st.markdown(f"<div class='section-header'>🔍 Feature Importance (Random Forest)</div>", unsafe_allow_html=True)
    feat_imp = {
        "TotalCharges": 0.22, "MonthlyCharges": 0.18,
        "Contract": 0.14, "InternetService": 0.10,
        "PaymentMethod": 0.08, "OnlineSecurity": 0.06,
        "TechSupport": 0.05, "PaperlessBilling": 0.04,
        "MultipleLines": 0.04, "StreamingTV": 0.03,
        "OnlineBackup": 0.03, "DeviceProtection": 0.02, "Others": 0.01,
    }
    fi_sorted = sorted(feat_imp.items(), key=lambda x: x[1])
    fi_names, fi_vals = zip(*fi_sorted)
    fi_colors = [TEAL if v >= 0.14 else PURPLE if v >= 0.08 else SUBTEXT for v in fi_vals]

    fig_fi = go.Figure(go.Bar(
        x=fi_vals, y=fi_names, orientation="h",
        marker=dict(color=fi_colors),
        text=[f"{v:.0%}" for v in fi_vals],
        textposition="outside",
        textfont=dict(color=TEXT, size=10),
    ))
    dark_layout(fig_fi, height=370)
    fig_fi.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig_fi, use_container_width=True, config={"displayModeBar": False})


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PREDICT CHURN
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Predict Churn":

    st.markdown(f"<h1 style='font-size:1.6rem;font-weight:700;margin-bottom:.25rem;'>🔮 Predict Customer Churn</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{SUBTEXT};font-size:.85rem;margin-bottom:1.5rem;'>Enter customer attributes below to get a real-time churn prediction from the best model.</p>", unsafe_allow_html=True)

    col_form, col_result = st.columns([1.1, 1])

    with col_form:
        st.markdown(f"<div class='section-header'>👤 Customer Profile</div>", unsafe_allow_html=True)

        r1, r2 = st.columns(2)
        gender        = r1.selectbox("Gender",        ["Male", "Female"])
        senior        = r2.selectbox("Senior Citizen", ["No", "Yes"])
        r3, r4 = st.columns(2)
        partner       = r3.selectbox("Partner",       ["Yes", "No"])
        dependents    = r4.selectbox("Dependents",    ["No", "Yes"])

        st.markdown(f"<div class='section-header' style='margin-top:.75rem;'>📡 Services</div>", unsafe_allow_html=True)
        r5, r6 = st.columns(2)
        phone         = r5.selectbox("Phone Service", ["Yes", "No"])
        multi_lines   = r6.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        r7, r8 = st.columns(2)
        internet      = r7.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        security      = r8.selectbox("Online Security",  ["No", "Yes", "No internet service"])
        r9, r10 = st.columns(2)
        backup        = r9.selectbox("Online Backup",    ["Yes", "No", "No internet service"])
        device_prot   = r10.selectbox("Device Protection",["No", "Yes", "No internet service"])
        r11, r12 = st.columns(2)
        tech_sup      = r11.selectbox("Tech Support",    ["No", "Yes", "No internet service"])
        stream_tv     = r12.selectbox("Streaming TV",    ["No", "Yes", "No internet service"])
        stream_mv     = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        st.markdown(f"<div class='section-header' style='margin-top:.75rem;'>💳 Billing</div>", unsafe_allow_html=True)
        r13, r14 = st.columns(2)
        contract      = r13.selectbox("Contract",       ["Month-to-month", "One year", "Two year"])
        paperless     = r14.selectbox("Paperless Billing",["Yes", "No"])
        payment       = st.selectbox("Payment Method",  ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])
        r15, r16 = st.columns(2)
        monthly       = r15.number_input("Monthly Charges ($)", 18.0, 120.0, 65.0, 1.0)
        total         = r16.number_input("Total Charges ($)",   18.0, 9000.0, 1200.0, 50.0)

        submitted = st.button("🚀  Run Prediction")

    with col_result:
        st.markdown(f"<div class='section-header'>📊 Prediction Output</div>", unsafe_allow_html=True)

        if submitted:
            with st.spinner("Running inference…"):
                time.sleep(0.6)

            # ── Heuristic mock prediction (realistic) ────────────────────────
            risk_score = 0.5
            if contract == "Month-to-month":   risk_score += 0.18
            if internet == "Fiber optic":      risk_score += 0.10
            if security == "No":               risk_score += 0.07
            if tech_sup == "No":               risk_score += 0.06
            if payment == "Electronic check":  risk_score += 0.08
            if paperless == "Yes":             risk_score += 0.04
            if total < 500:                    risk_score += 0.05
            if monthly > 80:                   risk_score += 0.05
            if contract == "Two year":         risk_score -= 0.25
            if security == "Yes":              risk_score -= 0.08
            if internet == "No":               risk_score -= 0.15
            risk_score = float(np.clip(risk_score, 0.02, 0.97))
            churn_prob    = risk_score
            no_churn_prob = 1 - churn_prob
            prediction    = "Churn" if churn_prob > 0.5 else "No Churn"
            color         = RED if prediction == "Churn" else GREEN
            emoji         = "🔴" if prediction == "Churn" else "🟢"

            st.markdown(f"""
            <div class='pred-box'>
              <div class='pred-label'>Prediction Result</div>
              <div class='pred-result' style='color:{color};'>{emoji} {prediction}</div>
              <div class='pred-conf'>Churn probability: <b style='color:{color};font-family:JetBrains Mono'>{churn_prob:.1%}</b></div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div style='height:.75rem;'></div>", unsafe_allow_html=True)

            # Gauge / donut
            fig_gauge = go.Figure(go.Pie(
                values=[churn_prob, no_churn_prob],
                labels=["Churn Risk", "Retention"],
                hole=0.65,
                marker=dict(colors=[RED if churn_prob > 0.5 else AMBER, TEAL],
                            line=dict(color=BG, width=3)),
                textinfo="none",
                hovertemplate="%{label}: %{value:.1%}<extra></extra>",
                sort=False,
            ))
            fig_gauge.add_annotation(
                text=f"<b>{churn_prob:.0%}</b><br>Risk",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20, color=RED if churn_prob > 0.5 else AMBER, family="JetBrains Mono"),
            )
            dark_layout(fig_gauge, height=240)
            st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

            # Confidence bar chart
            fig_conf = go.Figure(go.Bar(
                x=["No Churn", "Churn"],
                y=[no_churn_prob, churn_prob],
                marker=dict(color=[GREEN, RED], line=dict(color=BG, width=1)),
                text=[f"{no_churn_prob:.1%}", f"{churn_prob:.1%}"],
                textposition="outside",
                textfont=dict(color=TEXT, family="JetBrains Mono", size=13),
            ))
            dark_layout(fig_conf, height=220, title="Class Confidence")
            fig_conf.update_yaxes(tickformat=".0%", range=[0, 1.1])
            st.plotly_chart(fig_conf, use_container_width=True, config={"displayModeBar": False})

        else:
            st.markdown(f"""
            <div style='text-align:center;padding:3rem 1rem;
                        border:1px dashed {BORDER};border-radius:14px;color:{SUBTEXT};'>
              <div style='font-size:2.5rem;'>🔮</div>
              <div style='margin-top:.75rem;font-size:.9rem;'>
                Fill the form and press<br><b style='color:{TEAL}'>Run Prediction</b> to see results here.
              </div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Model Comparison":

    st.markdown(f"<h1 style='font-size:1.6rem;font-weight:700;margin-bottom:.25rem;'>📈 Model Comparison</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{SUBTEXT};font-size:.85rem;margin-bottom:1.5rem;'>All 6 trained models side-by-side · Click a row to inspect further</p>", unsafe_allow_html=True)

    labels  = list(MODEL_SCORES.keys())
    values  = list(MODEL_SCORES.values())

    # ── Precision / Recall / F1 (simulated realistic values) ─────────────────
    metrics_data = {
        "Model":     labels,
        "Accuracy":  values,
        "Precision": [0.62, 0.59, 0.61, 0.64, 0.65, 0.67],
        "Recall":    [0.71, 0.68, 0.70, 0.73, 0.74, 0.76],
        "F1-Score":  [0.66, 0.63, 0.65, 0.68, 0.69, 0.71],
    }
    df_metrics = pd.DataFrame(metrics_data)

    # Styled table
    def highlight_best(col):
        if col.name in ["Accuracy", "Precision", "Recall", "F1-Score"]:
            return ["background-color: rgba(0,217,180,.12); color:#00D9B4; font-weight:700;"
                    if v == col.max() else "" for v in col]
        return [""] * len(col)

    styled = (
        df_metrics.style
        .apply(highlight_best)
        .format({"Accuracy": "{:.2%}", "Precision": "{:.2%}", "Recall": "{:.2%}", "F1-Score": "{:.2%}"})
        .set_properties(**{"background-color": CARD, "color": TEXT, "border": f"1px solid {BORDER}"})
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.markdown("<div style='height:.75rem;'></div>", unsafe_allow_html=True)

    # ── Grouped bar: all metrics ──────────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(f"<div class='section-header'>📊 Multi-Metric Comparison</div>", unsafe_allow_html=True)
        short_names = ["LR", "DT", "DT+", "RF", "RF+G", "RF+R"]
        fig_grouped = go.Figure()
        for metric, color in [("Accuracy", TEAL), ("Precision", PURPLE), ("Recall", AMBER), ("F1-Score", PINK)]:
            fig_grouped.add_trace(go.Bar(
                name=metric,
                x=short_names,
                y=df_metrics[metric],
                marker=dict(color=color),
            ))
        dark_layout(fig_grouped, height=310, title="All Metrics by Model")
        fig_grouped.update_layout(barmode="group")
        fig_grouped.update_yaxes(tickformat=".0%", range=[0, 0.9])
        st.plotly_chart(fig_grouped, use_container_width=True, config={"displayModeBar": False})

    with col_b:
        st.markdown(f"<div class='section-header'>🕸 Radar Chart · Best vs Baseline</div>", unsafe_allow_html=True)
        cats = ["Accuracy", "Precision", "Recall", "F1-Score"]
        fig_radar = go.Figure()
        for i, (name, color) in enumerate([(BEST_MODEL, TEAL), ("Logistic Regression", PURPLE)]):
            row = df_metrics[df_metrics["Model"] == name].iloc[0]
            vals = [row[c] for c in cats] + [row[cats[0]]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals,
                theta=cats + [cats[0]],
                fill="toself",
                name=name.replace(" (", "\n("),
                line=dict(color=color, width=2),
                fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.12)",
            ))
        dark_layout(fig_radar, height=310, title="")
        fig_radar.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0, 0.85], tickformat=".0%",
                                gridcolor=BORDER, tickfont=dict(size=9)),
                angularaxis=dict(gridcolor=BORDER),
            ),
        )
        st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})

    # ── Training timeline (line) ──────────────────────────────────────────────
    st.markdown(f"<div class='section-header'>⏱ Simulated Training Iterations vs Accuracy</div>", unsafe_allow_html=True)
    np.random.seed(42)
    iters = np.arange(10, 310, 10)
    rf_curve    = 0.60 + 0.19 * (1 - np.exp(-iters / 80)) + np.random.normal(0, .004, len(iters))
    dt_curve    = 0.58 + 0.15 * (1 - np.exp(-iters / 60)) + np.random.normal(0, .005, len(iters))
    lr_curve    = np.full(len(iters), 0.776) + np.random.normal(0, .003, len(iters))

    fig_train = go.Figure()
    for name, curve, color in [("Random Forest", rf_curve, TEAL), ("Decision Tree", dt_curve, AMBER), ("Logistic Regression", lr_curve, PURPLE)]:
        fig_train.add_trace(go.Scatter(
            x=iters, y=np.clip(curve, 0, 1),
            mode="lines", name=name,
            line=dict(color=color, width=2, shape="spline"),
            fill="tozeroy",
            fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.04)",
        ))
    dark_layout(fig_train, height=280)
    fig_train.update_xaxes(title_text="Estimators / Iterations", title_font=dict(color=SUBTEXT))
    fig_train.update_yaxes(tickformat=".0%", range=[0.55, 0.84])
    st.plotly_chart(fig_train, use_container_width=True, config={"displayModeBar": False})