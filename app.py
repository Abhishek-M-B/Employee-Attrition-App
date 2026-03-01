import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AttritionIQ · HR Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Dark background */
.stApp {
    background-color: #0A0E1A;
    color: #E8EAF0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1220 0%, #111827 100%);
    border-right: 1px solid #1E2A3A;
}

/* Headers */
h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #111827 0%, #1a2436 100%);
    border: 1px solid #1E2A3A;
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-2px);
    border-color: #3B82F6;
}
.metric-card .label {
    font-size: 0.78rem;
    color: #6B7280;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.metric-card .value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #F9FAFB;
    line-height: 1;
}
.metric-card .sub {
    font-size: 0.72rem;
    color: #9CA3AF;
    margin-top: 0.25rem;
}

/* Risk badge */
.risk-high {
    background: linear-gradient(135deg, #7f1d1d, #991b1b);
    border: 1px solid #ef4444;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
}
.risk-low {
    background: linear-gradient(135deg, #064e3b, #065f46);
    border: 1px solid #10b981;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
}
.risk-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    margin: 0;
}

/* Section headings */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #3B82F6;
    margin-bottom: 0.5rem;
}

/* ── GLOBAL: force all text/labels to be bright white ── */
label, p, span, div {
    color: #F1F5F9;
}

/* Section markdown text */
.stMarkdown p, .stMarkdown span, .stMarkdown li {
    color: #F1F5F9 !important;
}

/* Every widget label (slider, selectbox, radio, number input) */
[data-testid="stWidgetLabel"] > div > p,
[data-testid="stWidgetLabel"] p,
.stSlider label, .stSelectbox label,
.stNumberInput label, .stRadio label,
.stSelectSlider label {
    color: #FFFFFF !important;
    font-size: 0.92rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
}

/* Slider min/max tick text and current value */
[data-testid="stSlider"] p,
[data-testid="stSlider"] span,
[data-testid="stSlider"] div {
    color: #F1F5F9 !important;
}

/* Select slider option labels */
[data-testid="stSelectSlider"] p,
[data-testid="stSelectSlider"] span {
    color: #F1F5F9 !important;
}

/* Selectbox input box */
[data-testid="stSelectbox"] > div > div {
    background-color: #1E293B !important;
    border: 1.5px solid #3B82F6 !important;
    color: #FFFFFF !important;
    border-radius: 8px !important;
}
/* Selectbox selected text */
[data-baseweb="select"] span,
[data-baseweb="select"] div {
    color: #FFFFFF !important;
    font-size: 0.95rem !important;
}

/* Number input box */
[data-testid="stNumberInput"] > div > div > input {
    background-color: #1E293B !important;
    border: 1.5px solid #3B82F6 !important;
    color: #FFFFFF !important;
    border-radius: 8px !important;
    font-size: 1rem !important;
}

/* Slider track fill */
[data-testid="stSlider"] > div > div > div > div {
    background-color: #3B82F6 !important;
}

/* Radio button labels */
[data-testid="stRadio"] label,
[data-testid="stRadio"] > div > label {
    background: #1E293B !important;
    border: 1.5px solid #334155 !important;
    border-radius: 8px !important;
    padding: 0.3rem 0.8rem !important;
    color: #FFFFFF !important;
    font-weight: 500 !important;
    transition: border-color 0.15s;
}
[data-testid="stRadio"] > div > label:hover {
    border-color: #3B82F6 !important;
}

/* Caption text */
.stCaption, [data-testid="stCaptionContainer"] p {
    color: #94A3B8 !important;
}

/* Predict button */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #1d4ed8, #2563eb);
    color: white;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 0.05em;
    border: none;
    border-radius: 10px;
    padding: 0.75rem 2rem;
    width: 100%;
    transition: all 0.2s;
}
div[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #1e40af, #1d4ed8);
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
}

/* Tab styling */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: transparent;
    gap: 0.5rem;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: #111827;
    border: 1px solid #1E2A3A;
    border-radius: 8px;
    color: #9CA3AF;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    padding: 0.5rem 1.2rem;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: #1d4ed8 !important;
    border-color: #1d4ed8 !important;
    color: white !important;
}

/* Divider */
hr {
    border-color: #1E2A3A;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0A0E1A; }
::-webkit-scrollbar-thumb { background: #1E2A3A; border-radius: 3px; }

/* Alert overrides */
[data-testid="stAlert"] {
    border-radius: 10px;
    border: 1px solid #1E2A3A;
}
</style>
""", unsafe_allow_html=True)

# ─── Load Model ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        with open("attrition_pipeline.pkl", "rb") as f:
            pipeline = pickle.load(f)
        with open("best_threshold.pkl", "rb") as f:
            threshold = pickle.load(f)
        return pipeline, float(threshold), True
    except FileNotFoundError:
        return None, 0.43, False

pipeline, THRESHOLD, model_loaded = load_model()

# ─── Header ─────────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 10])
with col_title:
    st.markdown("""
    <h1 style='margin:0; padding:0; font-size:2.2rem; color:#F9FAFB;'>
        AttritionIQ
        <span style='font-size:1rem; font-weight:400; color:#4B5563; margin-left:1rem; font-family:DM Sans;'>
            HR Intelligence Platform
        </span>
    </h1>
    """, unsafe_allow_html=True)

if not model_loaded:
    st.warning("⚠️  **Model files not found.** Place `attrition_pipeline.pkl` and `best_threshold.pkl` in the same directory as this app. Running in **demo mode** with simulated predictions.")

st.markdown("---")

# ─── Tabs ────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🎯  Predict Attrition", "📊  Dashboard & Analytics"])

# ════════════════════════════════════════════════════════════════════════
# TAB 1: PREDICTION FORM
# ════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<p class='section-label'>Employee Risk Assessment</p>", unsafe_allow_html=True)

    form_col, result_col = st.columns([3, 2], gap="large")

    with form_col:
        with st.container():
            # ── Personal Info ──────────────────────────────────────────
            st.markdown("#### 👤 Personal Information")
            c1, c2, c3 = st.columns(3)
            with c1:
                age = st.slider("Age", 18, 65, 35, key="age")
            with c2:
                gender = st.radio("Gender", ["Male", "Female"], horizontal=True, key="gender")
            with c3:
                marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"], key="marital")

            c4, c5 = st.columns(2)
            with c4:
                education = st.selectbox("Education Level", [
                    "High School", "Associate Degree", "Bachelors Degree", "Masters Degree", "PhD"
                ], index=2, key="edu")
            with c5:
                dependents = st.slider("Number of Dependents", 0, 10, 1, key="dep")

            st.markdown("---")

            # ── Job Details ────────────────────────────────────────────
            st.markdown("#### 💼 Job Details")
            c6, c7 = st.columns(2)
            with c6:
                job_role = st.selectbox("Job Role", [
                    "Software Engineer", "Data Analyst", "Product Manager",
                    "HR Manager", "Sales Executive", "Finance Analyst",
                    "Marketing Specialist", "Operations Manager", "Customer Support"
                ], key="role")
            with c7:
                job_level = st.selectbox("Job Level", ["Entry", "Mid", "Senior"], index=1, key="jlevel")

            c8, c9 = st.columns(2)
            with c8:
                years_at_company = st.slider("Years at Company", 0, 40, 5, key="yac")
                # Auto-calculate tenure in months from years
                tenure_months = years_at_company * 12
                st.caption(f"📅 Tenure auto-set to **{tenure_months} months**")
            with c9:
                promotions = st.slider("# of Promotions", 0, 10, 1, key="promo")

            c11, c12 = st.columns(2)
            with c11:
                monthly_income = st.number_input("Monthly Income (₹/$)", 1000, 200000, 60000, step=5000, key="income")
            with c12:
                distance_home = st.slider("Distance from Home (km)", 0, 100, 10, key="dist")

            st.markdown("---")

            # ── Work Environment ────────────────────────────────────────
            st.markdown("#### 🏢 Work Environment")
            c13, c14 = st.columns(2)
            with c13:
                company_size = st.selectbox("Company Size", ["Small", "Medium", "Large"], index=1, key="csize")
            with c14:
                company_rep = st.selectbox("Company Reputation", ["Poor", "Fair", "Good", "Excellent"], index=2, key="crep")

            c15, c16 = st.columns(2)
            with c15:
                overtime = st.radio("Overtime", ["Yes", "No"], index=1, horizontal=True, key="ot")
            with c16:
                remote_work = st.radio("Remote Work", ["Yes", "No"], horizontal=True, key="rw")

            wlb = st.select_slider(
                "Work-Life Balance",
                options=["Poor", "Fair", "Good", "Excellent"],
                value="Good", key="wlb"
            )

            st.markdown("---")

            # ── Satisfaction & Growth ────────────────────────────────────
            st.markdown("#### 🌟 Satisfaction & Growth")
            c18, c19 = st.columns(2)
            with c18:
                job_sat = st.select_slider(
                    "Job Satisfaction",
                    options=["Low", "Medium", "High", "Very High"],
                    value="High", key="jsat"
                )
            with c19:
                perf_rating = st.select_slider(
                    "Performance Rating",
                    options=["Low", "Below Average", "Average", "High"],
                    value="Average", key="perf"
                )

            c20, c21, c22 = st.columns(3)
            with c20:
                emp_recognition = st.select_slider(
                    "Employee Recognition",
                    options=["Low", "Medium", "High", "Very High"],
                    value="Medium", key="recog"
                )
            with c21:
                leadership = st.radio("Leadership Opps", ["Yes", "No"], horizontal=True, key="lead")
            with c22:
                innovation = st.radio("Innovation Opps", ["Yes", "No"], horizontal=True, key="innov")

            st.markdown("---")
            predict_btn = st.button("⚡  Predict Attrition Risk", use_container_width=True)

    # ── Result Panel ──────────────────────────────────────────────────────
    with result_col:
        st.markdown("#### 📋 Risk Result")

        if predict_btn or "last_prob" in st.session_state:
            if predict_btn:
                # Feature engineering (mirrors training)
                import math
                log_income = math.log1p(monthly_income)
                promo_rate = promotions / years_at_company if years_at_company > 0 else 0
                high_perf_no_promo = 1 if (perf_rating == "High" and promotions == 0) else 0

                # We'll use a placeholder for Income_vs_Role_Avg (1.0 = average)
                income_vs_role_avg = 1.0

                input_data = pd.DataFrame([{
                    "Age": age,
                    "Distance from Home": distance_home,
                    "Years at Company": years_at_company,
                    "Number of Promotions": promotions,
                    "Number of Dependents": dependents,
                    "Company Tenure (In Months)": tenure_months,
                    "Monthly Income": log_income,
                    "Promotion_Rate": promo_rate,
                    "Income_vs_Role_Avg": income_vs_role_avg,
                    "Work-Life Balance": wlb,
                    "Performance Rating": perf_rating,
                    "Job Level": job_level,
                    "Company Reputation": company_rep,
                    "Education Level": education,
                    "Job Satisfaction": job_sat,
                    "Employee Recognition": emp_recognition,
                    "Gender": gender,
                    "Overtime": overtime,
                    "Remote Work": remote_work,
                    "Leadership Opportunities": leadership,
                    "Innovation Opportunities": innovation,
                    "HighPerf_NoPromo": high_perf_no_promo,
                    "Job Role": job_role,
                    "Marital Status": marital_status,
                    "Company Size": company_size,
                }])

                if model_loaded:
                    prob = pipeline.predict_proba(input_data)[0][1]
                else:
                    # Demo mode: heuristic simulation
                    risk_score = 0.3
                    if overtime == "Yes": risk_score += 0.15
                    if wlb in ["Poor", "Fair"]: risk_score += 0.1
                    if job_sat in ["Low", "Medium"]: risk_score += 0.1
                    if remote_work == "No": risk_score += 0.05
                    if emp_recognition in ["Low"]: risk_score += 0.08
                    if promo_rate == 0 and years_at_company > 3: risk_score += 0.07
                    prob = min(risk_score, 0.98)

                st.session_state["last_prob"] = prob
                st.session_state["last_threshold"] = THRESHOLD

            prob = st.session_state["last_prob"]
            threshold_val = st.session_state["last_threshold"]
            will_leave = prob >= threshold_val
            pct = prob * 100

            # Risk badge
            if will_leave:
                st.markdown(f"""
                <div class="risk-high">
                    <p style='color:#fca5a5; font-size:0.75rem; letter-spacing:0.15em; text-transform:uppercase; margin:0 0 4px 0;'>Prediction</p>
                    <p class='risk-title' style='color:#ef4444;'>⚠️ HIGH RISK</p>
                    <p style='color:#fca5a5; margin:4px 0 0 0; font-size:0.85rem;'>Likely to Leave</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-low">
                    <p style='color:#6ee7b7; font-size:0.75rem; letter-spacing:0.15em; text-transform:uppercase; margin:0 0 4px 0;'>Prediction</p>
                    <p class='risk-title' style='color:#10b981;'>✅ LOW RISK</p>
                    <p style='color:#6ee7b7; margin:4px 0 0 0; font-size:0.85rem;'>Likely to Stay</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pct,
                number={"suffix": "%", "font": {"size": 32, "color": "#F9FAFB", "family": "Syne"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#4B5563",
                             "tickfont": {"color": "#6B7280", "size": 10}},
                    "bar": {"color": "#ef4444" if will_leave else "#10b981", "thickness": 0.25},
                    "bgcolor": "#111827",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 40], "color": "#064e3b"},
                        {"range": [40, 65], "color": "#78350f"},
                        {"range": [65, 100], "color": "#7f1d1d"},
                    ],
                    "threshold": {
                        "line": {"color": "#F9FAFB", "width": 3},
                        "thickness": 0.8,
                        "value": threshold_val * 100
                    }
                },
                title={"text": "Attrition Probability", "font": {"color": "#9CA3AF", "size": 13}}
            ))
            fig_gauge.update_layout(
                height=240, margin=dict(t=40, b=0, l=20, r=20),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Key factors summary
            st.markdown("**📌 Key Risk Factors Detected:**")
            factors = []
            if overtime == "Yes": factors.append("🔴 Working overtime")
            if wlb in ["Poor", "Fair"]: factors.append("🔴 Poor work-life balance")
            if job_sat in ["Low", "Medium"]: factors.append("🟡 Low job satisfaction")
            if emp_recognition == "Low": factors.append("🟡 Lack of recognition")
            if remote_work == "No": factors.append("🟡 No remote work")
            if promo_rate == 0 and years_at_company > 3: factors.append("🔴 No promotions in 3+ years")
            if perf_rating == "High" and promotions == 0: factors.append("🔴 High performer, no promotion")

            if factors:
                for f in factors:
                    st.markdown(f"- {f}")
            else:
                st.markdown("✅ No major risk signals detected.")

            # Threshold info
            st.caption(f"Model threshold: {threshold_val:.2f} | Probability: {pct:.1f}%")

        else:
            st.markdown("""
            <div style='background:#111827; border:1px dashed #1E2A3A; border-radius:14px;
                        padding:2rem; text-align:center; color:#4B5563; margin-top:1rem;'>
                <p style='font-size:2.5rem; margin:0;'>🎯</p>
                <p style='font-family:Syne; font-size:1rem; color:#6B7280; margin:0.5rem 0 0 0;'>
                    Fill in the form and click<br><strong style='color:#3B82F6;'>Predict Attrition Risk</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════
# TAB 2: DASHBOARD
# ════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<p class='section-label'>Workforce Analytics Overview</p>", unsafe_allow_html=True)

    # Generate sample/demo data for the dashboard
    np.random.seed(42)
    n = 400
    demo_df = pd.DataFrame({
        "Age": np.random.randint(22, 60, n),
        "Department": np.random.choice(["Engineering", "Sales", "HR", "Finance", "Marketing", "Operations"], n,
                                        p=[0.3, 0.2, 0.1, 0.15, 0.15, 0.1]),
        "Job Level": np.random.choice(["Entry", "Mid", "Senior"], n, p=[0.35, 0.45, 0.2]),
        "Years at Company": np.random.randint(0, 30, n),
        "Monthly Income": np.random.lognormal(10.5, 0.6, n).astype(int),
        "Overtime": np.random.choice(["Yes", "No"], n, p=[0.35, 0.65]),
        "Work-Life Balance": np.random.choice(["Poor", "Fair", "Good", "Excellent"], n, p=[0.1, 0.25, 0.45, 0.2]),
        "Job Satisfaction": np.random.choice(["Low", "Medium", "High", "Very High"], n, p=[0.15, 0.25, 0.35, 0.25]),
        "Attrition": np.random.choice([0, 1], n, p=[0.72, 0.28]),
    })

    # Override attrition to be correlated
    demo_df.loc[(demo_df["Overtime"] == "Yes") & (demo_df["Work-Life Balance"].isin(["Poor", "Fair"])), "Attrition"] = \
        np.random.choice([0, 1], sum((demo_df["Overtime"] == "Yes") & (demo_df["Work-Life Balance"].isin(["Poor", "Fair"]))), p=[0.4, 0.6])

    attrition_rate = demo_df["Attrition"].mean() * 100
    total_emp = len(demo_df)
    at_risk = (demo_df["Attrition"] == 1).sum()
    avg_tenure = demo_df["Years at Company"].mean()

    # ── KPI Row ────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    kpi_data = [
        (k1, str(total_emp), "Total Employees", "active records"),
        (k2, f"{attrition_rate:.1f}%", "Attrition Rate", "of workforce"),
        (k3, str(at_risk), "At-Risk Employees", "flagged by model"),
        (k4, f"{avg_tenure:.1f} yrs", "Avg Tenure", "years at company"),
    ]
    for col, val, label, sub in kpi_data:
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='label'>{label}</div>
                <div class='value'>{val}</div>
                <div class='sub'>{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: Attrition by Dept + by Job Level ────────────────────────
    r1c1, r1c2 = st.columns(2, gap="medium")

    with r1c1:
        dept_attr = demo_df.groupby("Department")["Attrition"].mean().reset_index()
        dept_attr.columns = ["Department", "Rate"]
        dept_attr["Rate"] = dept_attr["Rate"] * 100
        dept_attr = dept_attr.sort_values("Rate", ascending=True)

        colors = ["#ef4444" if r > 35 else "#f59e0b" if r > 25 else "#10b981" for r in dept_attr["Rate"]]
        fig1 = go.Figure(go.Bar(
            x=dept_attr["Rate"], y=dept_attr["Department"],
            orientation="h", marker_color=colors,
            text=[f"{r:.1f}%" for r in dept_attr["Rate"]],
            textposition="outside", textfont=dict(color="#9CA3AF", size=11)
        ))
        fig1.update_layout(
            title=dict(text="Attrition Rate by Department", font=dict(color="#F9FAFB", family="Syne", size=14)),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, color="#4B5563", range=[0, 60]),
            yaxis=dict(color="#9CA3AF"),
            margin=dict(t=40, b=20, l=10, r=60), height=280
        )
        st.plotly_chart(fig1, use_container_width=True)

    with r1c2:
        level_attr = demo_df.groupby("Job Level")["Attrition"].mean().reset_index()
        level_attr["Rate"] = level_attr["Attrition"] * 100
        level_order = {"Entry": 0, "Mid": 1, "Senior": 2}
        level_attr["sort"] = level_attr["Job Level"].map(level_order)
        level_attr = level_attr.sort_values("sort")

        fig2 = go.Figure(go.Bar(
            x=level_attr["Job Level"], y=level_attr["Rate"],
            marker=dict(
                color=level_attr["Rate"],
                colorscale=[[0, "#10b981"], [0.5, "#f59e0b"], [1, "#ef4444"]],
                showscale=False
            ),
            text=[f"{r:.1f}%" for r in level_attr["Rate"]],
            textposition="outside", textfont=dict(color="#9CA3AF", size=12)
        ))
        fig2.update_layout(
            title=dict(text="Attrition Rate by Job Level", font=dict(color="#F9FAFB", family="Syne", size=14)),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(color="#9CA3AF"), yaxis=dict(showgrid=False, color="#4B5563"),
            margin=dict(t=40, b=20, l=20, r=20), height=280
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Row 2: Overtime Impact + WLB Breakdown ─────────────────────────
    r2c1, r2c2 = st.columns(2, gap="medium")

    with r2c1:
        ot_attr = demo_df.groupby("Overtime")["Attrition"].mean().reset_index()
        ot_attr["Rate"] = ot_attr["Attrition"] * 100
        fig3 = go.Figure(go.Bar(
            x=ot_attr["Overtime"], y=ot_attr["Rate"],
            marker_color=["#10b981", "#ef4444"],
            text=[f"{r:.1f}%" for r in ot_attr["Rate"]],
            textposition="outside", textfont=dict(color="#9CA3AF", size=12)
        ))
        fig3.update_layout(
            title=dict(text="Attrition: Overtime vs No Overtime", font=dict(color="#F9FAFB", family="Syne", size=14)),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(color="#9CA3AF"), yaxis=dict(showgrid=False, color="#4B5563"),
            margin=dict(t=40, b=20, l=20, r=20), height=280
        )
        st.plotly_chart(fig3, use_container_width=True)

    with r2c2:
        wlb_order = ["Poor", "Fair", "Good", "Excellent"]
        wlb_attr = demo_df.groupby("Work-Life Balance")["Attrition"].mean().reindex(wlb_order).reset_index()
        wlb_attr["Rate"] = wlb_attr["Attrition"] * 100
        fig4 = go.Figure(go.Scatter(
            x=wlb_attr["Work-Life Balance"], y=wlb_attr["Rate"],
            mode="lines+markers",
            line=dict(color="#3B82F6", width=3),
            marker=dict(size=10, color="#3B82F6", line=dict(color="#F9FAFB", width=2)),
            fill="tozeroy", fillcolor="rgba(59,130,246,0.1)",
            text=[f"{r:.1f}%" for r in wlb_attr["Rate"]],
            textposition="top center", textfont=dict(color="#9CA3AF")
        ))
        fig4.update_layout(
            title=dict(text="Attrition vs Work-Life Balance", font=dict(color="#F9FAFB", family="Syne", size=14)),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(color="#9CA3AF"), yaxis=dict(showgrid=False, color="#4B5563"),
            margin=dict(t=40, b=20, l=20, r=20), height=280
        )
        st.plotly_chart(fig4, use_container_width=True)

    # ── Row 3: Age Distribution + Income vs Attrition ──────────────────
    r3c1, r3c2 = st.columns(2, gap="medium")

    with r3c1:
        stayed = demo_df[demo_df["Attrition"] == 0]["Age"]
        left = demo_df[demo_df["Attrition"] == 1]["Age"]
        fig5 = go.Figure()
        fig5.add_trace(go.Histogram(x=stayed, name="Stayed", marker_color="#10b981",
                                    opacity=0.7, nbinsx=20))
        fig5.add_trace(go.Histogram(x=left, name="Left", marker_color="#ef4444",
                                    opacity=0.7, nbinsx=20))
        fig5.update_layout(
            barmode="overlay",
            title=dict(text="Age Distribution by Attrition", font=dict(color="#F9FAFB", family="Syne", size=14)),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(color="#9CA3AF", title="Age"),
            yaxis=dict(showgrid=False, color="#4B5563"),
            legend=dict(font=dict(color="#9CA3AF"), bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=40, b=20, l=20, r=20), height=280
        )
        st.plotly_chart(fig5, use_container_width=True)

    with r3c2:
        fig6 = go.Figure()
        for label, color, name in [(0, "#10b981", "Stayed"), (1, "#ef4444", "Left")]:
            subset = demo_df[demo_df["Attrition"] == label]
            fig6.add_trace(go.Box(
                y=subset["Monthly Income"], name=name,
                marker_color=color, line_color=color,
                fillcolor=f"rgba({','.join(str(int(c*255)) for c in px.colors.hex_to_rgb(color))},0.15)"
            ))
        fig6.update_layout(
            title=dict(text="Monthly Income Distribution", font=dict(color="#F9FAFB", family="Syne", size=14)),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(color="#9CA3AF"),
            yaxis=dict(showgrid=False, color="#4B5563", title="Income"),
            legend=dict(font=dict(color="#9CA3AF"), bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=40, b=20, l=20, r=20), height=280
        )
        st.plotly_chart(fig6, use_container_width=True)

    # ── Feature Importance (static SHAP-based) ─────────────────────────
    st.markdown("---")
    st.markdown("#### 🔍 Top Attrition Drivers (SHAP Analysis)")

    features_imp = [
        "Overtime", "Monthly Income", "Work-Life Balance",
        "Job Satisfaction", "Years at Company", "Age",
        "Employee Recognition", "Company Reputation",
        "Promotion Rate", "Remote Work"
    ]
    left_pct = [68, 55, 61, 57, 48, 42, 53, 44, 46, 39]
    stay_pct = [32, 45, 39, 43, 52, 58, 47, 56, 54, 61]

    fig7 = go.Figure()
    fig7.add_trace(go.Bar(
        y=features_imp, x=left_pct, orientation="h",
        name="→ Pushed to LEAVE", marker_color="#ef4444",
        text=[f"{v}%" for v in left_pct], textposition="inside",
        textfont=dict(color="white", size=10)
    ))
    fig7.add_trace(go.Bar(
        y=features_imp, x=[-v for v in stay_pct], orientation="h",
        name="→ Pushed to STAY", marker_color="#10b981",
        text=[f"{v}%" for v in stay_pct], textposition="inside",
        textfont=dict(color="white", size=10)
    ))
    fig7.update_layout(
        barmode="relative",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, zeroline=True, zerolinecolor="#374151",
                   color="#4B5563", tickvals=[-80, -60, -40, -20, 0, 20, 40, 60, 80],
                   ticktext=["80%", "60%", "40%", "20%", "0", "20%", "40%", "60%", "80%"]),
        yaxis=dict(color="#9CA3AF"),
        legend=dict(font=dict(color="#9CA3AF"), bgcolor="rgba(0,0,0,0)", orientation="h", y=1.08),
        margin=dict(t=40, b=20, l=20, r=20), height=360
    )
    st.plotly_chart(fig7, use_container_width=True)

    st.caption("📌 Dashboard uses sample data for visualization. Connect your workforce dataset for live analytics.")