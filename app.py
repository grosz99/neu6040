"""
ALY 6040 Data Mining — Interactive Lab
Northeastern University | Spring 2026 | Prof. Grosz

One app, five weekly lessons. Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import StandardScaler

# ──────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ALY 6040 Data Mining Lab",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# NEU Theme injection
# ──────────────────────────────────────────────────────────────
NEU_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Lato:wght@300;400;700;900&display=swap');

/* Force light mode on main content */
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > *,
.main, .main > *,
[data-testid="stMainBlockContainer"],
section[data-testid="stMain"] {
    background-color: #FFFFFF !important;
    color: #333333 !important;
}
html, body, [class*="css"] { font-family: 'Lato', sans-serif; color: #333333 !important; }
p, span, li, td, th, label, div { color: #333333 !important; }
h1, h2, h3 { font-family: 'Lato', sans-serif !important; color: #D31B2C !important; font-weight: 700 !important; }

/* Dark sidebar */
[data-testid="stSidebar"] { background-color: #1A1A1A !important; }
[data-testid="stSidebar"] *,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div { color: #FFFFFF !important; }
[data-testid="stSidebar"] img { background: white; border-radius: 50%; padding: 4px; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stMultiSelect label { color: #E6D8D3 !important; }
[data-testid="stSidebar"] [data-baseweb="select"],
[data-testid="stSidebar"] [data-baseweb="select"] *,
[data-testid="stSidebar"] [data-baseweb="select"] div,
[data-testid="stSidebar"] [data-baseweb="select"] span,
[data-testid="stSidebar"] [data-baseweb="input"],
[data-testid="stSidebar"] [data-baseweb="input"] * { background-color: #333333 !important; color: #FFFFFF !important; }
[data-testid="stSidebar"] [data-baseweb="popover"],
[data-testid="stSidebar"] [data-baseweb="popover"] *,
[data-testid="stSidebar"] [role="listbox"],
[data-testid="stSidebar"] [role="listbox"] *,
[data-testid="stSidebar"] [role="option"],
[data-testid="stSidebar"] [role="option"] * { background-color: #333333 !important; color: #FFFFFF !important; }
[data-testid="stSidebar"] [role="option"]:hover { background-color: #555555 !important; }

/* Metrics */
[data-testid="stMetric"] { background-color: #FFFFFF !important; border-left: 4px solid #D31B2C; padding: 12px 16px; border-radius: 4px; }
[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #D31B2C !important; font-weight: 700 !important; }
[data-testid="stMetric"] [data-testid="stMetricLabel"],
[data-testid="stMetric"] [data-testid="stMetricLabel"] * { color: #333333 !important; }
[data-testid="stMetric"] [data-testid="stMetricDelta"],
[data-testid="stMetric"] [data-testid="stMetricDelta"] * { color: #555555 !important; }

/* Tabs & buttons */
.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] { color: #D31B2C !important; border-bottom-color: #D31B2C !important; }
.stTabs [data-baseweb="tab-list"] button { color: #333333 !important; }
.stButton > button { background-color: #D31B2C !important; color: white !important; border: none !important; font-weight: 700 !important; border-radius: 4px !important; }
.stButton > button:hover { background-color: #B01624 !important; }

/* Inputs & widgets in main area */
[data-testid="stAppViewContainer"] input,
[data-testid="stAppViewContainer"] textarea,
[data-testid="stAppViewContainer"] select { background-color: #FFFFFF !important; color: #333333 !important; }
[data-testid="stAppViewContainer"] [data-baseweb="select"],
[data-testid="stAppViewContainer"] [data-baseweb="select"] *,
[data-testid="stAppViewContainer"] [data-baseweb="select"] div,
[data-testid="stAppViewContainer"] [data-baseweb="select"] span,
[data-testid="stAppViewContainer"] [data-baseweb="input"],
[data-testid="stAppViewContainer"] [data-baseweb="input"] * { background-color: #FFFFFF !important; color: #333333 !important; }
[data-testid="stAppViewContainer"] [data-baseweb="popover"],
[data-testid="stAppViewContainer"] [data-baseweb="popover"] *,
[data-testid="stAppViewContainer"] [role="listbox"],
[data-testid="stAppViewContainer"] [role="listbox"] *,
[data-testid="stAppViewContainer"] [role="option"],
[data-testid="stAppViewContainer"] [role="option"] * { background-color: #FFFFFF !important; color: #333333 !important; }
[data-testid="stAppViewContainer"] [role="option"]:hover { background-color: #FFF5F5 !important; }

/* Radio buttons & checkboxes in main area */
[data-testid="stAppViewContainer"] [role="radiogroup"] label,
[data-testid="stAppViewContainer"] [role="radiogroup"] span,
[data-testid="stAppViewContainer"] .stRadio label,
[data-testid="stAppViewContainer"] .stCheckbox label { color: #333333 !important; }

/* Text areas */
[data-testid="stAppViewContainer"] .stTextArea textarea { background-color: #FFFFFF !important; color: #333333 !important; border: 1px solid #ccc !important; }

/* Expanders */
[data-testid="stExpander"] { background-color: #FFFFFF !important; }
[data-testid="stExpander"] summary span { color: #333333 !important; }

hr { border-color: #D31B2C !important; }
</style>
"""
st.markdown(NEU_CSS, unsafe_allow_html=True)

# Palette constants
NEU_PALETTE = ["#D31B2C", "#0ECCDE", "#0C3354", "#A0E0EF", "#8C8080", "#E6D8D3"]

def banner(title, subtitle):
    st.markdown(
        f'<div style="background:linear-gradient(135deg,#D31B2C 0%,#B01624 100%);color:white;'
        f'padding:1.2rem 1.5rem;border-radius:6px;margin-bottom:1.5rem;">'
        f'<h2 style="color:white!important;margin:0;">{title}</h2>'
        f'<p style="color:#E6D8D3;margin:0.3rem 0 0 0;">{subtitle}</p></div>',
        unsafe_allow_html=True,
    )

def info_box(text):
    st.markdown(
        f'<div style="background-color:#F0FAFB;border-left:4px solid #0ECCDE;'
        f'padding:12px 16px;border-radius:4px;margin:8px 0;">{text}</div>',
        unsafe_allow_html=True,
    )


# ╔══════════════════════════════════════════════════════════════╗
# ║  DATA GENERATORS                                            ║
# ╚══════════════════════════════════════════════════════════════╝

def generate_dirty_data():
    np.random.seed(42)
    n = 600
    specialties = [
        "Cardiology","Neurology","Orthopedics","Gastroenterology","Pulmonology",
        "Nephrology","Oncology","Psychiatry","Dermatology","Ophthalmology",
        "Otolaryngology","Rheumatology","Endocrinology","Infectious Disease",
        "Hematology","Urology","Vascular Surgery","Thoracic Surgery","Pediatrics",
        "Obstetrics","Emergency Medicine","Internal Medicine","Family Medicine",
        "Pathology","Radiology","Anesthesiology","Physical Medicine","Immunology",
        "Hepatology","Toxicology",
    ]
    payer_codes = ["BCBS","CCBS","HMO","POS","PPO","IND","GOV","UNK","NONE"]
    age_groups = ["[0-10)","[10-20)","[20-30)","[30-40)","[40-50)",
                  "[50-60)","[60-70)","[70-80)","[80-90)","[90-100)"]
    return pd.DataFrame({
        "patient_id": [f"P{str(i).zfill(6)}" for i in range(1, n+1)],
        "age_group": np.random.choice(age_groups, n),
        "gender": np.random.choice(["M","F"], n),
        "lab_result": np.where(np.random.rand(n) < 0.18, "?",
                               np.round(np.random.normal(100, 25, n), 1).astype(str)),
        "medical_specialty": np.where(np.random.rand(n) < 0.49, "?",
                                      np.random.choice(specialties, n)),
        "payer_code": np.where(np.random.rand(n) < 0.40, "?",
                               np.random.choice(payer_codes, n)),
        "time_in_hospital": np.random.randint(1, 15, n),
        "num_lab_procedures": np.random.randint(0, 100, n),
        "num_medications": np.random.randint(0, 80, n),
        "number_emergency": np.random.exponential(2, n).astype(int),
        "readmitted": np.random.choice([">30","<30","NO"], n, p=[.25,.35,.40]),
    })


def generate_retail_data():
    np.random.seed(42)
    n = 800
    units = np.random.randint(1, 100, n)
    price = np.random.uniform(5, 500, n)
    disc = np.random.uniform(0, 0.3, n)
    return pd.DataFrame({
        "customer_id": [f"C{str(i).zfill(6)}" for i in range(1, n+1)],
        "region": np.random.choice(["North","South","East","West"], n),
        "product_category": np.random.choice(["Electronics","Clothing","Food","Home"], n),
        "units_sold": units,
        "unit_price": price,
        "discount_pct": disc,
        "customer_satisfaction": np.random.uniform(1, 5, n),
        "marketing_spend": np.random.uniform(0, 5000, n),
        "revenue": units * price * (1 - disc),
        "repeat_customer": np.random.choice([0,1], n, p=[.6,.4]),
    })


def generate_churn_data():
    np.random.seed(42)
    n = 1000
    contract = np.random.choice(["Month-to-month","One year","Two year"], n, p=[.40,.35,.25])
    internet = np.random.choice(["DSL","Fiber optic","No"], n, p=[.40,.35,.25])
    tenure = np.random.uniform(1, 72, n)
    charges = np.random.uniform(20, 120, n)
    total = tenure * charges + np.random.normal(0, 50, n)
    total = np.maximum(total, 0)
    security = np.random.choice(["Yes","No"], n, p=[.3,.7])
    support = np.random.choice(["Yes","No"], n, p=[.25,.75])
    payment = np.random.choice(["Electronic check","Mailed check","Bank transfer","Credit card"],
                               n, p=[.30,.25,.22,.23])
    tickets = np.minimum(np.random.poisson(2, n), 9)
    # churn probability
    p = np.zeros(n)
    p += 0.50*(contract=="Month-to-month") + 0.25*(contract=="One year")
    p += 0.15*(internet=="Fiber optic") + 0.10*(charges>80)
    p -= 0.10*(tenure>24) + 0.08*(security=="Yes") + 0.08*(support=="Yes")
    p += 0.05*(payment=="Electronic check") + 0.02*tickets/10
    p = np.clip(p, 0, 1)
    p = p * 0.27 / p.mean()
    p = np.clip(p, 0, 1)
    churned = (np.random.random(n) < p).astype(int)
    return pd.DataFrame({
        "customer_id": [f"CUST_{i:05d}" for i in range(n)],
        "tenure": tenure, "monthly_charges": charges,
        "total_charges": total, "contract_type": contract,
        "internet_service": internet, "online_security": security,
        "tech_support": support, "payment_method": payment,
        "num_support_tickets": tickets, "churned": churned,
    })


def generate_reviews():
    np.random.seed(42)
    n = 400
    cats = ["Electronics","Kitchen","Books","Clothing","Sports"]
    ratings = np.random.choice([1,2,3,4,5], n, p=[.12,.10,.15,.25,.38])

    pos = ["excellent quality","great value","very satisfied","highly recommend",
           "exceeded expectations","fantastic product","impressive durability",
           "arrived quickly","great shipping","exactly as described","perfect fit",
           "absolutely love it","best purchase","outstanding","amazing quality",
           "worth every penny","fantastic deal","super happy","will buy again",
           "five star experience","love the design","very impressed"]
    neg = ["terrible quality","waste of money","broke immediately","very disappointed",
           "do not recommend","poor quality","defective","stopped working",
           "cheap material","not as described","false advertising","complete disappointment",
           "never again","absolute garbage","worst purchase","broken on arrival",
           "useless product","overpriced junk","regret buying","save your money",
           "totally unsatisfied","bad experience"]
    starters = ["This item","The product","It","This","My purchase"]
    connectors = ["The","Overall","However","But","What I like is","The problem is"]

    def _review(r):
        if r == 5:
            ph = np.random.choice(pos, 2, replace=False)
            return (f"{np.random.choice(starters)} is {ph[0]}. "
                    f"I {np.random.choice(['highly recommend','absolutely recommend','definitely recommend'])} this to anyone. "
                    f"{np.random.choice(connectors)} {ph[1]} makes it a great buy.")
        elif r == 1:
            ph = np.random.choice(neg, 2, replace=False)
            return (f"{np.random.choice(starters)} is {ph[0]}. "
                    f"Total {np.random.choice(['waste of money','disappointment','disaster'])}. "
                    f"{np.random.choice(connectors)} {ph[1]} - avoid this product.")
        elif r == 4:
            return (f"{np.random.choice(starters)} is {np.random.choice(pos)}. "
                    f"Works as expected and great {np.random.choice(['quality','value','shipping'])}. "
                    f"Would definitely buy again without hesitation.")
        elif r == 2:
            return (f"{np.random.choice(starters)} has some issues. "
                    f"While not terrible, it is definitely {np.random.choice(neg)}. "
                    f"Probably not worth the {np.random.choice(['price','money','cost'])}.")
        else:
            return (f"{np.random.choice(starters)} is okay. "
                    f"It has {np.random.choice(pos)} aspects, but also feels {np.random.choice(neg)}. "
                    f"Average product for the price. Acceptable but not great.")

    rows = []
    for i in range(n):
        rows.append({
            "review_id": f"REV_{i:05d}",
            "product_category": np.random.choice(cats),
            "rating": ratings[i],
            "review_text": _review(ratings[i]),
        })
    return pd.DataFrame(rows)


# ╔══════════════════════════════════════════════════════════════╗
# ║  WEEK 1 — DATA CLEANING EXPLORER                           ║
# ╚══════════════════════════════════════════════════════════════╝

def week1_cleaning_lab():
    banner("Week 1: Data Cleaning Explorer",
           "See how different cleaning strategies change your data")

    raw = generate_dirty_data()

    # --- Profile ---
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", raw.shape[0])
    c2.metric("Columns", raw.shape[1])
    q_total = int((raw == "?").sum().sum())
    c3.metric("Total '?' Values", f"{q_total:,}")

    with st.expander("Missing value breakdown"):
        for col in raw.columns:
            cnt = int((raw[col] == "?").sum())
            if cnt:
                pct = cnt / len(raw) * 100
                st.text(f"  {col:25s}  {cnt:>4d}  ({pct:.1f}%)")

    # --- Helper to apply a cleaning strategy ---
    def _fill_categorical(series, col_name, strategy, full_df):
        valid = series[series != "?"]
        mask = series == "?"
        if not mask.any():
            return series
        if strategy == "Fill with median (mode)":
            fill = valid.mode()[0] if len(valid) else "Unknown"
            return series.replace("?", fill)
        elif strategy == "Fill with mean (freq-weighted)":
            freqs = valid.value_counts(normalize=True)
            out = series.copy()
            np.random.seed(42)
            out.loc[mask] = np.random.choice(freqs.index, size=mask.sum(), p=freqs.values)
            return out
        elif strategy == "Fill by group (age_group)":
            out = series.copy()
            for grp in full_df["age_group"].unique():
                grp_valid = full_df.loc[(full_df["age_group"] == grp) & (full_df[col_name] != "?"), col_name]
                fill = grp_valid.mode()[0] if len(grp_valid) and len(grp_valid.mode()) else "Unknown"
                grp_mask = (full_df["age_group"] == grp) & (out == "?")
                out.loc[grp_mask] = fill
            return out
        return series

    # --- Variable config: only the three with missing data ---
    dirty_vars = {
        "lab_result": {
            "pct": "18 %",
            "dtype": "numeric",
            "options": ["Drop column", "Drop rows with missing", "Fill with median",
                        "Fill with mean", "Fill by group mean (age_group)"],
        },
        "medical_specialty": {
            "pct": "49 %",
            "dtype": "categorical",
            "options": ["Drop column", "Drop rows with missing", "Fill with median (mode)",
                        "Fill with mean (freq-weighted)", "Fill by group (age_group)"],
        },
        "payer_code": {
            "pct": "40 %",
            "dtype": "categorical",
            "options": ["Drop column", "Drop rows with missing", "Fill with median (mode)",
                        "Fill with mean (freq-weighted)", "Fill by group (age_group)"],
        },
    }

    # --- Step 1: Select variable ---
    st.markdown("---")
    st.markdown("### Step 1 — Select a variable with missing data")
    sel_var = st.selectbox(
        "Variable",
        list(dirty_vars.keys()),
        format_func=lambda v: f"{v}  ({dirty_vars[v]['pct']} missing)",
        key="w1_var",
    )

    # --- Step 2: Show current distribution ---
    st.markdown("### Step 2 — Current distribution (before cleaning)")
    raw_series = raw[sel_var]
    is_numeric_var = dirty_vars[sel_var]["dtype"] == "numeric"
    missing_ct = int((raw_series == "?").sum())
    valid_series = raw_series[raw_series != "?"]

    if is_numeric_var:
        valid_numeric = pd.to_numeric(valid_series, errors="coerce").dropna()
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor("white")
        ax.hist(valid_numeric, bins=20, color="#D31B2C", alpha=.7, edgecolor="black")
        ax.axvline(valid_numeric.median(), color="#0C3354", ls="--", lw=2, label=f"Median: {valid_numeric.median():.1f}")
        ax.axvline(valid_numeric.mean(), color="#0ECCDE", ls="--", lw=2, label=f"Mean: {valid_numeric.mean():.1f}")
        ax.legend()
        ax.set_title(f"{sel_var} — Known Values Only ({len(valid_numeric)} of {len(raw_series)})", fontweight="bold")
        ax.set_xlabel(sel_var); ax.set_ylabel("Count"); ax.grid(axis="y", alpha=.3)
        plt.tight_layout(); st.pyplot(fig)
    else:
        vc_raw = raw_series.value_counts().head(15)
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor("white")
        colors = ["#8C8080" if v == "?" else "#D31B2C" for v in vc_raw.index]
        ax.barh(range(len(vc_raw)), vc_raw.values, color=colors)
        ax.set_yticks(range(len(vc_raw)))
        ax.set_yticklabels(vc_raw.index)
        ax.invert_yaxis()
        ax.set_title(f"{sel_var} — Raw Distribution", fontweight="bold")
        ax.set_xlabel("Count")
        plt.tight_layout(); st.pyplot(fig)

    if is_numeric_var:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total values", len(raw_series))
        m2.metric("Missing ('?')", f"{missing_ct}  ({missing_ct/len(raw_series)*100:.1f}%)")
        m3.metric("Mean (known)", f"{valid_numeric.mean():.1f}")
        m4.metric("Median (known)", f"{valid_numeric.median():.1f}")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Total values", len(raw_series))
        m2.metric("Missing ('?')", f"{missing_ct}  ({missing_ct/len(raw_series)*100:.1f}%)")
        m3.metric("Unique (non-missing)", len(valid_series.unique()))

    # --- Step 3: Choose cleaning method ---
    st.markdown("### Step 3 — Choose a cleaning strategy")
    strategy = st.radio(
        "How should we handle the missing values?",
        dirty_vars[sel_var]["options"],
        key=f"w1_strat_{sel_var}",
        horizontal=True,
    )

    # --- Method explanation ---
    method_explanations = {
        "Drop column": (
            "Removes the entire column from the dataset. "
            "Use when the variable has too much missing data to be useful, or when it "
            "doesn't carry meaningful predictive signal. The upside is simplicity — no "
            "assumptions are introduced. The downside is complete loss of whatever information "
            "the column did contain."
        ),
        "Drop rows with missing": (
            "Deletes every row where this variable is missing. This is also called "
            "<b>listwise deletion</b> or <b>complete-case analysis</b>. It's valid when "
            "data is <b>Missing Completely At Random (MCAR)</b> — meaning the reason a value "
            "is missing has nothing to do with the value itself or any other variable. "
            "If missingness is related to the outcome (e.g., sicker patients skip labs), "
            "this method introduces <b>selection bias</b>."
        ),
        "Fill with median": (
            "Replaces every missing value with the <b>median</b> of the known values. "
            "The median is the middle value when sorted — it splits the data into two equal "
            "halves. It is <b>robust to outliers</b> because extreme values don't pull it. "
            "However, filling with a single value creates an artificial spike in the "
            "distribution and <b>reduces variance</b>, which can weaken model performance."
        ),
        "Fill with mean": (
            "Replaces every missing value with the <b>arithmetic mean</b> (average) of the "
            "known values. The mean preserves the overall center of the distribution but is "
            "<b>sensitive to outliers</b> — a few extreme values can shift it significantly. "
            "Like median fill, it creates an artificial spike and <b>underestimates variance</b>."
        ),
        "Fill by group mean (age_group)": (
            "Instead of one global fill value, this computes the <b>mean within each age group</b> "
            "and fills missing values with their group's mean. This is a form of "
            "<b>conditional imputation</b> — it assumes the variable's typical value depends "
            "on the group. It preserves <b>between-group differences</b> and produces a more "
            "realistic distribution than a single global fill, but it still reduces "
            "within-group variance."
        ),
        "Fill with median (mode)": (
            "For categorical variables, filling with the <b>median</b> means using the "
            "<b>most frequently occurring category</b>. Every missing value is replaced with "
            "this single most common value. This is simple and deterministic, but it "
            "<b>inflates that category's count</b> and can make the distribution look more "
            "skewed than it actually is. Best used when one category genuinely dominates."
        ),
        "Fill with mean (freq-weighted)": (
            "This is <b>frequency-weighted random imputation</b>. Instead of filling with one "
            "value, each missing entry is randomly assigned a category, where the probability "
            "of each category equals its <b>observed frequency</b> in the non-missing data. "
            "For example, if 30% of known values are 'Cardiology' and 20% are 'Neurology', "
            "missing values have a 30% chance of becoming 'Cardiology' and 20% 'Neurology'. "
            "This <b>preserves the overall distribution shape</b> but introduces randomness — "
            "running it twice gives different results."
        ),
        "Fill by group (age_group)": (
            "Fills each missing value with the <b>most frequent category within its "
            "age group</b>. This is <b>conditional imputation</b> — it assumes the most "
            "likely category depends on the patient's age. For example, older patients may "
            "more commonly see Cardiology, while younger patients see Orthopedics. This "
            "preserves <b>group-specific patterns</b> but can over-represent the dominant "
            "category within each group."
        ),
    }

    if strategy in method_explanations:
        st.markdown(
            f'<div style="background-color:#F0FAFB;border-left:4px solid #0C3354;'
            f'padding:14px 18px;border-radius:4px;margin:12px 0;">'
            f'<b style="color:#0C3354;">How it works — {strategy}</b><br><br>'
            f'{method_explanations[strategy]}</div>',
            unsafe_allow_html=True,
        )

    # --- Step 4: Show result ---
    st.markdown("### Step 4 — Result after cleaning")
    cleaned = raw[sel_var].copy()

    if strategy == "Drop column":
        st.info(f"**{sel_var}** would be removed from the dataset entirely.")
        st.markdown(
            f"All {missing_ct} missing values are eliminated, but you lose any signal the "
            f"column contained."
        )

    elif strategy == "Drop rows with missing":
        rows_kept = len(raw) - missing_ct
        pct_lost = missing_ct / len(raw) * 100
        st.warning(f"**{missing_ct} rows** ({pct_lost:.1f}%) would be dropped. "
                   f"**{rows_kept} rows** remain.")

        m1, m2 = st.columns(2)
        m1.metric("Rows before", len(raw))
        m2.metric("Rows after", rows_kept, delta=f"-{missing_ct}")

        fig, ax = plt.subplots(figsize=(6, 3))
        fig.patch.set_facecolor("white")
        ax.barh(["Kept", "Dropped"], [rows_kept, missing_ct],
                color=["#0ECCDE", "#8C8080"], edgecolor="black")
        for i, v in enumerate([rows_kept, missing_ct]):
            ax.text(v, i, f"  {v} ({v/len(raw)*100:.1f}%)", va="center", fontweight="bold")
        ax.set_title(f"Row Impact — {sel_var}", fontweight="bold")
        ax.set_xlabel("Rows"); plt.tight_layout(); st.pyplot(fig)

        if pct_lost > 30:
            info_box(f"<b>Caution:</b> Dropping {pct_lost:.0f}% of your data is significant. "
                     f"This reduces statistical power and may introduce bias if missingness "
                     f"is not completely random (MCAR).")
        else:
            info_box(f"Dropping {pct_lost:.0f}% of rows is moderate. This is safe if the data "
                     f"is <b>missing completely at random</b> (MCAR), but check whether the "
                     f"dropped rows differ systematically from the kept rows.")

    elif is_numeric_var:
        # Numeric imputation for weight
        numeric_raw = pd.to_numeric(cleaned.replace("?", np.nan), errors="coerce")
        known = numeric_raw.dropna()

        if strategy == "Fill with median":
            fill_val = known.median()
            numeric_clean = numeric_raw.fillna(fill_val)
            explain = (f"All {missing_ct} missing values filled with the <b>median</b>: "
                       f"<b>{fill_val:.1f}</b>. Creates a spike at this value but is robust "
                       f"to outliers.")
        elif strategy == "Fill with mean":
            fill_val = known.mean()
            numeric_clean = numeric_raw.fillna(fill_val)
            explain = (f"All {missing_ct} missing values filled with the <b>mean</b>: "
                       f"<b>{fill_val:.1f}</b>. Creates a spike at this value and is sensitive "
                       f"to outliers.")
        elif strategy == "Fill by group mean (age_group)":
            numeric_clean = numeric_raw.copy()
            group_fills = {}
            for grp in raw["age_group"].unique():
                grp_vals = numeric_raw[raw["age_group"] == grp].dropna()
                gm = grp_vals.mean() if len(grp_vals) else known.mean()
                group_fills[grp] = gm
                grp_mask = (raw["age_group"] == grp) & numeric_raw.isna()
                numeric_clean.loc[grp_mask] = gm
            explain = (f"Missing values filled with the <b>mean within each age group</b>. "
                       f"This captures group-specific patterns (e.g. younger patients weigh less).")
        else:
            numeric_clean = numeric_raw

        # Side-by-side histograms
        bef_col, aft_col = st.columns(2)
        with bef_col:
            st.markdown("#### Before")
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor("white")
            ax.hist(known, bins=20, color="#D31B2C", alpha=.7, edgecolor="black")
            ax.set_title(f"{sel_var} — Before ({len(known)} known)", fontweight="bold")
            ax.set_xlabel(sel_var); ax.set_ylabel("Count"); ax.grid(axis="y", alpha=.3)
            plt.tight_layout(); st.pyplot(fig)

        with aft_col:
            st.markdown("#### After")
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor("white")
            ax.hist(numeric_clean, bins=20, color="#0ECCDE", alpha=.7, edgecolor="black")
            ax.set_title(f"{sel_var} — After ({len(numeric_clean)} total)", fontweight="bold")
            ax.set_xlabel(sel_var); ax.set_ylabel("Count"); ax.grid(axis="y", alpha=.3)
            plt.tight_layout(); st.pyplot(fig)

        # Stats comparison — before vs after side by side
        st.markdown("#### Statistical Impact")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Mean before", f"{known.mean():.1f}")
        s2.metric("Mean after", f"{numeric_clean.mean():.1f}",
                  delta=f"{numeric_clean.mean() - known.mean():+.2f}")
        s3.metric("Median before", f"{known.median():.1f}")
        s4.metric("Median after", f"{numeric_clean.median():.1f}",
                  delta=f"{numeric_clean.median() - known.median():+.2f}")

        s5, s6, s7, s8 = st.columns(4)
        s5.metric("Std Dev before", f"{known.std():.1f}")
        s6.metric("Std Dev after", f"{numeric_clean.std():.1f}",
                  delta=f"{numeric_clean.std() - known.std():+.2f}")

        # Outlier detection (IQR method)
        q1_b, q3_b = known.quantile(.25), known.quantile(.75)
        iqr_b = q3_b - q1_b
        out_b = int(((known < q1_b - 1.5*iqr_b) | (known > q3_b + 1.5*iqr_b)).sum())

        q1_a, q3_a = numeric_clean.quantile(.25), numeric_clean.quantile(.75)
        iqr_a = q3_a - q1_a
        out_a = int(((numeric_clean < q1_a - 1.5*iqr_a) | (numeric_clean > q3_a + 1.5*iqr_a)).sum())

        s7.metric("Outliers before", out_b)
        s8.metric("Outliers after", out_a,
                  delta=f"{out_a - out_b:+d}" if out_a != out_b else "no change")

        # Box plot comparison
        fig, axes = plt.subplots(1, 2, figsize=(10, 3), sharey=True)
        fig.patch.set_facecolor("white")
        bp1 = axes[0].boxplot(known.values, vert=False, patch_artist=True,
                              showmeans=True, meanprops=dict(marker="D", markerfacecolor="#0ECCDE"))
        bp1["boxes"][0].set_facecolor("#D31B2C"); bp1["boxes"][0].set_alpha(.5)
        axes[0].set_title("Before", fontweight="bold"); axes[0].set_xlabel(sel_var)
        bp2 = axes[1].boxplot(numeric_clean.values, vert=False, patch_artist=True,
                              showmeans=True, meanprops=dict(marker="D", markerfacecolor="#D31B2C"))
        bp2["boxes"][0].set_facecolor("#0ECCDE"); bp2["boxes"][0].set_alpha(.5)
        axes[1].set_title("After", fontweight="bold"); axes[1].set_xlabel(sel_var)
        plt.tight_layout(); st.pyplot(fig)

        if strategy == "Fill by group mean (age_group)":
            with st.expander("Group means used"):
                for grp in sorted(group_fills.keys()):
                    st.text(f"  {grp:12s}  →  {group_fills[grp]:.1f}")

        info_box(explain)

    else:
        # Categorical imputation for medical_specialty / payer_code
        cleaned = _fill_categorical(cleaned, sel_var, strategy, raw)

        bef_col, aft_col = st.columns(2)
        with bef_col:
            st.markdown("#### Before")
            vc_b = raw_series.value_counts().head(12)
            fig, ax = plt.subplots(figsize=(6, 5))
            fig.patch.set_facecolor("white")
            colors_b = ["#8C8080" if v == "?" else "#D31B2C" for v in vc_b.index]
            ax.barh(range(len(vc_b)), vc_b.values, color=colors_b)
            ax.set_yticks(range(len(vc_b))); ax.set_yticklabels(vc_b.index)
            ax.invert_yaxis()
            ax.set_xlabel("Count"); ax.set_title("Before", fontweight="bold")
            plt.tight_layout(); st.pyplot(fig)

        with aft_col:
            st.markdown("#### After")
            vc_a = cleaned.value_counts().head(12)
            fig, ax = plt.subplots(figsize=(6, 5))
            fig.patch.set_facecolor("white")
            ax.barh(range(len(vc_a)), vc_a.values, color="#0ECCDE")
            ax.set_yticks(range(len(vc_a))); ax.set_yticklabels(vc_a.index)
            ax.invert_yaxis()
            ax.set_xlabel("Count"); ax.set_title("After", fontweight="bold")
            plt.tight_layout(); st.pyplot(fig)

        if strategy == "Fill with median (mode)":
            fill_val = valid_series.value_counts().index[0] if len(valid_series) else "N/A"
            info_box(f"All {missing_ct} missing values filled with the <b>most frequent "
                     f"category</b>: <b>{fill_val}</b>. This is simple but inflates that "
                     f"category's count and reduces variance.")
        elif strategy == "Fill with mean (freq-weighted)":
            info_box(f"Missing values filled by <b>random sampling</b> weighted by existing "
                     f"category frequencies. This preserves the overall distribution shape "
                     f"but introduces randomness.")
        elif strategy == "Fill by group (age_group)":
            info_box(f"Missing values filled with the <b>most frequent category within each "
                     f"age group</b>. This captures group-specific patterns — different age "
                     f"groups may have different dominant values.")

    # --- Step 5: Recommendation prompt ---
    st.markdown("---")
    st.markdown("### Your Recommendation")
    st.markdown(
        f"Based on what you've seen for **{sel_var}**, which strategy would you choose "
        f"and why? Consider:"
    )
    st.markdown(
        "- How much data is missing?\n"
        "- Does the variable carry useful signal?\n"
        "- Does the cleaning method distort the distribution?\n"
        "- Would the missingness itself be informative?"
    )
    st.text_area(
        f"Your recommendation for {sel_var}:",
        placeholder=f"I would choose … for {sel_var} because …",
        key=f"w1_rec_{sel_var}",
    )

    with st.expander("Key Takeaway"):
        info_box(
            "<b>Cleaning choices shape conclusions.</b> Dropping a column removes information. "
            "Imputation introduces assumptions. Binary indicators preserve the signal that data is "
            "missing — which itself can be predictive. There is no single right answer, but every "
            "choice must be <i>justified</i>."
        )


# ╔══════════════════════════════════════════════════════════════╗
# ║  WEEK 2 — BUSINESS INSIGHTS & KPIs                         ║
# ╚══════════════════════════════════════════════════════════════╝

def week2_business_insights():
    banner("Week 2: Business Insights & KPIs",
           "Discover patterns through correlation and regression")

    data = generate_retail_data()
    tab1, tab2, tab3 = st.tabs(["📊 KPI Dashboard", "🔗 Correlation Analysis", "📈 Regression Explorer"])

    # --- KPI Dashboard ---
    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Revenue", f"${data['revenue'].sum():,.0f}")
        c2.metric("Avg Satisfaction", f"{data['customer_satisfaction'].mean():.2f} / 5")
        c3.metric("Repeat Rate", f"{data['repeat_customer'].mean()*100:.1f}%")
        top_cat = data.groupby("product_category")["revenue"].sum().idxmax()
        c4.metric("Top Category", top_cat)

        left, right = st.columns(2)
        for col_block, group_col, title in [
            (left, "region", "Revenue by Region"),
            (right, "product_category", "Revenue by Category"),
        ]:
            with col_block:
                agg = data.groupby(group_col)["revenue"].sum().sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(7, 4))
                fig.patch.set_facecolor("white")
                ax.bar(agg.index, agg.values, color=NEU_PALETTE[:len(agg)], edgecolor="black")
                ax.set_title(title, fontweight="bold")
                ax.set_ylabel("Revenue ($)")
                for i, v in enumerate(agg.values):
                    ax.text(i, v, f"${v:,.0f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
                ax.grid(axis="y", alpha=.3)
                plt.tight_layout()
                st.pyplot(fig)

    # --- Correlation Analysis ---
    with tab2:
        num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        selected = st.multiselect("Columns to include", num_cols, default=num_cols, key="w2_corr")
        if len(selected) > 1:
            corr = data[selected].corr()
            fig, ax = plt.subplots(figsize=(9, 7))
            fig.patch.set_facecolor("white")
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                        vmin=-1, vmax=1, ax=ax)
            ax.set_title("Correlation Heatmap", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig)

            strong = []
            for i in range(len(corr)):
                for j in range(i+1, len(corr)):
                    r = corr.iloc[i, j]
                    if abs(r) > .5:
                        strong.append({"Var 1": corr.columns[i], "Var 2": corr.columns[j],
                                       "r": f"{r:.3f}"})
            if strong:
                st.markdown("#### Strong correlations (|r| > 0.5)")
                st.dataframe(pd.DataFrame(strong), use_container_width=True)
            else:
                st.info("No pairs with |r| > 0.5 in selected columns.")

    # --- Regression Explorer ---
    with tab3:
        num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        lc, rc = st.columns(2)
        x_var = lc.selectbox("X (predictor)", num_cols, key="w2_x")
        y_var = rc.selectbox("Y (response)", num_cols, index=min(1, len(num_cols)-1), key="w2_y")

        if x_var != y_var:
            x, y = data[x_var].values, data[y_var].values
            slope, intercept, r, p, se = scipy.stats.linregress(x, y)
            r2 = r**2

            fig, ax = plt.subplots(figsize=(9, 5))
            fig.patch.set_facecolor("white")
            ax.scatter(x, y, color="#0ECCDE", alpha=.5, s=30, edgecolor="black", linewidth=.3)
            xl = np.array([x.min(), x.max()])
            ax.plot(xl, intercept + slope*xl, color="#D31B2C", linewidth=3,
                    label=f"OLS (R²={r2:.3f})")
            ax.set_xlabel(x_var, fontweight="bold")
            ax.set_ylabel(y_var, fontweight="bold")
            ax.set_title(f"{y_var} vs {x_var}", fontweight="bold")
            ax.legend()
            ax.grid(alpha=.3)
            plt.tight_layout()
            st.pyplot(fig)

            m1, m2, m3 = st.columns(3)
            m1.metric("R²", f"{r2:.4f}")
            m2.metric("Slope", f"{slope:.4f}")
            m3.metric("p-value", f"{p:.2e}")

            if p < .05:
                info_box(f"<b>Statistically significant</b> (p = {p:.2e}). Each unit increase in "
                         f"{x_var} is associated with a {slope:.4f} change in {y_var}.")
            else:
                info_box(f"<b>Not significant</b> (p = {p:.2e}). We can't confidently say "
                         f"{x_var} predicts {y_var}.")
        else:
            st.warning("Pick different variables for X and Y.")


# ╔══════════════════════════════════════════════════════════════╗
# ║  WEEK 3 — ML MODEL COMPARISON LAB                          ║
# ╚══════════════════════════════════════════════════════════════╝

def week3_ml_comparison():
    banner("Week 3: ML Model Comparison Lab",
           "Decision Trees vs Random Forests — see the difference")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 Model Configuration")
    test_size = st.sidebar.slider("Test size", .10, .40, .25, .01, key="w3_ts")
    dt_depth  = st.sidebar.slider("DT max depth", 1, 20, 5, key="w3_dtd")
    rf_trees  = st.sidebar.slider("RF n_estimators", 10, 200, 100, 10, key="w3_rft")
    rf_depth  = st.sidebar.slider("RF max depth", 1, 20, 5, key="w3_rfd")

    df = generate_churn_data()
    enc = pd.get_dummies(df, columns=["contract_type","internet_service",
                                      "online_security","tech_support","payment_method"],
                         drop_first=True)
    X = enc.drop(["customer_id","churned"], axis=1)
    y = enc["churned"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42)

    dt = DecisionTreeClassifier(max_depth=dt_depth, random_state=42)
    dt.fit(X_tr, y_tr); dt_pred = dt.predict(X_te)

    rf = RandomForestClassifier(n_estimators=rf_trees, max_depth=rf_depth, random_state=42)
    rf.fit(X_tr, y_tr); rf_pred = rf.predict(X_te)

    def _metrics(y_true, y_pred):
        return {
            "Accuracy":  accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall":    recall_score(y_true, y_pred, zero_division=0),
            "F1":        f1_score(y_true, y_pred, zero_division=0),
        }
    dt_m = _metrics(y_te, dt_pred)
    rf_m = _metrics(y_te, rf_pred)

    t1, t2, t3 = st.tabs(["📋 Metrics", "🔲 Confusion Matrices", "📊 Feature Importance"])

    with t1:
        left, right = st.columns(2)
        with left:
            st.subheader("Decision Tree")
            for k in dt_m:
                st.metric(k, f"{dt_m[k]:.3f}", delta=f"{dt_m[k]-rf_m[k]:+.3f}")
        with right:
            st.subheader("Random Forest")
            for k in rf_m:
                st.metric(k, f"{rf_m[k]:.3f}")

        with st.expander("What do these metrics mean?"):
            st.markdown(
                "**Accuracy** — overall correctness.  \n"
                "**Precision** — of predicted churners, how many actually churned?  \n"
                "**Recall** — of actual churners, how many did we catch?  \n"
                "**F1** — harmonic mean of precision & recall."
            )

    with t2:
        left, right = st.columns(2)
        for col_block, pred, name in [(left, dt_pred, "Decision Tree"), (right, rf_pred, "Random Forest")]:
            with col_block:
                cm = confusion_matrix(y_te, pred)
                fig, ax = plt.subplots(figsize=(5, 4))
                fig.patch.set_facecolor("white")
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                            xticklabels=["Retained","Churned"],
                            yticklabels=["Retained","Churned"], ax=ax)
                ax.set_title(name, fontweight="bold")
                ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
                plt.tight_layout()
                st.pyplot(fig)
                tn, fp, fn, tp = cm.ravel()
                tpr = tp/(tp+fn) if (tp+fn) else 0
                fpr = fp/(fp+tn) if (fp+tn) else 0
                st.text(f"TPR (Sensitivity): {tpr:.3f}")
                st.text(f"FPR:               {fpr:.3f}")

    with t3:
        top_idx = np.argsort(rf.feature_importances_)[-10:]
        feats = X.columns[top_idx]
        imp_df = pd.DataFrame({
            "Feature": feats,
            "Decision Tree": dt.feature_importances_[top_idx],
            "Random Forest": rf.feature_importances_[top_idx],
        }).sort_values("Random Forest", ascending=True)

        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor("white")
        y_pos = np.arange(len(imp_df))
        ax.barh(y_pos - .2, imp_df["Decision Tree"], .4, label="Decision Tree", color="#D31B2C")
        ax.barh(y_pos + .2, imp_df["Random Forest"], .4, label="Random Forest", color="#0ECCDE")
        ax.set_yticks(y_pos); ax.set_yticklabels(imp_df["Feature"])
        ax.set_xlabel("Importance"); ax.set_title("Top 10 Feature Importance", fontweight="bold")
        ax.legend(loc="lower right")
        plt.tight_layout()
        st.pyplot(fig)

        info_box("<b>Key Insight:</b> Random Forests average importance across many trees, "
                 "producing more stable rankings. A single Decision Tree may over-index on one split.")


# ╔══════════════════════════════════════════════════════════════╗
# ║  WEEK 4 — TOPIC MODELING EXPLORER                          ║
# ╚══════════════════════════════════════════════════════════════╝

def week4_topic_modeling():
    banner("Week 4: Topic Modeling Explorer",
           "Uncover hidden themes in customer reviews")

    info_box("<b>Welcome, Detective.</b> Your mission: investigate the hidden themes in "
             "customer reviews. Adjust the parameters and discover what customers are "
             "really talking about.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Investigation Settings")
    n_topics    = st.sidebar.slider("Number of topics", 2, 8, 4, key="w4_nt")
    max_feat    = st.sidebar.slider("Vocabulary size", 100, 2000, 500, 100, key="w4_mf")
    min_df      = st.sidebar.slider("Min document frequency", 1, 10, 2, key="w4_md")
    cat_filter  = st.sidebar.multiselect("Product categories",
                                         ["Electronics","Kitchen","Books","Clothing","Sports"], key="w4_cf")
    rating_rng  = st.sidebar.slider("Rating range", 1, 5, (1, 5), key="w4_rr")

    df = generate_reviews()
    if cat_filter:
        df = df[df["product_category"].isin(cat_filter)]
    df = df[(df["rating"] >= rating_rng[0]) & (df["rating"] <= rating_rng[1])]

    if len(df) < 10:
        st.warning("Too few reviews match your filters. Broaden your criteria.")
        return

    vec = CountVectorizer(max_features=max_feat, min_df=min_df, stop_words="english")
    dtm = vec.fit_transform(df["review_text"])
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=10)
    lda.fit(dtm)
    topic_dist = lda.transform(dtm)
    df["dominant_topic"] = topic_dist.argmax(axis=1)
    df["topic_prob"] = topic_dist.max(axis=1)
    words = vec.get_feature_names_out()
    colors = [NEU_PALETTE[i % len(NEU_PALETTE)] for i in range(n_topics)]

    t1, t2, t3 = st.tabs(["🔎 Topic Discovery", "📖 Review Explorer", "🕵️ Deep Dive"])

    with t1:
        for tid in range(n_topics):
            idx = lda.components_[tid].argsort()[-10:][::-1]
            tw = words[idx]; tw_w = lda.components_[tid][idx]
            fig, ax = plt.subplots(figsize=(8, 3))
            fig.patch.set_facecolor("white")
            ax.barh(range(len(tw)), tw_w, color=colors[tid])
            ax.set_yticks(range(len(tw))); ax.set_yticklabels(tw)
            ax.invert_yaxis()
            ax.set_title(f"Topic {tid} — Top Words", fontweight="bold")
            ax.set_xlabel("Weight")
            plt.tight_layout()
            st.pyplot(fig)

        # Pie chart
        tc = df["dominant_topic"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor("white")
        ax.pie(tc.values, labels=[f"Topic {i}" for i in tc.index],
               colors=[colors[i] for i in tc.index], autopct="%1.1f%%", startangle=90)
        ax.set_title("Review Distribution by Topic", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)

    with t2:
        sel = st.selectbox("Filter by topic", range(n_topics),
                           format_func=lambda x: f"Topic {x}", key="w4_te")
        subset = df[df["dominant_topic"] == sel].sort_values("topic_prob", ascending=False)
        st.metric("Reviews in topic", len(subset))
        st.metric("Avg rating", f"{subset['rating'].mean():.2f}")
        st.dataframe(subset[["review_id","rating","product_category","topic_prob"]].head(15),
                     use_container_width=True)

    with t3:
        sel2 = st.selectbox("Select topic", range(n_topics),
                            format_func=lambda x: f"Topic {x}", key="w4_dd")
        idx = lda.components_[sel2].argsort()[-15:][::-1]
        tw = words[idx]; tw_w = lda.components_[sel2][idx]
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor("white")
        ax.barh(range(len(tw)), tw_w, color=colors[sel2])
        ax.set_yticks(range(len(tw))); ax.set_yticklabels(tw)
        ax.invert_yaxis()
        ax.set_title(f"Topic {sel2} — Top 15 Words", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)

        kw = ", ".join(tw[:5])
        avg_r = df[df["dominant_topic"]==sel2]["rating"].mean()
        info_box(f"<b>Detective's Notebook:</b> Top keywords are <i>{kw}</i>. "
                 f"Average rating for this topic is <b>{avg_r:.2f}</b>/5. "
                 f"{'This looks like a satisfaction cluster.' if avg_r > 3.5 else 'This cluster may indicate quality concerns.'}")

        st.markdown("#### Sample Reviews")
        for _, row in df[df["dominant_topic"]==sel2].head(5).iterrows():
            st.markdown(f"{'⭐'*row['rating']} — *{row['product_category']}*")
            st.caption(row["review_text"])
            st.markdown("---")

        # Rating comparison
        l, r = st.columns(2)
        for col_block, subset_df, title, color in [
            (l, df[df["dominant_topic"]==sel2], f"Topic {sel2}", colors[sel2]),
            (r, df, "Overall", "#0C3354"),
        ]:
            with col_block:
                rd = subset_df["rating"].value_counts().sort_index()
                fig, ax = plt.subplots(figsize=(5, 3.5))
                fig.patch.set_facecolor("white")
                ax.bar(rd.index, rd.values, color=color)
                ax.set_xticks([1,2,3,4,5]); ax.set_xlabel("Rating"); ax.set_ylabel("Count")
                ax.set_title(title, fontweight="bold")
                plt.tight_layout()
                st.pyplot(fig)


# ╔══════════════════════════════════════════════════════════════╗
# ║  WEEK 5 — CASE STUDY: BUILDING A DATA PRODUCT              ║
# ╚══════════════════════════════════════════════════════════════╝

def week5_case_study():
    banner("Week 5: Building a Data Product",
           "From raw data to deployed prediction tool — a guided case study")

    # --- Session state ---
    if "cs_stage" not in st.session_state:
        st.session_state.cs_stage = 1

    # --- Generate data once ---
    if "cs_data" not in st.session_state:
        np.random.seed(42)
        n = 1000
        tenure = np.random.uniform(1, 72, n)
        charges = np.random.uniform(20, 120, n)
        total = charges * tenure * np.random.uniform(.9, 1.1, n)
        contract = np.random.choice(["Month-to-month","One year","Two year"], n, p=[.4,.3,.3])
        internet = np.random.choice(["Fiber optic","DSL","No"], n, p=[.4,.4,.2])
        tickets = np.random.poisson(1.5, n)
        p = .27 + (1 - tenure/72)*.2 + (charges/120)*.15
        p -= np.isin(contract, ["One year","Two year"]).astype(float) * .15
        p += (tickets/10) * .1
        p = np.clip(p, 0, 1)
        churned = np.random.binomial(1, p)
        st.session_state.cs_data = pd.DataFrame({
            "tenure": tenure, "monthly_charges": charges,
            "total_charges": total, "contract_type": contract,
            "internet_service": internet, "num_support_tickets": tickets,
            "churned": churned,
        })

    df = st.session_state.cs_data
    stage = st.session_state.cs_stage

    # Progress bar
    st.progress(min(stage / 5, 1.0), text=f"Stage {stage} of 5")

    # ── STAGE 1 ──
    if stage >= 1:
        st.header("Stage 1 — The Business Problem")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Churn Rate","27%"); c2.metric("ARPU","$65")
        c3.metric("Acquisition Cost","$350"); c4.metric("Retention Cost","$50")
        info_box("<b>ROI:</b> Reducing churn by 5 % saves ~$2.1 M/year. "
                 "50,000 customers × $65 ARPU × 12 mo × 5 % = $1.95 M.")
        st.write("**TelcoNow** has a 27 % monthly churn rate. Leadership wants a tool that "
                 "customer-service reps can use mid-call to flag at-risk customers and offer "
                 "targeted retention deals.")
        quiz = st.radio("What type of ML problem is this?",
                        ["Regression","Classification","Clustering","Time Series"], key="cs_q1")
        if quiz == "Classification":
            st.success("Correct — binary classification (churn vs. retain).")
            if stage == 1 and st.button("Continue to Stage 2 →", key="cs1"):
                st.session_state.cs_stage = 2; st.rerun()
        else:
            st.error("Not quite — think: will they churn *yes or no*?")

    # ── STAGE 2 ──
    if stage >= 2:
        st.markdown("---")
        st.header("Stage 2 — Data Exploration")
        info_box("<b>Look for:</b> Which features differ most between churned and retained?")

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.patch.set_facecolor("white")

        # 1 – Churn by contract
        ax = axes[0]
        cr = df.groupby("contract_type")["churned"].mean()*100
        ax.bar(cr.index, cr.values, color=NEU_PALETTE[:3], edgecolor="black")
        ax.set_ylabel("Churn Rate (%)")
        ax.set_title("Churn by Contract", fontweight="bold")
        for i, v in enumerate(cr.values):
            ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=9)

        # 2 – Charges histogram
        ax = axes[1]
        ax.hist(df[df.churned==1]["monthly_charges"], bins=25, alpha=.6, color="#D31B2C",
                label="Churned", edgecolor="black")
        ax.hist(df[df.churned==0]["monthly_charges"], bins=25, alpha=.6, color="#0ECCDE",
                label="Retained", edgecolor="black")
        ax.set_xlabel("Monthly Charges ($)"); ax.legend(fontsize=8)
        ax.set_title("Charges Distribution", fontweight="bold")

        # 3 – Tenure box
        ax = axes[2]
        bp = ax.boxplot([df[df.churned==0]["tenure"], df[df.churned==1]["tenure"]],
                        labels=["Retained","Churned"], patch_artist=True, showmeans=True)
        for patch, c in zip(bp["boxes"], ["#0ECCDE","#D31B2C"]):
            patch.set_facecolor(c); patch.set_alpha(.7)
        ax.set_ylabel("Tenure (months)")
        ax.set_title("Tenure vs Churn", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)

        risk = st.multiselect("Select the 3 features most predictive of churn:",
                              df.columns.drop("churned").tolist(), max_selections=3, key="cs_rf")
        if len(risk) == 3:
            st.success(f"Selected: {', '.join(risk)}")
            if stage == 2 and st.button("Continue to Stage 3 →", key="cs2"):
                st.session_state.cs_stage = 3; st.rerun()

    # ── STAGE 3 ──
    if stage >= 3:
        st.markdown("---")
        st.header("Stage 3 — Modeling")
        info_box("We'll use a pre-tuned Random Forest. In production you'd compare multiple models.")

        st.code("# Encode + scale\n"
                "X = pd.get_dummies(df, drop_first=True)\n"
                "scaler = StandardScaler()\n"
                "X_scaled = scaler.fit_transform(X)", language="python")

        if "cs_model" not in st.session_state:
            X = df[["tenure","monthly_charges","total_charges",
                     "num_support_tickets"]].copy()
            cats = pd.get_dummies(df[["contract_type","internet_service"]], drop_first=True)
            X = pd.concat([X, cats], axis=1)
            y = df["churned"]
            scaler = StandardScaler()
            X_s = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
            X_tr, X_te, y_tr, y_te = train_test_split(X_s, y, test_size=.2, random_state=42)
            mdl = RandomForestClassifier(100, max_depth=10, random_state=42)
            mdl.fit(X_tr, y_tr)
            yp = mdl.predict(X_te)
            st.session_state.cs_model = mdl
            st.session_state.cs_scaler = scaler
            st.session_state.cs_Xcols = X.columns
            st.session_state.cs_metrics = {
                "acc": accuracy_score(y_te, yp),
                "prec": precision_score(y_te, yp),
                "rec": recall_score(y_te, yp),
                "f1": f1_score(y_te, yp),
                "cm": confusion_matrix(y_te, yp),
            }

        met = st.session_state.cs_metrics
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Accuracy", f"{met['acc']:.3f}")
        c2.metric("Precision", f"{met['prec']:.3f}")
        c3.metric("Recall", f"{met['rec']:.3f}")
        c4.metric("F1", f"{met['f1']:.3f}")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.patch.set_facecolor("white")
        sns.heatmap(met["cm"], annot=True, fmt="d", cmap="RdYlGn_r", cbar=False,
                    xticklabels=["Retained","Churned"], yticklabels=["Retained","Churned"],
                    ax=axes[0])
        axes[0].set_title("Confusion Matrix", fontweight="bold")
        axes[0].set_ylabel("True"); axes[0].set_xlabel("Predicted")

        imp = pd.Series(st.session_state.cs_model.feature_importances_,
                        index=st.session_state.cs_Xcols).sort_values()
        axes[1].barh(imp.index, imp.values, color=NEU_PALETTE[:len(imp)], edgecolor="black")
        axes[1].set_title("Feature Importance", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)

        if stage == 3 and st.button("Continue to Stage 4 →", key="cs3"):
            st.session_state.cs_stage = 4; st.rerun()

    # ── STAGE 4 ──
    if stage >= 4:
        st.markdown("---")
        st.header("Stage 4 — The Product Interface")
        st.write("Below is a working **Customer Risk Assessment Tool** — exactly what a "
                 "service rep would see on their screen.")

        left, right = st.columns([1.5, 1])
        with left:
            t_val = st.slider("Tenure (months)", 1, 72, 24, key="p_t")
            mc_val = st.slider("Monthly charges ($)", 20, 120, 70, key="p_mc")
            ct_val = st.selectbox("Contract", ["Month-to-month","One year","Two year"], key="p_ct")
            is_val = st.selectbox("Internet service", ["Fiber optic","DSL","No"], key="p_is")
            tk_val = st.slider("Support tickets (6 mo)", 0, 9, 2, key="p_tk")

        if st.button("⚡ Assess Risk", key="p_go"):
            inp = pd.DataFrame({
                "tenure": [t_val], "monthly_charges": [mc_val],
                "total_charges": [mc_val * t_val], "num_support_tickets": [tk_val],
            })
            cats = pd.DataFrame({
                "contract_type_One year": [1 if ct_val=="One year" else 0],
                "contract_type_Two year": [1 if ct_val=="Two year" else 0],
                "internet_service_Fiber optic": [1 if is_val=="Fiber optic" else 0],
                "internet_service_No": [1 if is_val=="No" else 0],
            })
            X_in = pd.concat([inp, cats], axis=1)
            # reorder to match training columns
            for c in st.session_state.cs_Xcols:
                if c not in X_in.columns:
                    X_in[c] = 0
            X_in = X_in[st.session_state.cs_Xcols]
            X_in = pd.DataFrame(st.session_state.cs_scaler.transform(X_in),
                                columns=st.session_state.cs_Xcols)
            prob = st.session_state.cs_model.predict_proba(X_in)[0][1]

            if prob >= .6:
                lvl, clr, action = "HIGH RISK", "#D31B2C", "Premium retention package + priority support"
            elif prob >= .4:
                lvl, clr, action = "MEDIUM RISK", "#FFA500", "Standard discount or service upgrade"
            else:
                lvl, clr, action = "LOW RISK", "#0ECCDE", "Monitor; standard engagement"

            with right:
                st.markdown(
                    f'<div style="background:{clr};color:white;padding:1rem;border-radius:6px;'
                    f'text-align:center;margin-top:.5rem;">'
                    f'<h3 style="color:white!important;margin:.3rem 0;">{lvl}</h3>'
                    f'<p style="font-size:20px;font-weight:bold;margin:0;">'
                    f'{prob*100:.1f}% churn probability</p></div>',
                    unsafe_allow_html=True,
                )
                st.markdown(f"**Recommended action:** {action}")

        with st.expander("View the Streamlit code"):
            st.code("""tenure = st.slider("Tenure", 1, 72, 24)
charges = st.slider("Monthly Charges", 20, 120, 70)
contract = st.selectbox("Contract", options)

if st.button("Assess Risk"):
    X = prepare_input(tenure, charges, contract, ...)
    prob = model.predict_proba(X)[0][1]
    st.metric("Churn Risk", f"{prob*100:.1f}%")
    st.write("Action:", recommended_action(prob))""", language="python")

        info_box("<b>Product thinking:</b> The interface is designed for a service rep — simple "
                 "inputs, prominent result, clear action. The data science is hidden behind a "
                 "clean UI.")

        if stage == 4 and st.button("Continue to Stage 5 →", key="cs4"):
            st.session_state.cs_stage = 5; st.rerun()

    # ── STAGE 5 ──
    if stage >= 5:
        st.markdown("---")
        st.header("Stage 5 — Deployment & Reflection")

        st.subheader("Pre-Deployment Checklist")
        l, r = st.columns(2)
        with l:
            st.checkbox("Model versioning", key="ck1")
            st.checkbox("Data pipeline for retraining", key="ck2")
            st.checkbox("Monitoring for model drift", key="ck3")
        with r:
            st.checkbox("A/B testing retention offers", key="ck4")
            st.checkbox("Feedback loop from reps", key="ck5")

        st.subheader("Architecture")
        st.code("""
┌───────────────────────────────────────────────────────┐
│              TelcoNow Churn Product                   │
├───────────────────────────────────────────────────────┤
│                                                       │
│  CRM Data → Preprocess → Model → Risk Prediction     │
│             (encode,     (RF)    (API / Streamlit)    │
│              scale)                                   │
│                          ↓                            │
│  ┌─────────────────────────────────────────────────┐  │
│  │ Monitoring & Feedback Loop                      │  │
│  │ • Track accuracy vs actual churn                │  │
│  │ • Monitor rep actions & outcomes                │  │
│  │ • Retrain on drift detection                    │  │
│  └─────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────┘""")

        st.subheader("Reflection")
        st.markdown(
            "1. **How would you measure success?** — churn reduction, rep adoption, ROI.\n\n"
            "2. **Accuracy drops below 70 %?** — investigate distribution shift, retrain, try ensembles.\n\n"
            "3. **Explain to a non-technical stakeholder?** — we flag risky customers so reps can act.\n\n"
            "4. **Biggest deployment risk?** — fairness across segments, over-reliance on the model."
        )

        st.markdown(
            '<div style="background:linear-gradient(135deg,#0ECCDE 0%,#0C3354 100%);color:white;'
            'padding:1.5rem;border-radius:6px;text-align:center;margin-top:2rem;">'
            '<h2 style="color:white!important;margin:.5rem 0;">Congratulations!</h2>'
            '<p>You just walked through building a complete data product — from problem '
            'definition to deployment strategy. This is the full lifecycle of real-world '
            'data science.</p></div>',
            unsafe_allow_html=True,
        )


# ╔══════════════════════════════════════════════════════════════╗
# ║  MAIN — SIDEBAR NAVIGATION                                 ║
# ╚══════════════════════════════════════════════════════════════╝

st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/6/6f/Northeastern_seal.svg", width=80)
st.sidebar.markdown("## ALY 6040 Data Mining")
st.sidebar.markdown("*Spring 2026 — Prof. Grosz*")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select a Lesson",
    [
        "Week 1 — Data Cleaning",
        "Week 2 — Business Insights",
        "Week 3 — ML Comparison",
        "Week 4 — Topic Modeling",
        "Week 5 — Case Study",
    ],
    key="nav",
)

if page.startswith("Week 1"):
    week1_cleaning_lab()
elif page.startswith("Week 2"):
    week2_business_insights()
elif page.startswith("Week 3"):
    week3_ml_comparison()
elif page.startswith("Week 4"):
    week4_topic_modeling()
elif page.startswith("Week 5"):
    week5_case_study()
