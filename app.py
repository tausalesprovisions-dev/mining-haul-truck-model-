import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score, roc_curve, precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(page_title="Mining Truck PdM", layout="wide", page_icon="🚜")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = "logistics_truck.csv"          # keep this file next to app.py

TYPE_MAP = {"L": "Low Tier", "M": "Medium Tier", "H": "High Tier"}
REVERSE_TYPE_MAP = {v: k for k, v in TYPE_MAP.items()}
TIERS = ["Healthy", "Watch", "Warning", "Critical"]
TIER_BINS = [-0.001, 0.3, 0.5, 0.7, 1.001]

RAW_TO_INTERNAL = {
    "Air temperature [K]": "air_temp",
    "Process temperature [K]": "proc_temp",
    "Rotational speed [rpm]": "rpm",
    "Torque [Nm]": "torque",
    "Tool wear [min]": "wear",
    "Machine failure": "failure",
}
FAILURE_MODES = {
    "TWF": "Tool wear failure",
    "HDF": "Heat dissipation failure",
    "PWF": "Power failure",
    "OSF": "Overstrain failure",
    "RNF": "Random failure",
}
SENSORS = ["air_temp", "proc_temp", "rpm", "torque", "wear"]
SENSOR_LABELS = {
    "air_temp": "Air temperature [K]",
    "proc_temp": "Process temperature [K]",
    "rpm": "Rotational speed [rpm]",
    "torque": "Torque [Nm]",
    "wear": "Tool wear [min]",
}


# ==========================================
# DATA
# ==========================================
def clean_columns(df):
    """Accept raw headers (Air temperature [K]) or short ones (air_temp)."""
    df = df.rename(columns=RAW_TO_INTERNAL)
    lower = {c: c.lower() for c in df.columns if c.lower() in SENSORS}
    return df.rename(columns=lower)


def find_data_file():
    """Look next to app.py, in the current folder, then Downloads/Desktop/Documents."""
    home = os.path.expanduser("~")
    folders = [BASE_DIR, os.getcwd()] + [os.path.join(home, f) for f in ("Downloads", "Desktop", "Documents")]
    for folder in folders:
        p = os.path.join(folder, DATA_FILE)
        if os.path.exists(p):
            return p
    return None


@st.cache_data(show_spinner=False)
def load_builtin(path):
    return clean_columns(pd.read_csv(path)) if path else None


def read_any(file):
    return pd.read_csv(file) if file.name.lower().endswith(".csv") else pd.read_excel(file)


def make_features(df):
    """Model inputs: the 5 sensors + Type + 3 physics-based features."""
    x = pd.DataFrame(index=df.index)
    x["Type_num"] = df["Type"].map({"L": 0, "M": 1, "H": 2}).astype(float)
    for c in SENSORS:
        x[c] = df[c].astype(float)
    x["temp_diff"] = x["proc_temp"] - x["air_temp"]
    x["power_w"] = x["torque"] * x["rpm"] * 2 * np.pi / 60
    x["strain"] = x["wear"] * x["torque"]
    return x


def diagnose(df):
    """Likely cause per truck, from the failure rules documented for this dataset."""
    temp_diff = df["proc_temp"] - df["air_temp"]
    power = df["torque"] * df["rpm"] * 2 * np.pi / 60
    limit = df["Type"].map({"L": 11000, "M": 12000, "H": 13000})
    out = []
    flags = [
        ((temp_diff < 8.6) & (df["rpm"] < 1380), "Heat dissipation (low temp gap + low rpm)"),
        ((power < 3500) | (power > 9000), "Power out of safe range"),
        (df["wear"] * df["torque"] > limit, "Overstrain (wear x torque too high)"),
        (df["wear"] >= 200, "Tool near end of life (200+ min)"),
    ]
    mask_cols = [m for m, _ in flags]
    for i in range(len(df)):
        causes = [name for m, name in flags if m.iloc[i]]
        out.append(", ".join(causes) if causes else "-")
    return pd.Series(out, index=df.index)


def add_risk(df, probs):
    out = df.copy()
    out["Failure_Probability"] = probs
    out["Risk_Tier"] = pd.cut(probs, bins=TIER_BINS, labels=TIERS)
    out["Diagnosis"] = diagnose(out)
    return out


# ==========================================
# MODELS (trained here from logistics_truck.csv, no .pkl needed)
# ==========================================
@st.cache_resource(show_spinner="Training models on the truck dataset...")
def train_all(path):
    df = load_builtin(path)
    if df is None:
        return None
    X, y = make_features(df), df["failure"].astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    spw = (ytr == 0).sum() / (ytr == 1).sum()

    try:
        from xgboost import XGBClassifier
        champion = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="aucpr",
            random_state=42, n_jobs=1,
        )
        champ_name = "XGBoost"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        champion = HistGradientBoostingClassifier(class_weight="balanced", random_state=42)
        champ_name = "Gradient Boosting (xgboost not installed)"

    models = {
        champ_name: champion,
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
            random_state=42, n_jobs=-1),
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)),
    }
    probs = {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        probs[name] = m.predict_proba(Xte)[:, 1]

    imp = permutation_importance(champion, Xte, yte, scoring="average_precision",
                                 n_repeats=5, random_state=42)
    importance = pd.Series(imp.importances_mean, index=X.columns).sort_values(ascending=False)

    return dict(models=models, champ=champ_name, Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte,
                probs=probs, importance=importance, features=list(X.columns))


def metrics_table(bundle, threshold):
    rows = {}
    y = bundle["yte"]
    for name, p in bundle["probs"].items():
        pred = (p >= threshold).astype(int)
        rows[name] = {
            "Recall": recall_score(y, pred), "Precision": precision_score(y, pred, zero_division=0),
            "F1-Score": f1_score(y, pred), "PR-AUC": average_precision_score(y, p),
            "ROC-AUC": roc_auc_score(y, p), "MCC": matthews_corrcoef(y, pred),
        }
    return pd.DataFrame(rows).T


def confusion(bundle, threshold):
    p = bundle["probs"][bundle["champ"]]
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(bundle["yte"], pred, labels=[0, 1]).ravel()
    return int(tn), int(fp), int(fn), int(tp)


def score_input(df, bundle):
    model = bundle["models"][bundle["champ"]]
    return model.predict_proba(make_features(df))[:, 1]


# ==========================================
# LOAD EVERYTHING
# ==========================================
data_path = find_data_file()
if data_path is None:
    st.title("🚜 Mining Truck Predictive Maintenance")
    st.warning(f"`{DATA_FILE}` wasn't found automatically. Upload it here once and the app will continue.")
    up = st.file_uploader(f"Upload {DATA_FILE}", type=["csv"])
    if up is not None:
        with open(os.path.join(BASE_DIR, DATA_FILE), "wb") as fh:   # saved next to app.py for next time
            fh.write(up.getbuffer())
        st.rerun()
    st.stop()

data = load_builtin(data_path)
bundle = train_all(data_path)
champ = bundle["champ"]

for key in ("ds_extra",):
    st.session_state.setdefault(key, None)

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.title("🚜 Navigation")
page = st.sidebar.radio(
    "Select a Page:",
    [
        "1. Live Predictive Dashboard",
        "2. What-If Simulator",
        "3. Datasets",
        "4. Model Performance",
        "5. Explainability",
        "6. Business Impact",
    ],
)
st.sidebar.divider()

uploaded_file = None
if page == "1. Live Predictive Dashboard":
    st.sidebar.header("📁 Data Source")
    st.sidebar.markdown("Optional: upload a **RAW** truck CSV (same columns as `logistics_truck.csv`) to score new trucks.")
    uploaded_file = st.sidebar.file_uploader("Upload Raw Telemetry (CSV)", type=["csv"])
    st.sidebar.divider()

st.sidebar.markdown("**⚙️ Model setting**")
threshold = st.sidebar.slider(
    "Failure alert threshold", 0.05, 0.95, 0.50, 0.05,
    help="A truck is flagged as 'will fail' when its probability is at or above this. "
         "Lower = catch more failures but more false alarms.",
)
st.sidebar.divider()
st.sidebar.markdown("**🔑 Truck Type Key**")
st.sidebar.markdown("- **L** = Low Tier\n- **M** = Medium Tier\n- **H** = High Tier")

# ==========================================
# PAGE 1: LIVE DASHBOARD
# ==========================================
if page == "1. Live Predictive Dashboard":
    st.title("🚜 Live Predictive Maintenance Dashboard")

    if uploaded_file is not None:
        raw = clean_columns(pd.read_csv(uploaded_file))
        missing = [c for c in ["Type"] + SENSORS if c not in raw.columns]
        if missing:
            st.error(f"Uploaded file is missing columns: {missing}")
            st.stop()
        if "Product ID" in raw.columns:
            raw = raw.set_index("Product ID")
        df = add_risk(raw, score_input(raw, bundle))
        st.info(f"⚡ Live prediction mode: {len(df):,} uploaded trucks scored by {champ}.")
    else:
        scope = st.radio(
            "Fleet to view:",
            ["Hold-out test set (trucks the model has never seen)", "All trucks"],
            horizontal=True,
        )
        base = data.set_index("Product ID") if "Product ID" in data.columns else data
        if scope.startswith("Hold-out"):
            base = base.loc[data.loc[bundle["Xte"].index, "Product ID"]] if "Product ID" in data.columns else base.loc[bundle["Xte"].index]
        probs = score_input(base, bundle)
        df = add_risk(base, probs)
        if "failure" in df.columns:
            df["Actually_Failed"] = df["failure"].map({0: "No", 1: "Yes"})

    # KPIs
    tiers = df["Risk_Tier"].value_counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Trucks Monitored", f"{len(df):,}")
    c2.metric("🚨 Critical", int(tiers.get("Critical", 0)))
    c3.metric("⚠️ Warning", int(tiers.get("Warning", 0)))
    c4.metric("👀 Watch", int(tiers.get("Watch", 0)))

    ch1, ch2 = st.columns(2)
    with ch1:
        st.caption("Trucks per risk tier")
        st.bar_chart(tiers.reindex(TIERS).fillna(0))
    with ch2:
        st.caption("Distribution of failure probability")
        hist, edges = np.histogram(df["Failure_Probability"], bins=20, range=(0, 1))
        st.bar_chart(pd.DataFrame({"Trucks": hist}, index=[f"{e:.2f}" for e in edges[:-1]]))

    st.divider()
    st.subheader("Fleet Status")
    st.caption("🔑 Truck Type Key: **L** = Low Tier · **M** = Medium Tier · **H** = High Tier")
    f1, f2, f3 = st.columns([1, 2, 1])
    with f1:
        types = [t for t in ["L", "M", "H"] if t in set(df["Type"])]
        sel = st.selectbox("Truck Type:", ["All Types"] + [TYPE_MAP[t] for t in types])
    with f2:
        status_filter = st.radio(
            "Risk filter:",
            ["All Alerts", "Critical Only", "Warning Only", "Watch Only", "Healthy Only", "Everything"],
            horizontal=True,
        )
    with f3:
        min_prob = st.slider("Min failure probability", 0.0, 1.0, 0.0, 0.05)

    fdf = df.copy()
    if sel != "All Types":
        fdf = fdf[fdf["Type"] == REVERSE_TYPE_MAP[sel]]
    rules = {"Critical Only": ["Critical"], "Warning Only": ["Warning"], "Watch Only": ["Watch"],
             "Healthy Only": ["Healthy"], "All Alerts": ["Critical", "Warning"]}
    if status_filter in rules:
        fdf = fdf[fdf["Risk_Tier"].isin(rules[status_filter])]
    fdf = fdf[fdf["Failure_Probability"] >= min_prob].sort_values("Failure_Probability", ascending=False)

    shown = fdf.copy()
    shown["Type"] = shown["Type"].map(TYPE_MAP).fillna(shown["Type"])
    cols = [c for c in ["Type", "Failure_Probability", "Risk_Tier", "Diagnosis", "wear", "torque", "rpm",
                        "air_temp", "proc_temp", "Actually_Failed"] if c in shown.columns]
    n_rows = st.slider("Rows to display", 10, 200, 50, 10)
    st.caption(f"{len(fdf):,} trucks match the filters.")
    st.dataframe(
        shown[cols].head(n_rows), use_container_width=True,
        column_config={"Failure_Probability": st.column_config.ProgressColumn(
            "Failure probability", min_value=0.0, max_value=1.0, format="%.3f")},
    )
    st.download_button("⬇️ Download filtered alerts (CSV)", shown[cols].to_csv().encode("utf-8"),
                       file_name="truck_alerts.csv", mime="text/csv")

    st.divider()
    st.subheader("🔧 Inspect a Specific Truck")
    pool = df if sel == "All Types" else df[df["Type"] == REVERSE_TYPE_MAP[sel]]
    pool = pool.sort_values("Failure_Probability", ascending=False)
    if pool.empty:
        st.warning("No trucks for the selected type.")
    else:
        truck = st.selectbox("Select Truck ID (highest risk first)", pool.index[:1000])
        t = pool.loc[truck]
        a, b = st.columns([1, 2])
        with a:
            st.write(f"**Truck Type:** {TYPE_MAP.get(t['Type'], t['Type'])}")
            st.write(f"**Failure Probability:** {t['Failure_Probability']:.2%}")
            st.progress(float(t["Failure_Probability"]))
            tier = str(t["Risk_Tier"])
            {"Critical": st.error, "Warning": st.warning, "Watch": st.info, "Healthy": st.success}[tier](
                {"Critical": "Critical: schedule maintenance immediately!",
                 "Warning": "Warning: inspect at next shift.",
                 "Watch": "Watch: monitor closely.",
                 "Healthy": "Healthy: no action needed."}[tier])
            st.write(f"**🩺 Diagnosis:** {t['Diagnosis']}")
        with b:
            vals = pd.DataFrame({
                "Reading": [t[s] for s in SENSORS],
                "Fleet average": [df[s].mean() for s in SENSORS],
            }, index=[SENSOR_LABELS[s] for s in SENSORS])
            st.dataframe(vals.style.format("{:.1f}"), use_container_width=True)

# ==========================================
# PAGE 2: WHAT-IF SIMULATOR
# ==========================================
elif page == "2. What-If Simulator":
    st.title("🎛️ What-If Failure Simulator")
    st.markdown("Move the sliders to see how sensor readings change failure risk and the diagnosis.")
    row = {}
    cols = st.columns(3)
    with cols[0]:
        ttype = st.selectbox("Truck Type", ["L", "M", "H"], format_func=lambda k: f"{k} - {TYPE_MAP[k]}")
    row["Type"] = ttype
    for i, s in enumerate(SENSORS):
        with cols[(i + 1) % 3]:
            lo, hi, med = float(data[s].min()), float(data[s].max()), float(data[s].median())
            row[s] = st.slider(SENSOR_LABELS[s], lo, hi, med)
    one = pd.DataFrame([row])
    p = float(score_input(one, bundle)[0])
    res = add_risk(one, np.array([p])).iloc[0]
    st.divider()
    a, b = st.columns(2)
    a.metric("Predicted failure probability", f"{p:.2%}")
    a.progress(p)
    tier = str(res["Risk_Tier"])
    {"Critical": st.error, "Warning": st.warning, "Watch": st.info, "Healthy": st.success}[tier](f"Risk tier: {tier}")
    b.write(f"**🩺 Diagnosis:** {res['Diagnosis']}")

# ==========================================
# PAGE 3: DATASETS
# ==========================================
elif page == "3. Datasets":
    st.title("📁 Project Datasets")
    tab1, tab2 = st.tabs(["1️⃣ Truck dataset (used for the model)", "2️⃣ Additional dataset (optional upload)"])

    with tab1:
        st.markdown(f"Loaded automatically from `{DATA_FILE}`. Each row is one truck reading with its sensors and failure outcome.")
        a, b, c, d = st.columns(4)
        a.metric("Rows", f"{len(data):,}")
        b.metric("Columns", data.shape[1])
        c.metric("Failures", int(data["failure"].sum()))
        d.metric("Failure rate", f"{data['failure'].mean():.2%}")

        sub1, sub2, sub3, sub4, sub5 = st.tabs(
            ["Preview", "Column guide", "Failure diagnosis", "Distributions", "Correlations"])
        with sub1:
            st.dataframe(data.head(100), use_container_width=True)
        with sub2:
            guide = pd.DataFrame([
                ("Product ID", "Unique truck/reading identifier"),
                ("Type", "L / M / H quality tier (Low / Medium / High)"),
                ("air_temp", "Air temperature around the truck (Kelvin)"),
                ("proc_temp", "Process temperature (Kelvin)"),
                ("rpm", "Rotational speed"),
                ("torque", "Torque applied (Nm)"),
                ("wear", "Minutes the tool has been in use"),
                ("failure", "TARGET: 1 = failed, 0 = did not"),
                ("TWF / HDF / PWF / OSF / RNF", "Failure causes (the diagnosis): tool wear, heat dissipation, power, overstrain, random"),
            ], columns=["Column", "What it means"])
            st.dataframe(guide, use_container_width=True, hide_index=True)
        with sub3:
            st.markdown("**Why did the trucks fail?** Counts of each failure cause in the dataset. This is the diagnosis layer.")
            fm = pd.Series({FAILURE_MODES[k]: int(data[k].sum()) for k in FAILURE_MODES if k in data.columns})
            st.bar_chart(fm)
            st.caption("The model predicts *whether* a truck will fail. These causes explain *why*, so maintenance knows what to inspect.")
            by_type = data.groupby("Type")["failure"].mean().reindex(["L", "M", "H"]).rename(index=TYPE_MAP)
            st.markdown("**Failure rate by truck tier**")
            st.bar_chart(by_type)
        with sub4:
            col = st.selectbox("Sensor", SENSORS, format_func=lambda s: SENSOR_LABELS[s])
            hist, edges = np.histogram(data[col], bins=30)
            st.bar_chart(pd.DataFrame({"Count": hist}, index=[f"{e:.1f}" for e in edges[:-1]]))
        with sub5:
            corr = data[SENSORS + ["failure"]].corr()
            try:
                st.dataframe(corr.style.background_gradient(cmap="RdBu_r", vmin=-1, vmax=1).format("{:.2f}"),
                             use_container_width=True)
            except Exception:
                st.dataframe(corr.round(2), use_container_width=True)

    with tab2:
        st.markdown("Upload any second dataset to explore it (for example, operational logistics records).")
        f = st.file_uploader("Upload dataset (CSV / Excel)", type=["csv", "xlsx", "xls"], key="extra")
        if f is not None:
            try:
                st.session_state.ds_extra = read_any(f)
            except Exception as e:
                st.error(f"Could not read the file: {e}")
        d = st.session_state.ds_extra
        if d is None:
            st.info("No file uploaded yet.")
        else:
            st.success(f"Loaded: {d.shape[0]:,} rows x {d.shape[1]} columns")
            st.dataframe(d.head(50), use_container_width=True)
            st.dataframe(pd.DataFrame({"Type": d.dtypes.astype(str), "Missing": d.isna().sum(), "Unique": d.nunique()}),
                         use_container_width=True)
            st.dataframe(d.describe(include="all").T, use_container_width=True)

# ==========================================
# PAGE 4: MODEL PERFORMANCE
# ==========================================
elif page == "4. Model Performance":
    st.title("📊 Model Performance")
    st.markdown(
        f"Three models were trained on 80% of the trucks and tested on a hold-out 20% "
        f"({len(bundle['yte']):,} trucks, {int(bundle['yte'].sum())} real failures). "
        f"Final model: **{champ}**. Use the sidebar threshold to see the trade-off."
    )
    perf = metrics_table(bundle, threshold)
    st.subheader(f"Evaluation table (threshold = {threshold:.2f})")
    st.dataframe(perf.style.highlight_max(axis=0, color="#1f6f43").format("{:.3f}"), use_container_width=True)

    picked = st.multiselect("Metrics to chart", list(perf.columns), default=["Recall", "Precision", "PR-AUC"])
    if picked:
        st.bar_chart(perf[picked])

    tn, fp, fn, tp = confusion(bundle, threshold)
    st.subheader(f"Confusion matrix: {champ}")
    a, b, c, d = st.columns(4)
    a.metric("✅ Failures caught (TP)", tp)
    b.metric("❌ Failures missed (FN)", fn)
    c.metric("⚠️ False alarms (FP)", fp)
    d.metric("Correctly healthy (TN)", tn)
    st.dataframe(pd.DataFrame([[tn, fp], [fn, tp]], index=["Actual: OK", "Actual: Failure"],
                              columns=["Predicted: OK", "Predicted: Failure"]), use_container_width=True)

    st.subheader("ROC and Precision-Recall curves")
    grid = np.linspace(0, 1, 101)
    roc_df, pr_df = {}, {}
    for name, p in bundle["probs"].items():
        fpr, tpr, _ = roc_curve(bundle["yte"], p)
        roc_df[name] = np.interp(grid, fpr, tpr)
        pr, rc, _ = precision_recall_curve(bundle["yte"], p)
        pr_df[name] = np.interp(grid, rc[::-1], pr[::-1])
    c1, c2 = st.columns(2)
    with c1:
        st.caption("ROC: True positive rate vs False positive rate")
        st.line_chart(pd.DataFrame(roc_df, index=grid))
    with c2:
        st.caption("Precision vs Recall")
        st.line_chart(pd.DataFrame(pr_df, index=grid))

# ==========================================
# PAGE 5: EXPLAINABILITY
# ==========================================
elif page == "5. Explainability":
    st.title("🧠 Explainability: what drives failures?")
    st.markdown(
        "A black-box model is hard to trust. These charts show which readings the model relies on, "
        "to confirm it learned real physical rules rather than noise."
    )
    st.subheader("Feature importance (permutation importance on the test set)")
    st.caption("How much the model's PR-AUC drops when a feature is scrambled. Bigger = more important.")
    st.bar_chart(bundle["importance"])

    try:
        import shap
        import matplotlib.pyplot as plt
        st.subheader("SHAP summary")
        st.markdown("Red = high feature value, blue = low. Right of centre pushes toward predicting **failure**.")
        sample = bundle["Xte"].sample(min(1000, len(bundle["Xte"])), random_state=0)
        sv = shap.TreeExplainer(bundle["models"][champ]).shap_values(sample)
        fig = plt.figure()
        shap.summary_plot(sv, sample, show=False)
        st.pyplot(fig, clear_figure=True)
    except Exception:
        st.info("Install `shap` (`pip install shap`) to also see the SHAP summary plot here.")

    p = os.path.join(BASE_DIR, "shap_summary.png")
    if os.path.exists(p):
        st.subheader("SHAP summary from the notebook")
        st.image(p, use_container_width=True)

# ==========================================
# PAGE 6: BUSINESS IMPACT
# ==========================================
elif page == "6. Business Impact":
    st.title("💰 Business Impact")
    tn, fp, fn, tp = confusion(bundle, threshold)
    failures = tp + fn
    st.markdown(
        f"Calculated live from the hold-out test set at threshold **{threshold:.2f}** "
        f"({len(bundle['yte']):,} trucks, **{failures}** real failures). Change the costs below."
    )
    a, b = st.columns(2)
    with a:
        downtime = st.number_input("Cost of one unplanned breakdown ($)", 0, 1_000_000, 50_000, 1_000)
        inspect = st.number_input("Cost of one inspection ($)", 0, 100_000, 1_000, 100)
    with b:
        include_tp = st.checkbox("Also charge an inspection for each caught failure", value=True)
        st.write(f"**Caught (TP):** {tp}  |  **Missed (FN):** {fn}  |  **False alarms (FP):** {fp}")

    without_ai = failures * downtime
    with_ai = fn * downtime + fp * inspect + (tp * inspect if include_tp else 0)
    savings = without_ai - with_ai

    c1, c2, c3 = st.columns(3)
    c1.metric("Cost without AI", f"${without_ai:,.0f}")
    c2.metric("Cost with AI", f"${with_ai:,.0f}")
    c3.metric("Estimated savings", f"${savings:,.0f}", f"{savings / without_ai:.1%}" if without_ai else None)
    st.bar_chart(pd.DataFrame({"Cost ($)": [without_ai, with_ai]}, index=["Without AI", "With AI"]))

    st.markdown(f"""
### Breakdown
* **Without AI:** {failures} breakdowns x ${downtime:,} = **${without_ai:,.0f}**
* **With AI:**
    * {fn} missed failures x ${downtime:,} = ${fn * downtime:,.0f}
    * {fp} false alarms x ${inspect:,} = ${fp * inspect:,.0f}
""" + (f"    * {tp} caught failures inspected x ${inspect:,} = ${tp * inspect:,.0f}\n" if include_tp else "") + f"""    * Total = **${with_ai:,.0f}**
* **Savings:** **${savings:,.0f}**
""")

    st.subheader("Savings at different thresholds")
    rows = []
    for th in np.arange(0.1, 0.95, 0.05):
        _, fp_, fn_, tp_ = confusion(bundle, th)
        cost = fn_ * downtime + fp_ * inspect + (tp_ * inspect if include_tp else 0)
        rows.append({"Threshold": round(th, 2), "Savings ($)": failures * downtime - cost})
    st.line_chart(pd.DataFrame(rows).set_index("Threshold"))
    st.caption("The best threshold is where savings peak. Because a miss costs far more than a false alarm, it is usually below 0.5.")