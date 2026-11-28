"""Input controls and result rendering, without numerical algorithms."""

import logging
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache" / "matplotlib")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from services.catalog import METHODS


def render_result(r):
    if r["converged"] is False:
        st.warning(r["message"])
    else:
        st.success(r["message"])
    cols = st.columns(3)
    value = r["result"]
    if isinstance(value, (int, float)):
        cols[0].metric("Result", f"{value:.9g}")
        st.caption(f"Result at full display precision: {value:.16g}")
    else:
        st.write("Result", value)
    cols[1].metric("Recorded steps", len(r["history"]))
    error = r["error_estimate"]
    cols[2].metric(
        r["diagnostics"].get("error_kind", "Error estimate"),
        "Not available" if error is None else f"{error:.5g}",
    )
    st.subheader("Steps and iterations")
    frame = pd.DataFrame(r["history"])
    st.dataframe(frame, hide_index=True, width="stretch")
    st.download_button(
        "Download step table", frame.to_csv(index=False), "steps.csv", "text/csv"
    )
    st.subheader("Visualization")
    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor="#101f36")
    ax.set_facecolor("#101f36")
    try:
        plot = r.get("plot")
        if plot and plot.get("kind") == "matrix":
            im = ax.imshow(plot["matrix"], cmap="coolwarm", aspect="auto")
            fig.colorbar(im, ax=ax, label="Entry value")
            ax.set(
                title=plot.get("title", "Matrix"),
                xlabel="Column (zero-based)",
                ylabel="Row (zero-based)",
            )
        elif plot:
            for name, values in plot["series"].items():
                if plot.get("kind") == "bar":
                    ax.bar(plot["x"], values, label=name, color="#b7e84a")
                else:
                    ax.plot(plot["x"], values, label=name, linewidth=2.1, color="#b7e84a")
            if "points" in plot:
                ax.scatter(
                    *plot["points"], color="#65c8ff", label="Input data", zorder=3
                )
            ax.set(xlabel=plot.get("xlabel", "x"), ylabel=plot.get("ylabel", "Value"))
            if plot.get("logx"):
                ax.set_xscale("log")
            if plot.get("logy"):
                ax.set_yscale("log")
        else:
            numeric = frame.select_dtypes(include="number")
            for name in list(numeric.columns)[-3:]:
                ax.plot(range(1, len(frame) + 1), numeric[name], marker=".", label=name)
            ax.set(xlabel="Recorded step", ylabel="Value")
        ax.tick_params(colors="#dbe7f6")
        ax.xaxis.label.set_color("#dbe7f6")
        ax.yaxis.label.set_color("#dbe7f6")
        for spine in ax.spines.values():
            spine.set_color("#38536e")
        ax.grid(alpha=0.22, color="#6686a8")
        if ax.get_legend_handles_labels()[0]:
            legend = ax.legend(facecolor="#162b46", edgecolor="#38536e")
            for label in legend.get_texts():
                label.set_color("#eef5ff")
        fig.tight_layout()
        st.pyplot(fig)
    finally:
        plt.close(fig)
    with st.expander("Verification and numerical diagnostics", expanded=True):
        st.write(r["diagnostics"])


def main():
    st.set_page_config(
        page_title="Numerical Calculator | CS-2008", page_icon="∑", layout="wide"
    )
    st.markdown(
        """<style>
        :root { --nc-navy: #07172b; --nc-panel: #102440; --nc-panel-2: #132d4d;
                --nc-line: #2b4b6c; --nc-text: #eef5ff; --nc-muted: #9db3cb;
                --nc-lime: #b7e84a; --nc-blue: #65c8ff; }
        .stApp, [data-testid="stAppViewContainer"] {
            background-color: var(--nc-navy);
            background-image: radial-gradient(circle at 82% 5%, rgba(101, 200, 255, .10), transparent 24rem),
                linear-gradient(rgba(101, 200, 255, .027) 1px, transparent 1px),
                linear-gradient(90deg, rgba(101, 200, 255, .027) 1px, transparent 1px);
            background-size: auto, 34px 34px, 34px 34px;
            background-position: 0 0, 0 0, 0 0;
            color: var(--nc-text);
            animation: nc-grid-drift 28s linear infinite;
        }
        [data-testid="stSidebar"] { background: linear-gradient(180deg, rgba(11, 30, 53, .96) 0%, rgba(7, 23, 43, .96) 100%); border-right: 1px solid var(--nc-line); }
        [data-testid="stSidebar"] * { color: var(--nc-text); }
        [data-testid="stHeader"] { background: rgba(7, 23, 43, 0.92); }
        .block-container { max-width: 1260px; padding-top: 2.25rem; padding-bottom: 3rem; }
        .nc-brand { border-bottom: 1px solid var(--nc-line); padding: 0.25rem 0 1.35rem; margin-bottom: 1.4rem; }
        .nc-eyebrow { color: var(--nc-lime); font-size: 0.72rem; font-weight: 800; letter-spacing: 0.15em; text-transform: uppercase; }
        .nc-brand h1 { color: var(--nc-text); font-size: clamp(2rem, 4vw, 3.35rem); font-weight: 850; line-height: 1; letter-spacing: -0.05em; margin: 0.45rem 0; }
        .nc-brand p { color: var(--nc-muted); font-size: 1rem; margin: 0; }
        h1, h2, h3 { color: var(--nc-text) !important; }
        [data-testid="stCaptionContainer"] { color: var(--nc-lime); font-size: 0.7rem; font-weight: 750; letter-spacing: 0.11em; }
        [data-testid="stMetric"] { background: linear-gradient(145deg, #142d4c, #102440); border: 1px solid var(--nc-line); border-radius: 0.85rem; padding: 1rem; transition: transform .18s ease, border-color .18s ease, box-shadow .18s ease; }
        [data-testid="stMetric"]:hover { transform: translateY(-2px); border-color: #65c8ff; box-shadow: 0 10px 24px rgba(0, 0, 0, .18); }
        [data-testid="stMetricLabel"] { color: var(--nc-muted); }
        [data-testid="stMetricValue"] { color: var(--nc-lime); }
        .stButton > button, [data-testid="stFormSubmitButton"] > button { background: var(--nc-lime); border: 1px solid var(--nc-lime); border-radius: 0.45rem; color: #102014; font-weight: 800; letter-spacing: 0.025em; }
        .stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover { background: #d2f57a; border-color: #d2f57a; color: #07172b; }
        [data-baseweb="input"] > div, [data-baseweb="select"] > div, textarea { background: #102440 !important; border-color: #385b7d !important; color: var(--nc-text) !important; border-radius: 0.45rem !important; }
        [data-baseweb="input"] input, textarea { color: var(--nc-text) !important; }
        [data-baseweb="select"] span { color: var(--nc-text) !important; }
        [data-testid="stDataFrame"] { border: 1px solid var(--nc-line); border-radius: 0.7rem; overflow: hidden; }
        [data-testid="stExpander"] { background: #102440; border: 1px solid var(--nc-line); border-radius: 0.7rem; }
        [data-testid="stInfo"] { background: #102c49; border: 1px solid #286c90; color: var(--nc-text); }
        [data-testid="stSidebar"] .stSelectbox label { color: var(--nc-muted) !important; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.09em; text-transform: uppercase; }
        @keyframes nc-grid-drift { from { background-position: 0 0, 0 0, 0 0; } to { background-position: 0 0, 34px 34px, 34px 34px; } }
        @media (prefers-reduced-motion: reduce) { .stApp, [data-testid="stAppViewContainer"] { animation: none; } }
        </style>""",
        unsafe_allow_html=True,
    )
    st.sidebar.title("Numerical Calculator")
    st.sidebar.caption("CS-2008 · Numerical Computing")
    topics = list(dict.fromkeys(m.topic for m in METHODS))
    topic = st.sidebar.selectbox("Topic", topics, key="topic")
    available = [m for m in METHODS if m.topic == topic]
    families = list(dict.fromkeys(m.group or m.name for m in available))
    family = st.sidebar.selectbox("Method", families, key="method")
    variants = [m for m in available if (m.group or m.name) == family]
    if len(variants) > 1:
        name = st.sidebar.selectbox(
            "Rule / variant", [m.name for m in variants], key=f"variant:{topic}:{family}"
        )
        method = next(m for m in variants if m.name == name)
    else:
        method = variants[0]
        name = method.name
    st.markdown(
        f"""<div class="nc-brand"><div class="nc-eyebrow">CS-2008 / Numerical Workbench</div>
        <h1>{method.name}</h1><p>{topic} · Configure inputs, run the method, inspect the numerical trail.</p></div>""",
        unsafe_allow_html=True,
    )
    st.info(method.description)
    st.sidebar.markdown(
        "**Workflow**\n\n1. Select a method\n2. Enter your problem\n3. Run and inspect the steps"
    )
    st.sidebar.caption("Real-valued inputs. Trigonometric arguments use radians.")
    values = {}
    with st.form(f"form_{name}"):
        for field in method.fields:
            key = f"{name}:{field.key}"
            if field.kind == "number":
                values[field.key] = st.number_input(
                    field.label,
                    value=float(field.default),
                    format="%.10g",
                    key=key,
                    help=field.help,
                )
            elif field.kind == "integer":
                values[field.key] = st.number_input(
                    field.label,
                    value=int(field.default),
                    step=1,
                    key=key,
                    help=field.help,
                )
            elif field.kind == "choice":
                values[field.key] = st.selectbox(
                    field.label, field.options, key=key, help=field.help
                )
            elif field.kind == "area":
                values[field.key] = st.text_area(
                    field.label,
                    str(field.default),
                    key=key,
                    help=field.help,
                    max_chars=10000,
                )
            else:
                values[field.key] = st.text_input(
                    field.label,
                    str(field.default),
                    key=key,
                    help=field.help,
                    max_chars=300,
                )
        run = st.form_submit_button("Run computation", type="primary")
    if run:
        st.session_state.pop("calculation", None)
        try:
            with st.spinner("Computing and checking..."):
                st.session_state.calculation = (name, method.run(values))
        except ValueError as exc:
            st.error(str(exc))
        except Exception:
            logging.exception("Unexpected computation failure")
            st.error(
                "This calculation could not be completed. Check the inputs, domain, and numerical scale."
            )
    saved = st.session_state.get("calculation")
    if saved and saved[0] == name:
        try:
            render_result(saved[1])
        except Exception:
            logging.exception("Result rendering failure")
            st.error(
                "The result could not be displayed. Try a smaller problem or a different numerical scale."
            )
