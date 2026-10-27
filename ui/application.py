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
    fig, ax = plt.subplots(figsize=(9, 4.5))
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
                    ax.bar(plot["x"], values, label=name, color="#276e95")
                else:
                    ax.plot(plot["x"], values, label=name, linewidth=1.8)
            if "points" in plot:
                ax.scatter(
                    *plot["points"], color="#e38c22", label="Input data", zorder=3
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
        ax.grid(alpha=0.2)
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
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
    st.caption(topic.upper())
    st.title(method.name)
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
