# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "altair>=5.4",
#     "marimo>=0.23.15",
#     "pandas>=2.2",
# ]
# ///

import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import altair as alt
    import marimo as mo
    import pandas as pd

    return alt, mo, pd


@app.cell
def _(mo):
    mo.md(r"""
    # Pass the Baton: audited reduced-scale reproduction

    **Verdict: partially reproduced.** The released-code version of Relay
    on-policy distillation produced repeatable gains on a fixed,
    disjoint 512-problem mathematical-reasoning evaluation. Its strongest
    reduced-scale settings improved accuracy by about 3.3 percentage
    points, but the experiment does not reproduce the paper's full-scale
    headline and the released takeover predicate differs from the formula
    printed in the paper.

    This notebook is self-contained: every repeat-level value used below
    is embedded in the source and comes from a Kubernetes run with terminal
    measurement evidence. Each value is a trained-minus-base change in the
    number of correct answers out of 512.
    """)
    return


@app.cell
def _():
    GROUPS = {
        "Headline methods": {
            "Standard OPD · 16 updates": [1, 17],
            "Paper-formula Relay · 16 updates": [-3, 13],
            "Released K5 · 16 upd · 1536 tok": [32, 20, 8, 8],
            "Released K5 · 16 upd · 1280 tok": [27, 24, 1, 17],
            "Released K5 · 12 upd · 1024 tok": [16, 11, 8, 0, 3, 7],
        },
        "Response budget": {
            "1024 tokens": [14, 23, 19, -8],
            "1280 tokens": [27, 24, 1, 17],
            "1408 tokens": [6, 17],
            "1536 tokens": [32, 20, 8, 8],
        },
        "Top-K threshold": {
            "K=3": [7, 28, 2, 12],
            "K=5": [32, 20, 8, 8],
            "K=7": [6, 9],
            "K=10": [9, -1],
        },
        "Update horizon at 1024 tokens": {
            "8 updates": [11, 5],
            "12 updates": [16, 11, 8, 0, 3, 7],
            "16 updates": [14, 23, 19, -8],
        },
    }
    TRIGGER_AUDIT = {
        "Paper predicate": 0,
        "Released-code predicate": 8,
    }
    return GROUPS, TRIGGER_AUDIT


@app.cell
def _(GROUPS, pd):
    measurement_rows = []
    for panel_name, panel_conditions in GROUPS.items():
        for condition_name, repeat_deltas in panel_conditions.items():
            for repeat_index, correct_delta in enumerate(repeat_deltas):
                measurement_rows.append(
                    {
                        "panel": panel_name,
                        "condition": condition_name,
                        "repeat": repeat_index + 1,
                        "correct_delta": correct_delta,
                        "accuracy_delta_points": correct_delta / 512 * 100,
                    }
                )

    measurements = pd.DataFrame(measurement_rows)
    summary = (
        measurements.groupby(["panel", "condition"], sort=False)
        .agg(
            repeats=("repeat", "count"),
            mean_correct_delta=("correct_delta", "mean"),
            mean_accuracy_delta_points=("accuracy_delta_points", "mean"),
            min_correct_delta=("correct_delta", "min"),
            max_correct_delta=("correct_delta", "max"),
        )
        .reset_index()
    )
    return measurements, summary


@app.cell
def _(GROUPS, mo):
    panel_picker = mo.ui.dropdown(
        options=list(GROUPS),
        value="Headline methods",
        label="Evidence view",
    )
    panel_picker
    return (panel_picker,)


@app.cell
def _(GROUPS, alt, measurements, panel_picker):
    selected_order = list(GROUPS[panel_picker.value])
    selected_measurements = measurements[
        measurements["panel"] == panel_picker.value
    ]
    selected_means = (
        selected_measurements.groupby("condition", sort=False)
        .agg(mean_accuracy_delta_points=("accuracy_delta_points", "mean"))
        .reset_index()
    )

    zero_rule = (
        alt.Chart(selected_measurements)
        .mark_rule(color="#344054")
        .encode(y=alt.datum(0))
    )
    mean_bars = (
        alt.Chart(selected_means)
        .mark_bar(color="#2a9d8f", opacity=0.82, cornerRadiusTopLeft=4,
                  cornerRadiusTopRight=4)
        .encode(
            x=alt.X(
                "condition:N",
                sort=selected_order,
                title=None,
                axis=alt.Axis(labelAngle=-20, labelLimit=180),
            ),
            y=alt.Y(
                "mean_accuracy_delta_points:Q",
                title="Accuracy change (percentage points)",
            ),
            tooltip=[
                alt.Tooltip("condition:N"),
                alt.Tooltip(
                    "mean_accuracy_delta_points:Q",
                    title="Mean change",
                    format="+.3f",
                ),
            ],
        )
    )
    repeat_points = (
        alt.Chart(selected_measurements)
        .mark_circle(size=90, color="white", stroke="#182230", strokeWidth=1.5)
        .encode(
            x=alt.X("condition:N", sort=selected_order),
            xOffset=alt.XOffset("repeat:O"),
            y=alt.Y("accuracy_delta_points:Q"),
            tooltip=[
                alt.Tooltip("condition:N"),
                alt.Tooltip("repeat:O"),
                alt.Tooltip("correct_delta:Q", title="Correct-answer change"),
                alt.Tooltip(
                    "accuracy_delta_points:Q",
                    title="Accuracy change",
                    format="+.3f",
                ),
            ],
        )
    )
    evidence_chart = (
        (zero_rule + mean_bars + repeat_points)
        .properties(
            width="container",
            height=390,
            title=f"{panel_picker.value}: means and repeat-level evidence",
        )
        .configure_view(stroke=None)
    )
    evidence_chart
    return


@app.cell
def _(mo, panel_picker, summary):
    selected_summary = summary[summary["panel"] == panel_picker.value].copy()
    selected_summary["mean_accuracy_delta_points"] = selected_summary[
        "mean_accuracy_delta_points"
    ].round(3)
    mo.vstack(
        [
            mo.md(
                """
                Bars summarize repeat means; circles preserve the observed
                decoding and training variation. Positive values mean the
                trained checkpoint solved more problems than its matched base
                model.
                """
            ),
            mo.ui.table(
                selected_summary.drop(columns=["panel"]),
                selection=None,
                pagination=False,
            ),
        ]
    )
    return


@app.cell
def _(TRIGGER_AUDIT, alt, pd):
    trigger_frame = pd.DataFrame(
        {
            "predicate": list(TRIGGER_AUDIT),
            "shards_firing": list(TRIGGER_AUDIT.values()),
        }
    )
    trigger_chart = (
        alt.Chart(trigger_frame)
        .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
        .encode(
            x=alt.X("predicate:N", title=None),
            y=alt.Y(
                "shards_firing:Q",
                title="GPU shards where takeover fired",
                scale=alt.Scale(domain=[0, 8]),
            ),
            color=alt.Color(
                "predicate:N",
                scale=alt.Scale(range=["#c27c71", "#287271"]),
                legend=None,
            ),
            tooltip=["predicate:N", "shards_firing:Q"],
        )
        .properties(
            width="container",
            height=300,
            title="Controlled audit: the two takeover predicates are not equivalent",
        )
    )
    trigger_chart
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## How to read the findings

    - **Headline:** at 16 updates, the strongest released Relay settings
      averaged about **+3.3 accuracy points**, versus **+1.76** for matched
      standard distillation and **+0.98** for the literal paper-formula
      implementation.
    - **Response budget:** 1,280 training-response tokens was the strongest
      tested point (**+3.37 points**), essentially tied with 1,536 tokens.
    - **Takeover threshold:** the released top-\(K\) trigger had a broad
      optimum around \(K=5\).
    - **Training horizon:** at 1,024 tokens, eight updates already yielded
      **+11 and +5** correct answers. Across three 12-update training seeds,
      all six repeat deltas were nonnegative:
      `[16, 11, 8, 0, 3, 7]`.
    - **Implementation caveat:** in a constructed divergence case, the
      paper predicate fired on 0/8 shards while the released-code predicate
      fired on 8/8. The strongest result therefore reproduces the released
      implementation, not a literal end-to-end implementation of the
      written formula.

    ## Experimental scope and limitations

    The student was Qwen3-1.7B and the teacher
    Qwen3-4B-Instruct-2507. Training used 2,048 DAPO-Math prompts; evaluation
    used a fixed disjoint tail slice of 512 prompts and two decoding seeds
    per checkpoint. Only selected settings used independent training seeds.
    The campaign is much smaller than the paper's full regime, repeat counts
    differ across conditions, and no formal hypothesis test was
    preregistered. Accuracy gains also came with longer generations and
    higher truncation, so the data support a qualitative direction of
    effect rather than a precise full-scale effect size.
    """)
    return


if __name__ == "__main__":
    app.run()
