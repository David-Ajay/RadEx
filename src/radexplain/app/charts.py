"""The Findings bar chart: clean horizontal bars with breathing room, red / amber / grey by assessment,
a bold dark tick for each finding's flag threshold, and the probability printed at the end of each bar."""
import altair as alt
import pandas as pd

COLORS = {"likely": "#ef4444", "possible": "#f59e0b", "not flagged": "#94a3b8"}
INK = "#12304f"


def findings_chart(rows: list[dict], width="container"):
    df = pd.DataFrame(rows)
    df["Label"] = df["Probability"].map(lambda p: f"{p:.1f}%")
    df["LabelX"] = df[["Probability", "Threshold"]].max(axis=1)         # keep the label clear of the tick
    y = alt.Y("Finding:N", sort=list(df["Finding"]), title=None,
              scale=alt.Scale(paddingInner=0.45, paddingOuter=0.1),            # gaps between bars
              axis=alt.Axis(ticks=False, domain=False, labelFontSize=13, labelColor=INK, labelLimit=220,
                            labelOverlap=False, labelPadding=8))               # show every label
    base = alt.Chart(df).encode(y=y)
    bars = base.mark_bar(cornerRadiusEnd=6).encode(
        x=alt.X("Probability:Q", title="Calibrated probability (%)   |   dark tick = flag threshold",
                scale=alt.Scale(domainMin=0),
                axis=alt.Axis(grid=False, domain=False, ticks=False, labelColor="#5b7692", titleColor="#5b7692",
                              titleFontWeight="normal", tickCount=6)),
        color=alt.Color("Assessment:N", legend=alt.Legend(title=None, orient="bottom", symbolType="square",
                                                          labelFontSize=12, labelColor=INK),
                        scale=alt.Scale(domain=list(COLORS), range=list(COLORS.values()))),
        tooltip=["Finding", "Probability", "Threshold", "Assessment"])
    ticks = base.mark_tick(color="#0a2540", thickness=4, size=30, opacity=1).encode(x="Threshold:Q")
    labels = base.mark_text(align="left", dx=10, fontSize=12, fontWeight="bold", color=INK).encode(
        x="LabelX:Q", text="Label:N")
    return (bars + ticks + labels).properties(height=alt.Step(36), width=width).configure_view(stroke=None)
