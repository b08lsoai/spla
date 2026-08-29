import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("mst_profile.csv")

steps = [
    ("step1_mxv",          "Step 1 (mxv)"),
    ("step2_cedge",        "Step 2 (cedge)"),
    ("step3_t_vec",        "Step 3 (t_vec)"),
    ("step4_index",        "Step 4 (index)"),
    ("step5_search",       "Step 5a (search min edge)"),
    ("step5_update_parent","Step 5b (update parent)"),
    ("step6_filter_s",     "Step 6 (filter S)"),
]

# Step 5a/5b — два оттенка одного красного, остальные шаги — исходная палитра
colors = [
    "#1f77b4",  # Step 1 — mxv
    "#2ca02c",  # Step 2 — cedge
    "#ffd700",  # Step 3 — t_vec
    "#ff7f0e",  # Step 4 — index
    "#e74c3c",  # Step 5a — search (светлее)
    "#a93226",  # Step 5b — update parent (темнее, тот же тон)
    "#9b59b6",  # Step 6 — filter S
]

fig, ax = plt.subplots(figsize=(12, 7))
bottom = pd.Series([0.0] * len(df))
x_labels = df["iteration"].astype(str).radd("Iteration ")

for (col, label), color in zip(steps, colors):
    bars = ax.bar(x_labels, df[col], bottom=bottom, label=label, color=color)
    for bar, val in zip(bars, df[col]):
        if val > 1:  # не подписывать крошечные сегменты
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + val / 2,
                    f"{val:.1f}", ha="center", va="center", color="white", fontsize=9)
    bottom += df[col]

ax.set_title("MST Algorithm - Time Distribution by Iteration")
ax.set_xlabel("Iteration")
ax.set_ylabel("Time (ms)")
ax.legend(loc="upper right")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("mst_profile.png", dpi=150)
plt.show()