import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# Source: https://data.worldbank.org/indicator/FP.CPI.TOTL.ZG?locations=CH
df = pd.read_csv("data.csv")


df = df.drop(columns=["Country Code", "Indicator Name", "Indicator Code"])


# Some data is inexistent and messes the code up
notWanted = ["Suriname"]
df = df[~df["Country Name"].isin(notWanted)]


def compound(row: pd.Series) -> pd.Series:
    out, running = [], 1.0
    for v in row:
        running *= 1 + v / 100
        out.append(running)
    return pd.Series(out, index=row.index)


year_cols = [str(y) for y in range(1960, 2025)]
df[year_cols] = df[year_cols].apply(compound, axis=1)

# Only take where sufficient data (ie whole set)
# Interestingly, when when set to 1970 or 1980, the rankings do not change
df = df[df["1960"].notna()]
plot_df = df.sort_values(by="2024").head(15)  # Get only lowest 15
plot_df[year_cols] = plot_df[year_cols] * 100  # Get in percentage

# Print the values
for _, row in plot_df.iterrows():
    print(f"{row['Country Name']}: {int(row['2024'])}")


# melt to long format
long = plot_df.melt(
    id_vars="Country Name",
    value_vars=year_cols,
    var_name="Year",
    value_name="CumGrowth",
)

long["Year"] = long["Year"].astype(int)

# plot
sns.set_theme(style="whitegrid")
ax = sns.lineplot(
    data=long, x="Year", y="CumGrowth", hue="Country Name", marker="o", linewidth=3
)

# ------------------------------------
# Directly label the lines on the right-hand side
ax.legend().remove()  # no legend
right_margin = 4  # space for text

for country, grp in long.groupby("Country Name"):
    # last non-NaN point for this country
    last_point = grp.dropna(subset=["CumGrowth"]).iloc[-1]
    x = last_point["Year"]
    y = last_point["CumGrowth"]
    ax.text(x + 0.3, y, country, va="center", fontsize=7)

# extend x-axis so the text sits inside the frame
ax.set_xlim(long["Year"].min(), long["Year"].max() + right_margin)
ax.set_ylabel("Inflation %")
ax.set_title("Top 15 Countries With Lowest Compound Inflation from 1960 - 2024")

plt.tight_layout()
plt.savefig("fig.png")
plt.show()
