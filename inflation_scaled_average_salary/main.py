# Data taken from:
# https://data-explorer.oecd.org/vis?fs[0]=Topic%2C1%7CEmployment%23JOB%23%7CBenefits%252C%20earnings%20and%20wages%23JOB_BW%23&pg=0&fc=Topic&bp=true&snb=21&df[ds]=dsDisseminateFinalDMZ&df[id]=DSD_EARNINGS%40AV_AN_WAGE&df[ag]=OECD.ELS.SAE&df[vs]=1.0&dq=......&pd=1990%2C&to[TIME_PERIOD]=false&vw=ov
# And from previous dataset
import sys
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def prev():
    # TAKEN from beating_inflation_revisited
    # Source: https://data.worldbank.org/indicator/FP.CPI.TOTL.ZG?locations=CH
    df = pd.read_csv("prev_data.csv")

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

    year_cols = [str(y) for y in range(1990, 2025)]
    df[year_cols] = df[year_cols].apply(compound, axis=1)

    # Only take where sufficient data (ie whole set)
    # Interestingly, when when set to 1970 or 1980, the rankings do not change
    df = df[df["1990"].notna()]
    plot_df = df.sort_values(by="2024")  # .head(15)  # Get only lowest 15
    plot_df[year_cols] = plot_df[year_cols] * 100  # Get in percentage
    return plot_df


df_data = pd.read_csv("./data.csv")
df_data = df_data[df_data["UNIT_MEASURE"] == "USD_PPP"]
df_data = df_data[["Reference area", "TIME_PERIOD", "OBS_VALUE"]]

df_data = df_data.pivot(
    index="Reference area", columns="TIME_PERIOD", values="OBS_VALUE"
)
df_data = df_data.rename_axis("Country Name").reset_index()

df_prev_data = prev()
df_prev_data = df_prev_data.reset_index()


intersection = set(df_data["Country Name"]).intersection(
    set(df_prev_data["Country Name"])
)

df_data = df_data[df_data["Country Name"].isin(intersection)]
df_prev_data = df_prev_data[df_prev_data["Country Name"].isin(intersection)]


# Remove 'TIME_PERIOD' from df_data if present (handle MultiIndex columns)
if "TIME_PERIOD" in df_data.columns.names or "TIME_PERIOD" in df_data.columns:
    df_data.columns = [
        col if not (isinstance(col, str) and col == "TIME_PERIOD") else None
        for col in df_data.columns
    ]
    if None in df_data.columns:
        df_data = df_data.drop(columns=[None])
    if "TIME_PERIOD" in df_data.columns:
        df_data = df_data.drop(columns=["TIME_PERIOD"])

# Remove 'index' from df_prev_data if present
if "index" in df_prev_data.columns:
    df_prev_data = df_prev_data.drop(columns=["index"])


# Convert column names that are strings of integers to actual integers
def convert_cols_to_int(df):
    new_cols = []
    for col in df.columns:
        try:
            # Only convert if the column name is a string and can be converted to int
            if isinstance(col, str) and col.isdigit():
                new_cols.append(int(col))
            else:
                new_cols.append(col)
        except Exception:
            new_cols.append(col)
    df.columns = new_cols
    return df


df_data = convert_cols_to_int(df_data)
df_prev_data = convert_cols_to_int(df_prev_data)
df_data = df_data.reset_index(drop=True)
df_prev_data = df_prev_data.reset_index(drop=True)

# Find intersection of year columns (as integers) between both dataframes
year_cols_data = set(col for col in df_data.columns if isinstance(col, int))
year_cols_prev = set(col for col in df_prev_data.columns if isinstance(col, int))
intersect_years = sorted(year_cols_data & year_cols_prev)

# Keep only intersecting year columns (plus 'Country Name')
cols_to_keep = ["Country Name"] + intersect_years
df_data = df_data[cols_to_keep]
df_prev_data = df_prev_data[cols_to_keep]


print(df_data)
print(df_prev_data)
# Align the rows of both dataframes to have the same country order
df_data = df_data.sort_values("Country Name").reset_index(drop=True)
df_prev_data = df_prev_data.sort_values("Country Name").reset_index(drop=True)

# Calculate the ratio for each country and year
plot_df = df_data.copy()
for year in intersect_years:
    plot_df[year] = df_data[year] / df_prev_data[year]

# # Only keep United States, France, and Switzerland
# plot_df = plot_df[
#     plot_df["Country Name"].isin(["United States", "France", "Switzerland"])
# ]

# Express each consecutive year as a percentage change from the previous, starting at 100
plot_df_pct = plot_df.copy()
for country in plot_df_pct["Country Name"]:
    row = plot_df_pct[plot_df_pct["Country Name"] == country]
    values = row[intersect_years].values.flatten()
    pct_changes = [100]
    for i in range(1, len(values)):
        if pd.notna(values[i - 1]) and values[i - 1] != 0:
            pct = (values[i] / values[i - 1]) * pct_changes[-1]
        else:
            pct = np.nan
        pct_changes.append(pct)
    plot_df_pct.loc[plot_df_pct["Country Name"] == country, intersect_years] = (
        pct_changes
    )

plot_df = plot_df_pct
print(plot_df)
# Only keep top 15 countries with highest value in the last year column
last_year = intersect_years[-1]
plot_df = (
    plot_df.sort_values(by=last_year, ascending=False).head(15).reset_index(drop=True)
)
# Print the 2024 column for each country, rounded to int
for idx, row in plot_df.iterrows():
    country = row["Country Name"]
    value_2024 = row[2024]
    print(f"{idx + 1}) {country}: {int(round(value_2024))}")
plot_df.to_csv("inflation_salary_ratio.csv", index=False)


# PLOT


# melt to long format
long = plot_df.melt(
    id_vars="Country Name",
    value_vars=intersect_years,
    var_name="Year",
    value_name="CumGrowth",
)


# plot
sns.set_theme(style="whitegrid")
ax = sns.lineplot(
    data=long, x="Year", y="CumGrowth", hue="Country Name", marker="o", linewidth=3
)

# ------------------------------------
# Directly label the lines on the right-hand side
ax.legend().remove()  # no legend
right_margin = 4  # space for text

# Increase figure size for better spacing
fig = plt.gcf()
fig.set_size_inches(14, 8)  # width, height in inches

# To reduce label overlap, stagger the x-position of the labels slightly (smaller offset)
n_countries = len(long["Country Name"].unique())
label_offsets = (
    np.linspace(0, right_margin - 0.5, n_countries) * 0.3
)  # reduce staggering
for i, (country, grp) in enumerate(long.groupby("Country Name")):
    # last non-NaN point for this country
    last_point = grp.dropna(subset=["CumGrowth"]).iloc[-1]
    x = last_point["Year"]
    y = last_point["CumGrowth"]
    # Apply a small offset to x to reduce overlap
    ax.text(x + 0.3 + label_offsets[i], y, country, va="center", fontsize=10)

# extend x-axis so the text sits inside the frame
ax.set_xlim(long["Year"].min(), long["Year"].max() + right_margin)
ax.set_ylabel("Compound inflation to Average Salary Change %")
ax.set_title(
    "Top 15 Countries With Lowest Compound Inflation to Average Salary ratio from 1990 - 2024"
)

plt.tight_layout(rect=[0, 0, 1, 1])  # leave more room for labels
plt.savefig("fig.png", bbox_inches="tight")
plt.show()
