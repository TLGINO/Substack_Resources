import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# data taken from https://fred.stlouisfed.org/series/QCHN628BIS


def prev():
    # TAKEN from beating_inflation_revisited
    # Source: https://data.worldbank.org/indicator/FP.CPI.TOTL.ZG?locations=CH
    df = pd.read_csv("prev_data.csv")
    df = df.drop(columns=["Country Code", "Indicator Name", "Indicator Code"])
    df = df[df["Country Name"] == "Switzerland"]

    def compound(row: pd.Series) -> pd.Series:
        out, running = [], 1.0
        for v in row:
            running *= 1 + v / 100
            out.append(running)
        return pd.Series(out, index=row.index)

    year_cols = [str(y) for y in range(1970, 2025)]
    df = df[["Country Name"] + year_cols]
    df[year_cols] = df[year_cols].apply(compound, axis=1)
    df[year_cols] = df[year_cols] * 100
    return df


def load_house_price_data(filepath, year_cols):
    df = pd.read_csv(filepath)
    df["Year"] = pd.to_datetime(df["observation_date"]).dt.year.astype(str)
    df = df.groupby("Year")["QCHN628BIS"].last().astype(float).reset_index()
    df = df.set_index("Year").T
    base_value = df.iloc[0, 0]
    df = df.apply(lambda x: (x / base_value) * 100, axis=1)
    df = df.reindex(columns=year_cols)
    df.index = [""]
    return df


def prepare_prev_data(year_cols):
    df_prev = prev()
    df_prev = df_prev[["Country Name"] + year_cols]
    df_prev_values = df_prev.drop(columns=["Country Name"])
    df_prev_values.index = [""]
    return df_prev_values


def compute_ratio(df_data, df_prev_data):
    ratio = df_data.copy()
    ratio.iloc[0] = df_data.iloc[0].values / df_prev_data.iloc[0].values * 100
    ratio["Type"] = "Residential Property Prices (CH) / Compound Inflation"
    return ratio


def prepare_plot_df(df_data, df_prev_data, df_ratio, year_cols):
    df_data_plot = df_data.copy()
    df_data_plot["Type"] = "Residential Property Prices (CH)"
    df_prev_plot = df_prev_data.copy()
    df_prev_plot["Type"] = "Compound inflation (CH)"
    plot_df = pd.concat([df_data_plot, df_prev_plot, df_ratio], ignore_index=True)
    intersect_years = [col for col in year_cols if not (plot_df[col].isnull().all())]
    return plot_df, intersect_years


def melt_long(plot_df, intersect_years):
    long = plot_df.melt(
        id_vars="Type",
        value_vars=intersect_years,
        var_name="Year",
        value_name="CumGrowth",
    )
    long["Year"] = long["Year"].astype(int)
    return long


def plot_growth(long):
    sns.set_theme(style="whitegrid")
    ax = sns.lineplot(
        data=long, x="Year", y="CumGrowth", hue="Type", marker="o", linewidth=3
    )
    ax.legend().remove()
    right_margin = 4
    fig = plt.gcf()
    fig.set_size_inches(14, 8)
    n_types = len(long["Type"].unique())
    label_offsets = np.linspace(0, right_margin - 0.5, n_types) * 0.3
    for i, (typ, grp) in enumerate(long.groupby("Type")):
        last_point = grp.dropna(subset=["CumGrowth"]).iloc[-1]
        x = last_point["Year"]
        y = last_point["CumGrowth"]
        ax.text(x + 0.3 + label_offsets[i], y, typ, va="center", fontsize=12)
    ax.set_xlim(long["Year"].min(), long["Year"].max() + right_margin)
    ax.set_ylabel("Compound Growth (%)")
    ax.set_title(
        "Switzerland: Residential Property Prices vs. Compound Inflation 1970-2024"
    )
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig("fig.png", bbox_inches="tight")
    plt.show()


def main():
    year_cols = [str(y) for y in range(1970, 2025)]
    df_data = load_house_price_data("data.csv", year_cols)
    df_prev_data = prepare_prev_data(year_cols)
    df_ratio = compute_ratio(df_data, df_prev_data)
    plot_df, intersect_years = prepare_plot_df(
        df_data, df_prev_data, df_ratio, year_cols
    )
    long = melt_long(plot_df, intersect_years)
    plot_growth(long)


if __name__ == "__main__":
    main()
