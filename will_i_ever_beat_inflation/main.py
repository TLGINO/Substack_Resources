import pandas as pd

import matplotlib.pyplot as plt


# Taken from: https://fred.stlouisfed.org/series/CORESTICKM159SFRBATL
# Read the CSV file
df = pd.read_csv("data.csv", parse_dates=["observation_date"])


# Plot the data
def draw():
    plt.figure(figsize=(8, 4))
    plt.plot(df["observation_date"], df["value"], marker="o")
    plt.xlabel("Date")
    plt.ylabel("Inflation %")
    plt.title("Inflation in the USA Over Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("fig1.png")
    plt.show()


draw()


def draw_compound_inflation():
    # Aggregate average yearly value
    df["year"] = df["observation_date"].dt.year
    yearly_avg = df.groupby("year")["value"].mean().reset_index()
    print("Yearly average values:")
    print(yearly_avg)

    # Compound inflation YoY
    vals = list(yearly_avg["value"])
    print(vals)
    avg = sum(vals) / len(vals) if vals else 0
    print("Average yearly value:", avg)

    compound_res = 1
    compound_inflation_list = []
    for v in vals:
        n_v = 1 + v / 100
        compound_res *= n_v
        print(n_v)
        compound_inflation_list.append(compound_res)
    print(compound_inflation_list)

    # Plot compound inflation over years
    plt.figure(figsize=(8, 4))
    plt.plot(yearly_avg["year"], [a * 100 for a in compound_inflation_list], marker="o")
    plt.xlabel("Year")
    plt.ylabel("Inflation %")
    plt.title("Compound Inflation in the USA Over Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("fig2.png")
    plt.show()


draw_compound_inflation()


# Compound salary YoY
start_salary = 100
compound_salary_list = []
for i in range(50):
    start_salary = start_salary * 1.043
    compound_salary_list.append(start_salary)
    if i % 10 == 0:
        print(i, compound_salary_list[-1])
print(compound_salary_list)
