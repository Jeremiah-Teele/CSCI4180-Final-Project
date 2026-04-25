import pandas as pd
import numpy as np
import psycopg2
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

DB_HOST     = "127.0.0.1"
DB_PORT     = 5432
DB_NAME     = "spotify"
DB_USER     = "root"
DB_PASSWORD = "root"

FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
    "duration_s", "mode"
]

def fetch_data():
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
        user=DB_USER, password=DB_PASSWORD
    )
    base_cols = [f for f in FEATURES if f != "duration_s"] + ["popularity", "genre"]
    df = pd.read_sql(
        f"SELECT {', '.join(base_cols)}, duration_ms / 1000.0 AS duration_s "
        f"FROM songs WHERE popularity IS NOT NULL;", conn
    )
    conn.close()
    return df.dropna()

def run_ols(X, y):
    return sm.OLS(y, sm.add_constant(X)).fit()

def print_summary(model, label):
    print(f"\n{'='*58}")
    print(f"  {label}")
    print(f"  R2: {model.rsquared:.4f}   Adj R2: {model.rsquared_adj:.4f}   N: {int(model.nobs)}")
    print(f"  {'Feature':<22} {'Coef':>10} {'p-value':>10}")
    print(f"  {'-'*44}")
    for feat in FEATURES:
        coef = model.params.get(feat, np.nan)
        pval = model.pvalues.get(feat, np.nan)
        print(f"  {feat:<22} {coef:>10.4f} {pval:>10.4f}")

def plot_coefficient_heatmap(coef_df, pval_df):
    _, ax = plt.subplots(figsize=(13, max(6, len(coef_df) * 0.6)))
    sns.heatmap(
        coef_df, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
        linewidths=0.5, ax=ax, cbar_kws={"label": "Coefficient"}
    )
    for i in range(len(coef_df)):
        for j in range(len(coef_df.columns)):
            if pval_df.iloc[i, j] > 0.05:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color="lightgray", alpha=0.7))
    gray_patch = mpatches.Patch(color="lightgray", alpha=0.7, label="p > 0.05 (not significant)")
    ax.legend(handles=[gray_patch], loc="upper right", bbox_to_anchor=(1.35, 1.1))
    ax.set_title("Linear Regression Coefficients by Genre", fontsize=13, pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig("coefficient_heatmap.png", dpi=150)
    plt.close()
    print("Saved: coefficient_heatmap.png")


def save_excel(coef_df, pval_df):
    with pd.ExcelWriter("linear_results.xlsx", engine="openpyxl") as writer:
        coef_df.to_excel(writer, sheet_name="Coefficients")
        pval_df.to_excel(writer, sheet_name="P-Values")
    print("Saved: linear_results.xlsx")


def main():
    df = fetch_data()
    print(f"Loaded {len(df)} songs across {df['genre'].nunique()} genres.")

    coef_records, pval_records = [], []

    groups = [("All Songs", df)] + [
        (genre, group) for genre, group in df.groupby("genre")
        if len(group) >= 50
    ]

    for label, data in groups:
        model = run_ols(data[FEATURES], data["popularity"])
        print_summary(model, label if label == "All Songs" else f"Genre: {label}  (n={len(data)})")
        coef_records.append({"Genre": label, **{f: model.params.get(f) for f in FEATURES}})
        pval_records.append({"Genre": label, **{f: model.pvalues.get(f) for f in FEATURES}})

    coef_df = pd.DataFrame(coef_records).set_index("Genre")
    pval_df  = pd.DataFrame(pval_records).set_index("Genre")

    plot_coefficient_heatmap(coef_df, pval_df)
    save_excel(coef_df, pval_df)

if __name__ == "__main__":
    main()
