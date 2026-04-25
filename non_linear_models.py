import pandas as pd
import psycopg2
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
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
MIN_GENRE_SAMPLES = 50


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


def run_models(X, y, label):
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    xgb = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    xgb.fit(X, y)

    print(f"\n{'='*58}")
    print(f"  {label}")
    print(f"  {'Feature':<22} {'RF Importance':>14} {'XGB Importance':>15}")
    print(f"  {'-'*53}")
    for feat, rf_imp, xgb_imp in zip(FEATURES, rf.feature_importances_, xgb.feature_importances_):
        print(f"  {feat:<22} {rf_imp:>14.4f} {xgb_imp:>15.4f}")

    return rf.feature_importances_, xgb.feature_importances_


def plot_importance_heatmap(imp_df, title, filename):
    _, ax = plt.subplots(figsize=(13, max(6, len(imp_df) * 0.6)))
    sns.heatmap(
        imp_df, annot=True, fmt=".3f", cmap="YlOrRd",
        linewidths=0.5, ax=ax, cbar_kws={"label": "Importance"}
    )
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved: {filename}")


def save_excel(rf_df, xgb_df):
    with pd.ExcelWriter("ensemble_results.xlsx", engine="openpyxl") as writer:
        rf_df.to_excel(writer, sheet_name="RF Importances")
        xgb_df.to_excel(writer, sheet_name="XGB Importances")
    print("Saved: ensemble_results.xlsx")


def main():
    df = fetch_data()
    print(f"Loaded {len(df)} songs across {df['genre'].nunique()} genres.")

    rf_records, xgb_records = [], []

    groups = [("All Songs", df)] + [
        (genre, group) for genre, group in df.groupby("genre")
        if len(group) >= MIN_GENRE_SAMPLES
    ]

    for label, data in groups:
        rf_imp, xgb_imp = run_models(data[FEATURES], data["popularity"], label)
        rf_records.append({"Genre": label, **dict(zip(FEATURES, rf_imp))})
        xgb_records.append({"Genre": label, **dict(zip(FEATURES, xgb_imp))})

    rf_df  = pd.DataFrame(rf_records).set_index("Genre")
    xgb_df = pd.DataFrame(xgb_records).set_index("Genre")

    plot_importance_heatmap(rf_df,  "Random Forest Feature Importances by Genre", "rf_importance_heatmap.png")
    plot_importance_heatmap(xgb_df, "XGBoost Feature Importances by Genre",       "xgb_importance_heatmap.png")
    save_excel(rf_df, xgb_df)


if __name__ == "__main__":
    main()
