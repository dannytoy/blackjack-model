import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

def run_diagnostic(data_path):
    df = pd.read_csv(data_path)

    df['Dealer_Upcard'] = pd.to_numeric(df['Dealer_Upcard'], errors='coerce')
    df['Player_Total'] = pd.to_numeric(df['Player_Total'], errors='coerce')
    df = df.dropna(subset=['Dealer_Upcard', 'Player_Total', 'Action'])

    # correlation heatmap
    plt.figure(figsize=(10, 8))
    corr = df[['Player_Total', 'Dealer_Upcard', 'Is_Soft', 'Action', 'True_Count']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")

    plt.title("Feature Correlation Heatmap")
    plt.savefig('visuals/correlation_heatmap.png')
    print("Correlation Heatmap saved to visuals/correlation_heatmap.png")

    # test w/ random forest
    features = ['Player_Total', 'Dealer_Upcard', 'Is_Soft']
    X = df[features]
    y = df['Action']

    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X, y)

    importances = dict(zip(features, rf.feature_importances_))
    print("\n--- Diagnostic Results ---")
    print(f"Feature Importances (Random Forest): {importances}")

    if importances['Dealer_Upcard'] < 0.01:
        print("\nCRITICAL: Dealer_Upcard still has near-zero importance.")
        print(f"Unique Dealer Upcards: {df['Dealer_Upcard'].unique()}")
    else:
        print("\nSUCCESS: Dealer_Upcard importance found. Proceeding with this model configuration.")

if __name__ == "__main__":
    run_diagnostic("../cleaned_blkjckhands.csv")