import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def train_decision_tree(data_path):
    print(f"Loading cleaned dataset from {data_path}...")
    df = pd.read_csv(data_path)

    # filter true count 0 for basic strategy
    df_baseline = df[(df['True_Count'] == 0)]

    features = ['Player_Total', 'Dealer_Upcard', 'Is_Soft']
    X = df_baseline[features]
    y = df_baseline['Action']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    dt_model = DecisionTreeClassifier(max_depth=6, criterion='gini', random_state=42)
    dt_model.fit(X_train, y_train)

    y_pred = dt_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Baseline Model Accuracy: {acc * 100:.2f}%")

    joblib.dump(dt_model, 'baseline_tree.pkl')

    importances = dt_model.feature_importances_

    plt.figure(figsize=(8, 5))
    plt.barh(features, importances, color='#10b981')
    plt.xlabel('Mathematical Importance (Gini)')
    plt.title('Baseline Model: Feature Importance')
    plt.gca().invert_yaxis()
    plt.savefig('visuals/feature_importance.png', dpi=300, bbox_inches='tight')

    plt.figure(figsize=(24, 12))
    plot_tree(
        dt_model,
        feature_names=features,
        class_names=['Stand', 'Hit'],
        filled=True,
        rounded=True,
        fontsize=8
    )
    plt.title("Decision Tree: Learned Blackjack Basic Strategy (True Count = 0)", fontsize=16)
    plt.savefig('visuals/basic_strategy_tree.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    train_decision_tree("cleaned_blkjckhands.csv")