import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def train_mlp_model(data_path):
    print("Loading data for MLP training...")
    df = pd.read_csv(data_path)

    # train only WIN or PUSH to force winning behavior to be learned
    df_winning = df[df['winloss'].isin(['Win', 'Push'])]

    features = ['Player_Total', 'Dealer_Upcard', 'Is_Soft', 'True_Count_Scaled']
    X = df_winning[features]
    y = df_winning['Action']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Architecture: 2 hidden layers (16 neurons, 8 neurons)
    mlp = MLPClassifier(
        hidden_layer_sizes=(16, 8),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42,
        early_stopping=True
    )

    mlp.fit(X_train, y_train)

    predictions = mlp.predict(X_test)
    print("\n--- MLP Performance (Winning Logic) ---")
    print(classification_report(y_test, predictions))

    joblib.dump(mlp, 'blackjack_mlp.pkl')

    plt.figure(figsize=(10, 6))
    plt.plot(mlp.loss_curve_)
    plt.title("MLP Training Loss Curve")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    os.makedirs('visuals', exist_ok=True)
    plt.savefig('visuals/mlp_loss_curve.png')

if __name__ == "__main__":
    train_mlp_model("cleaned_blkjckhands.csv")