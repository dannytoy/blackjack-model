import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

def get_hilo_value(card_val):
    if pd.isna(card_val) or card_val == 0:
        return 0
    if 2 <= card_val <= 6:
        return 1
    elif 7 <= card_val <= 9:
        return 0
    elif card_val >= 10 or card_val == 1:
        return -1
    return 0

def prep_blackjack_data(file_path, output_path):
    print(f"Loading dataset: {file_path}...")
    df = pd.read_csv(file_path)

    # Engineering True Count (Sequential Hi-Lo)
    running_count = 0
    cards_dealt = 0
    total_decks = 6 # assumption data is 6 deck based shoe
    total_cards_in_shoe = total_decks * 52
    penetration_limit = total_cards_in_shoe * 0.75

    true_counts = []

    for row in df.itertuples():
        decks_remaining = max(1.0, float(total_cards_in_shoe - cards_dealt) / 52.0)
        current_true_count = round(running_count / decks_remaining)
        true_counts.append(current_true_count)

        seen_cards = [
            getattr(row, 'card1', 0), getattr(row, 'card2', 0), getattr(row, 'card3', 0),
            getattr(row, 'card4', 0), getattr(row, 'card5', 0),
            getattr(row, 'dealcard1', 0), getattr(row, 'dealcard2', 0), getattr(row, 'dealcard3', 0),
            getattr(row, 'dealcard4', 0), getattr(row, 'dealcard5', 0)
        ]

        for card in seen_cards:
            if pd.notna(card) and card > 0:
                running_count += get_hilo_value(card)
                cards_dealt += 1

        if cards_dealt >= penetration_limit:
            running_count = 0
            cards_dealt = 0

    df['True_Count'] = true_counts

    # column transformations

    # 1. Is_Soft (1 if card1 or card2 is an Ace)
    df['Is_Soft'] = ((df['card1'] == 1) | (df['card1'] == 11) |
                     (df['card2'] == 1) | (df['card2'] == 11)).astype(int)

    # 2. Action (1 for Hit, 0 for Stand) -> Hit if card3 > 0
    df['Action'] = np.where(df['card3'] > 0, 1, 0)

    # rename columns for clarity
    df = df.rename(columns={
        'ply2cardsum': 'Player_Total',
        'dealcard1': 'Dealer_Upcard'
    })

    # drop unused columns
    cols_to_drop = [
        'PlayerNo', 'plwinamt', 'dlwinamt', 'dealcard2', 'dealcard3',
        'dealcard4', 'dealcard5', 'sumofdeal', 'plybustbeat', 'dlbustbeat',
        'blkjck', 'card3', 'card4', 'card5'
    ]
    df = df.drop(columns=cols_to_drop, errors='ignore')

    # scale
    scaler = StandardScaler()
    df['True_Count_Scaled'] = scaler.fit_transform(df[['True_Count']])
    joblib.dump(scaler, 'scaler.pkl')

    # cleanup
    df = df.dropna(subset=['Player_Total', 'Dealer_Upcard', 'Action', 'winloss'])

    print(f"Data Prep Complete! Saving to {output_path}...")
    df.to_csv(output_path, index=False)
    print(f"Final dataset shape: {df.shape}")

if __name__ == "__main__":
    input_csv = "blkjckhands.csv"
    output_csv = "cleaned_blkjckhands.csv"
    prep_blackjack_data(input_csv, output_csv)