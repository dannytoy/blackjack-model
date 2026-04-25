from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Blackjack ML Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    print("Loading ML Models...")
    baseline_tree = joblib.load('baseline_tree.pkl')
    mlp_model = joblib.load('blackjack_mlp.pkl')
    scaler = joblib.load('scaler.pkl')
    print("Models loaded successfully!")
except Exception as e:
    print(f"Warning: Models not found. Error: {e}")

class HandRequest(BaseModel):
    player_total: int
    dealer_upcard: int
    is_soft: int
    true_count: int

# --- HARDCODED LOGIC ---

def get_basic_strategy(total: int, upcard: int, is_soft: int) -> str:
    """Standard Casino Basic Strategy (Simplified for Hit/Stand)."""
    if is_soft:
        if total <= 17: return "Hit"
        if total == 18 and upcard in [9, 10, 11]: return "Hit"
        return "Stand"
    else:
        if total <= 11: return "Hit"
        if total == 12 and upcard in [4, 5, 6]: return "Stand"
        if 13 <= total <= 16 and upcard <= 6: return "Stand"
        if total >= 17: return "Stand"
        return "Hit"

def get_illustrious_18_deviation(total: int, upcard: int, true_count: int, base_action: str) -> str:
    """Card Counting Deviations for Hit/Stand decisions."""
    if total == 16 and upcard == 10 and true_count >= 0: return "Stand"
    if total == 15 and upcard == 10 and true_count >= 4: return "Stand"
    if total == 16 and upcard == 9 and true_count >= 5: return "Stand"
    if total == 15 and upcard == 9 and true_count >= 5: return "Stand"
    if total == 13 and upcard == 2 and true_count >= -1: return "Stand"
    if total == 12 and upcard == 3 and true_count >= 2: return "Stand"
    if total == 12 and upcard == 2 and true_count >= 3: return "Stand"

    return base_action

# --- API ENDPOINTS ---

@app.get("/")
def health_check():
    return {"status": "Active", "models_loaded": True}

@app.post("/predict")
def predict_move(hand: HandRequest):
    basic_action = get_basic_strategy(hand.player_total, hand.dealer_upcard, hand.is_soft)
    counter_action = get_illustrious_18_deviation(hand.player_total, hand.dealer_upcard, hand.true_count, basic_action)

    scaled_tc = scaler.transform([[hand.true_count]])[0][0]

    tree_features = pd.DataFrame([{
        'Player_Total': hand.player_total,
        'Dealer_Upcard': hand.dealer_upcard,
        'Is_Soft': hand.is_soft
    }])

    mlp_features = pd.DataFrame([{
        'Player_Total': hand.player_total,
        'Dealer_Upcard': hand.dealer_upcard,
        'Is_Soft': hand.is_soft,
        'True_Count_Scaled': scaled_tc
    }])

    tree_pred = "Hit" if baseline_tree.predict(tree_features)[0] == 1 else "Stand"
    mlp_pred = "Hit" if mlp_model.predict(mlp_features)[0] == 1 else "Stand"

    return {
        "hints": {
            "basic_strategy": basic_action,
            "card_counter": counter_action,
            "ml_model": mlp_pred,
            "baseline_tree": tree_pred
        },
        "context": {
            "player_total": hand.player_total,
            "dealer_upcard": hand.dealer_upcard,
            "true_count": hand.true_count
        }
    }