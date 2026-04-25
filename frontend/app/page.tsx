"use client";

import { useState, useEffect, useCallback } from "react";

// --- TYPES & GAME LOGIC ---
type Card = { rank: string; suit: string; value: number; isHidden?: boolean };

const SUITS = ["♠", "♥", "♦", "♣"];
const RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"];

function getCardValue(rank: string): number {
  if (["J", "Q", "K"].includes(rank)) return 10;
  if (rank === "A") return 11;
  return parseInt(rank);
}

function getHiLoValue(rank: string): number {
  const val = getCardValue(rank);
  if (val >= 2 && val <= 6) return 1;
  if (val >= 7 && val <= 9) return 0;
  return -1; // 10, J, Q, K, A
}

function createShoe(decks = 6): Card[] {
  const shoe: Card[] = [];
  for (let d = 0; d < decks; d++) {
    for (const suit of SUITS) {
      for (const rank of RANKS) {
        shoe.push({ rank, suit, value: getCardValue(rank) });
      }
    }
  }
  // Fisher-Yates Shuffle
  for (let i = shoe.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shoe[i], shoe[j]] = [shoe[j], shoe[i]];
  }
  return shoe;
}

function calculateHand(hand: Card[]) {
  let total = 0;
  let aces = 0;
  hand.forEach((card) => {
    if (card.isHidden) return;
    total += card.value;
    if (card.rank === "A") aces += 1;
  });
  while (total > 21 && aces > 0) {
    total -= 10;
    aces -= 1;
  }
  return { total, isSoft: aces > 0 };
}

export default function BlackjackGame() {
  // --- STATE ---
  const [shoe, setShoe] = useState<Card[]>([]);
  const [runningCount, setRunningCount] = useState(0);

  const [playerHand, setPlayerHand] = useState<Card[]>([]);
  const [dealerHand, setDealerHand] = useState<Card[]>([]);

  const [gameState, setGameState] = useState<"idle" | "playing" | "dealerTurn" | "gameOver">("idle");
  const [message, setMessage] = useState("");

  const [hints, setHints] = useState({
    basic_strategy: "-",
    card_counter: "-",
    ml_model: "-",
    baseline_tree: "-",
  });

  // --- GAME ACTIONS ---
  const initializeGame = () => {
    setShoe(createShoe(6));
    setRunningCount(0);
    setGameState("idle");
    setPlayerHand([]);
    setDealerHand([]);
    setMessage("Shoe Shuffled. Ready to Deal.");
  };

  useEffect(() => { initializeGame(); }, []);

  const dealRound = () => {
    if (shoe.length < 20) {
      setMessage("Shoe is almost empty. Reshuffling...");
      initializeGame();
      return;
    }

    const newShoe = [...shoe];
    const pHand = [newShoe.pop()!, newShoe.pop()!];
    const dHand = [newShoe.pop()!, { ...newShoe.pop()!, isHidden: true }];

    let newRc = runningCount;
    pHand.forEach(c => newRc += getHiLoValue(c.rank));
    newRc += getHiLoValue(dHand[0].rank);

    setShoe(newShoe);
    setPlayerHand(pHand);
    setDealerHand(dHand);
    setRunningCount(newRc);
    setGameState("playing");
    setMessage("");
  };

  const hit = () => {
    const newShoe = [...shoe];
    const card = newShoe.pop()!;
    const newHand = [...playerHand, card];

    setShoe(newShoe);
    setPlayerHand(newHand);
    setRunningCount(runningCount + getHiLoValue(card.rank));

    const { total } = calculateHand(newHand);
    if (total > 21) {
      setGameState("gameOver");
      setMessage("Player Busts! Dealer Wins.");
    }
  };

  const stand = async () => {
    setGameState("dealerTurn");
    let currentShoe = [...shoe];
    let dHand = [...dealerHand];

    dHand[1].isHidden = false;
    let newRc = runningCount + getHiLoValue(dHand[1].rank);

    let dTotal = calculateHand(dHand).total;
    while (dTotal < 17 && currentShoe.length > 0) {
      const card = currentShoe.pop()!;
      dHand.push(card);
      newRc += getHiLoValue(card.rank);
      dTotal = calculateHand(dHand).total;

      setDealerHand([...dHand]);
      await new Promise((r) => setTimeout(r, 800));
    }

    setShoe(currentShoe);
    setDealerHand(dHand);
    setRunningCount(newRc);

    const pTotal = calculateHand(playerHand).total;
    if (dTotal > 21) setMessage("Dealer Busts! Player Wins.");
    else if (dTotal > pTotal) setMessage("Dealer Wins.");
    else if (dTotal < pTotal) setMessage("Player Wins!");
    else setMessage("Push (Tie).");

    setGameState("gameOver");
  };

  // --- API INTEGRATION ---
  const decksRemaining = Math.max(1, shoe.length / 52);
  const trueCount = Math.round(runningCount / decksRemaining);

  const fetchPredictions = useCallback(async () => {
    if (gameState !== "playing" || playerHand.length === 0) return;

    const { total, isSoft } = calculateHand(playerHand);
    const upcardValue = getCardValue(dealerHand[0].rank);

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          player_total: total,
          dealer_upcard: upcardValue,
          is_soft: isSoft ? 1 : 0,
          true_count: trueCount,
        }),
      });
      const data = await response.json();
      setHints(data.hints);
    } catch (error) {
      console.error("API Error:", error);
    }
  }, [playerHand, dealerHand, gameState, trueCount]);

  useEffect(() => { fetchPredictions(); }, [fetchPredictions]);

  // --- UI HELPERS ---
  const renderTotal = (hand: Card[]) => {
    const { total, isSoft } = calculateHand(hand);
    if (isSoft && total < 21) {
      return `${total - 10} / ${total} (Soft)`;
    }
    return total;
  };

  return (
    <div className="min-h-screen bg-neutral-900 text-white flex font-sans">

      {/* LEFT PANE: The Table */}
      <div className="flex-1 p-8 flex flex-col relative bg-green-900/30 border-r border-neutral-700">

        {/* Top Info Bar */}
        <div className="flex justify-between items-center bg-black/40 p-4 rounded-lg mb-8">
          <div><span className="text-emerald-400 font-bold tracking-widest">CARDS REMAINING:</span> {shoe.length}</div>
          <div className="text-xl">
            <span className="text-emerald-400 font-bold tracking-widest">TRUE COUNT:</span>
            <span className="font-mono text-2xl ml-2">{trueCount > 0 ? `+${trueCount}` : trueCount}</span>
          </div>
          <div><span className="text-emerald-400 font-bold tracking-widest">RUNNING COUNT:</span> {runningCount}</div>

          {/* RESET SHOE BUTTON */}
          <button
            onClick={initializeGame}
            className="px-4 py-2 bg-red-900/50 hover:bg-red-800 border border-red-700 rounded text-sm font-bold tracking-widest transition-all"
          >
            RESET SHOE
          </button>
        </div>

        {/* Dealer Area */}
        <div className="flex-1 flex flex-col items-center justify-center">
          <h2 className="text-neutral-400 tracking-widest mb-4 font-bold">DEALER</h2>
          <div className="flex space-x-4 h-36">
            {dealerHand.map((card, idx) => <PlayingCard key={idx} card={card} />)}
          </div>
          {gameState === "gameOver" && (
            <div className="mt-4 text-xl font-bold text-neutral-300 bg-black/50 px-4 py-1 rounded-full">
              Total: {renderTotal(dealerHand)}
            </div>
          )}
        </div>

        {/* Game Message Area */}
        <div className="h-16 flex items-center justify-center">
           <h1 className="text-3xl font-black text-white drop-shadow-lg">{message}</h1>
        </div>

        {/* Player Area */}
        <div className="flex-1 flex flex-col items-center justify-center">
          <div className="flex space-x-4 h-36">
            {playerHand.map((card, idx) => <PlayingCard key={idx} card={card} />)}
          </div>
          {playerHand.length > 0 && (
            <div className="mt-4 text-xl font-bold mb-4 text-white bg-black/50 px-4 py-1 rounded-full">
              Total: {renderTotal(playerHand)}
            </div>
          )}

          {/* Controls */}
          <div className="flex space-x-4 mt-4">
            {gameState === "idle" || gameState === "gameOver" ? (
              <button onClick={dealRound} className="px-8 py-3 bg-emerald-600 hover:bg-emerald-500 rounded font-bold tracking-widest transition-all">
                DEAL HAND
              </button>
            ) : (
              <>
                <button onClick={hit} disabled={gameState !== "playing"} className="px-8 py-3 bg-neutral-700 hover:bg-neutral-600 rounded font-bold tracking-widest transition-all disabled:opacity-50">
                  HIT
                </button>
                <button onClick={stand} disabled={gameState !== "playing"} className="px-8 py-3 bg-red-600 hover:bg-red-500 rounded font-bold tracking-widest transition-all disabled:opacity-50">
                  STAND
                </button>
              </>
            )}
          </div>
          <h2 className="text-neutral-400 tracking-widest mt-8 font-bold">PLAYER</h2>
        </div>
      </div>

      {/* RIGHT PANE: Hints Sidebar */}
      <div className="w-96 bg-neutral-950 p-8 flex flex-col shadow-[-10px_0_30px_rgba(0,0,0,0.5)]">
        <h2 className="text-2xl font-bold mb-8 border-b border-neutral-800 pb-4 text-neutral-200">
          Optimal Moves
        </h2>
        <div className="space-y-6 flex-1">
          <HintCard title="Basic Strategy" action={gameState === "playing" ? hints.basic_strategy : "-"} description="Mathematically perfect move ignoring count." />
          <HintCard title="Card Counter (Ill. 18)" action={gameState === "playing" ? hints.card_counter : "-"} description="Optimal deviation based on True Count." />
          <hr className="border-neutral-800 my-6" />
          <h3 className="text-sm font-semibold text-neutral-500 uppercase tracking-widest mb-4">ML Predictions</h3>
          <HintCard title="Baseline Tree" action={gameState === "playing" ? hints.baseline_tree : "-"} description="Trained only on neutral counts." isML />
          <HintCard title="Neural Network (MLP)" action={gameState === "playing" ? hints.ml_model : "-"} description="Trained on all counts. (Bot Clone)" isML />
        </div>
      </div>
    </div>
  );
}

// Reusable Card UI
function PlayingCard({ card }: { card: Card }) {
  if (card.isHidden) {
    return <div className="w-24 h-36 bg-blue-900 border-2 border-white/20 rounded-lg shadow-xl bg-[url('https://www.transparenttextures.com/patterns/argyle.png')]"></div>;
  }
  const isRed = card.suit === "♥" || card.suit === "♦";
  return (
    <div className={`w-24 h-36 bg-white rounded-lg shadow-xl flex flex-col justify-between p-2 border border-neutral-200 ${isRed ? "text-red-600" : "text-black"}`}>
      <div className="text-xl font-bold leading-none">{card.rank}</div>
      <div className="text-4xl self-center">{card.suit}</div>
      <div className="text-xl font-bold leading-none self-end rotate-180">{card.rank}</div>
    </div>
  );
}

function HintCard({ title, action, description, isML = false }: { title: string, action: string, description: string, isML?: boolean }) {
  const isHit = action === "Hit";
  const isStand = action === "Stand";

  return (
    <div className={`p-5 rounded-lg border transition-all ${isML ? 'bg-indigo-900/20 border-indigo-500/30' : 'bg-neutral-800/50 border-neutral-700'} ${action === "-" ? "opacity-50" : "opacity-100"}`}>
      <h3 className={`text-sm font-bold uppercase tracking-wider mb-1 ${isML ? 'text-indigo-400' : 'text-neutral-400'}`}>
        {title}
      </h3>
      <p className="text-xs text-neutral-500 mb-3">{description}</p>
      <div className={`text-2xl font-black tracking-widest rounded py-2 text-center border-2 ${
        isHit ? "bg-red-500/10 text-red-500 border-red-500/50" : 
        isStand ? "bg-blue-500/10 text-blue-400 border-blue-500/50" : 
        "bg-neutral-800 text-neutral-600 border-neutral-700"
      }`}>
        {action.toUpperCase()}
      </div>
    </div>
  );
}