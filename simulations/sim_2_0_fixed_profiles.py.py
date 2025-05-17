#!/usr/bin/env python3
"""
Full-match hand-cricket simulation (two innings)

Save this file as  bot_vs_bot_simulation.py  and run:

    python bot_vs_bot_simulation.py
"""

import random, math, numpy as np, pandas as pd
from collections import Counter, defaultdict
from datetime import datetime

# ---------- 1.  BASIC PROBABILISTIC PROFILES  ----------
player_profiles = {
    "aggressive": {1: .05, 2: .1, 3: .15, 4: .20, 5: .20, 6: .30},
    "cautious"  : {1: .30, 2: .25, 3: .20, 4: .15, 5: .05, 6: .05},
    "deceptive" : {1: .15, 2: .25, 3: .05, 4: .05, 5: .20, 6: .30},
    "random"    : {n: 1/6 for n in range(1,7)}
}

counter_profiles = {
    "aggressive": "cautious",
    "cautious"  : "aggressive",
    "deceptive" : "random",
    "random"    : "deceptive"
}

def choose_from_dist(d):                       # weighted random helper
    nums, probs = zip(*d.items())
    return random.choices(nums, probs, k=1)[0]

def classify_strategy(profile, num):
    if profile=="aggressive" and num>=4: return "Risky"
    if profile=="cautious"   and num<=3: return "Safe"
    if profile=="deceptive"              : return "Bluff"
    return "Normal"

# ---------- 2.  ONE-INNINGS SIMULATION  ----------
def simulate_innings(bat_prof, bowl_prof, innings_no,
                     target=None, max_turns=30):
    """
    Returns (df, total_runs, was_out)
        * If target is given, innings stops early when target is passed.
    """
    bat_hist, bowl_hist = [], []
    total = 0
    records = []

    for turn in range(1, max_turns+1):
        # --- simple adaptive tweak for chase / defend ---
        bat_dist  = player_profiles[bat_prof].copy()
        bowl_dist = player_profiles[bowl_prof].copy()

        if target:             # chasing
            req = target-total
            run_rate_need = req / max(1, (max_turns-turn+1))
            if run_rate_need > 4:               # need to accelerate
                for n in [5,6]: bat_dist[n] *= 1.4
            elif req <= 6:                      # close to target, be careful
                for n in [1,2]: bat_dist[n] *= 1.3
        else:                    # setting a score – aggress near the end
            if turn > max_turns*0.7:
                for n in [4,5,6]: bat_dist[n] *= 1.25

        # bowler: protect total if defending a low target
        if target and target<12:
            for n in [4,5,6]: bowl_dist[n] *= 1.3

        # renormalise
        for dist in (bat_dist,bowl_dist):
            s=sum(dist.values())
            for k in dist: dist[k]/=s

        bat = choose_from_dist(bat_dist)
        bowl= choose_from_dist(bowl_dist)

        out = bat==bowl
        runs= 0 if out else bat
        total += runs

        records.append({
            "Innings": innings_no,
            "Turn"   : turn,
            "Batsman_Input": bat,
            "Bowler_Input" : bowl,
            "Outcome": "OUT" if out else "RUN",
            "Runs_Scored": runs,
            "Total_Runs" : total,
            "Is_Out"     : out,
            "Batsman_Strategy": classify_strategy(bat_prof,bat),
            "Bowler_Strategy" : classify_strategy(bowl_prof,bowl)
        })

        if out:  break
        if target and total>target: break

    return pd.DataFrame(records), total, out

# ---------- 3.  FULL MATCH (TWO INNINGS)  ----------
def simulate_match(profile_a, profile_b, max_turns=30):
    # INNINGS 1  : A bats, B bowls
    inn1, score_a, _ = simulate_innings(profile_a, profile_b, 1,
                                        target=None, max_turns=max_turns)

    # swap roles – INNINGS 2  : B bats, A bowls (chasing)
    inn2, score_b, out2 = simulate_innings(profile_b, profile_a, 2,
                                           target=score_a, max_turns=max_turns)

    result = ("B wins (chased)" if score_b>score_a else
              "A wins (defended)" if out2 else
              "Tie")

    summary = pd.DataFrame([{
        "Innings": "MATCH",
        "Turn": None,
        "Batsman_Input": None,
        "Bowler_Input" : None,
        "Outcome": result,
        "Runs_Scored": None,
        "Total_Runs": f"{score_a}-{score_b}",
        "Is_Out": None,
        "Batsman_Strategy": None,
        "Bowler_Strategy" : None
    }])

    return pd.concat([inn1,inn2,summary], ignore_index=True)

# ---------- 4.  CLI INTERFACE  ----------
if __name__ == "__main__":
    print("\n=== Hand-Cricket BOT vs BOT – FULL MATCH ===\n")
    print("Profiles available :", ", ".join(player_profiles.keys()))

    p1 = input("Profile for Team A (bats first): ").strip().lower() or "aggressive"
    p2 = input("Profile for Team B: ").strip().lower() or "cautious"
    if p1 not in player_profiles: p1="aggressive"
    if p2 not in player_profiles: p2="cautious"

    try:
        n_matches = int(input("How many matches to simulate? ") or "5")
    except ValueError:
        n_matches = 5

    try:
        overs = int(input("Max turns/innings (default 30): ") or "30")
    except ValueError:
        overs = 30

    all_matches=[]
    for m in range(1,n_matches+1):
        df = simulate_match(p1,p2,max_turns=overs)
        df.insert(0,"Match",m)
        all_matches.append(df)
        print(f"\n--- Match {m} complete ---")
        print(df.tail(3))

    big = pd.concat(all_matches, ignore_index=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"hand_cricket_fullmatch_{ts}.csv"
    big.to_csv(fname,index=False)
    print("\nSaved complete log to", fname)
