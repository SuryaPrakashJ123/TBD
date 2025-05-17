
import random
import pandas as pd

def simulate_innings(batsman_profile, bowler_profile, target=None):
    log = []
    runs = 0
    turn = 1
    is_out = False
    history = []

    for bat, bowl in zip(batsman_profile, bowler_profile):
        outcome = "RUN"
        run_scored = bat if bat != bowl else 0
        is_out = bat == bowl

        repeated = "Yes" if history and bat == history[-1] else "No"
        surprise = "Yes" if history and bat != history[-1] and history.count(bat) == 0 else "No"
        strategy = "Bluff"

        history.append(bat)
        if len(history) > 3:
            history.pop(0)

        log.append({
            "Turn": turn,
            "Batsman_Input": bat,
            "Bowler_Input": bowl,
            "Outcome": "OUT" if is_out else "RUN",
            "Runs_Scored": run_scored,
            "Is_Out": is_out,
            "Batsman_Strategy": strategy,
            "Repeated?": repeated,
            "Surprise_Move": surprise
        })

        if is_out or (target is not None and runs + run_scored >= target):
            runs += run_scored
            break
        runs += run_scored
        turn += 1

    return log, runs

def generate_profile(style, length=100):
    if style == "aggressive":
        return [random.choice([6, 6, 4, 5, 6, 2]) for _ in range(length)]
    elif style == "cautious":
        return [random.choice([1, 2, 3, 2, 1, 3]) for _ in range(length)]
    elif style == "deceptive":
        return [random.choice([2, 6, 2, 6, 3, 6]) for _ in range(length)]
    else:  # random
        return [random.randint(1, 6) for _ in range(length)]

def simulate_match():
    bat1 = generate_profile("deceptive")
    bowl1 = generate_profile("random")
    innings1_log, innings1_score = simulate_innings(bat1, bowl1)

    bat2 = generate_profile("random")
    bowl2 = generate_profile("deceptive")
    innings2_log, innings2_score = simulate_innings(bat2, bowl2, target=innings1_score)

    match_log = innings1_log + [{"Turn": "—"}] + innings2_log
    result = "Tie"
    if innings2_score > innings1_score:
        result = "Bot 2 Wins"
    elif innings2_score < innings1_score:
        result = "Bot 1 Wins"

    return match_log, innings1_score, innings2_score, result

if __name__ == "__main__":
    match_count = int(input("Enter number of matches to simulate: "))
    all_logs = []
    for match_id in range(1, match_count + 1):
        match_log, score1, score2, result = simulate_match()
        for entry in match_log:
            entry["Match_ID"] = match_id
        all_logs.extend(match_log)

    df = pd.DataFrame(all_logs)
    df.to_csv("bot_vs_bot_match_logs.csv", index=False)
    print(f"Simulation complete. Logs saved to bot_vs_bot_match_logs.csv.")
