import random
import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter, defaultdict
import math

# Player profiles with weighted probabilities for numbers
player_profiles = {
    "aggressive": {1: 0.05, 2: 0.1, 3: 0.15, 4: 0.2, 5: 0.2, 6: 0.3},
    "cautious": {1: 0.3, 2: 0.25, 3: 0.2, 4: 0.15, 5: 0.05, 6: 0.05},
    "deceptive": {1: 0.15, 2: 0.25, 3: 0.05, 4: 0.05, 5: 0.2, 6: 0.3},
    "random": {1: 1/6, 2: 1/6, 3: 1/6, 4: 1/6, 5: 1/6, 6: 1/6},
    "game_theory": {1: 0.4, 2: 0.3, 3: 0.1, 4: 0.1, 5: 0.03, 6: 0.07},  # Nash equilibrium-inspired
    "adaptive": {1: 1/6, 2: 1/6, 3: 1/6, 4: 1/6, 5: 1/6, 6: 1/6}  # Start with uniform distribution
}

# Counter profiles - what works best against each strategy
counter_profiles = {
    "aggressive": "cautious",    # Use cautious against aggressive to match their high numbers
    "cautious": "aggressive",    # Use aggressive against cautious to score more runs
    "deceptive": "game_theory",  # Use equilibrium strategy against deception
    "random": "deceptive",       # Use deception against randomness
    "game_theory": "aggressive", # Use aggressive against game theory optimality
    "adaptive": "deceptive",     # Default counter to adaptive
    "pattern_based": "random"    # Use randomness against pattern recognition
}

# Function to classify strategy
def classify_strategy(profile_name, input_num):
    if profile_name == "aggressive" and input_num >= 4:
        return "Risky_High"
    elif profile_name == "cautious" and input_num <= 3:
        return "Safe"
    elif profile_name == "deceptive":
        return "Bluff"
    elif profile_name == "adaptive":
        return "Adaptive"
    elif profile_name == "game_theory":
        return "Optimal"
    elif profile_name == "pattern_based":
        return "Predictive"
    else:
        return "Normal"

# Advanced pattern recognition using n-grams and frequency analysis
class PatternRecognizer:
    def __init__(self, history_window=10, ngram_sizes=[1, 2, 3]):
        self.history_window = history_window
        self.ngram_sizes = ngram_sizes
        self.ngram_counters = {n: defaultdict(Counter) for n in ngram_sizes}
        self.overall_counters = Counter()
        
    def update(self, history):
        """Update pattern counters with new history"""
        # Only use recent history
        recent_history = history[-self.history_window:] if len(history) > self.history_window else history
        
        # Update overall frequency counter
        self.overall_counters.update(recent_history)
        
        # Update n-gram counters
        for n in self.ngram_sizes:
            if len(recent_history) >= n:
                for i in range(len(recent_history) - n + 1):
                    context = tuple(recent_history[i:i+n-1])
                    next_move = recent_history[i+n-1]
                    self.ngram_counters[n][context][next_move] += 1
    
    def predict_next_move(self, history, randomness=0.2):
        """Predict the next move based on patterns in history"""
        if not history:
            return None, 0.0
            
        recent_history = history[-self.history_window:] if len(history) > self.history_window else history
        
        # Weight predictions from different n-gram sizes
        predictions = {}
        confidence_scores = {}
        
        # Check different n-gram sizes, prioritizing larger patterns
        for n in sorted(self.ngram_sizes, reverse=True):
            if len(recent_history) >= n-1:
                context = tuple(recent_history[-(n-1):])
                if context in self.ngram_counters[n]:
                    counter = self.ngram_counters[n][context]
                    total = sum(counter.values())
                    
                    # Only consider if we have enough occurrences
                    if total >= 2:
                        most_common = counter.most_common(1)[0]
                        predictions[n] = most_common[0]
                        confidence_scores[n] = most_common[1] / total
        
        # If we have any predictions
        if predictions:
            # Prioritize highest n-gram with reasonable confidence
            for n in sorted(predictions.keys(), reverse=True):
                if confidence_scores[n] > 0.5:  # Threshold for confidence
                    return predictions[n], confidence_scores[n]
            
            # Fall back to most frequent overall move if no strong pattern
            if self.overall_counters:
                most_common = self.overall_counters.most_common(1)[0]
                # Add some randomness to prevent being too predictable
                if random.random() > randomness:
                    return most_common[0], most_common[1]/sum(self.overall_counters.values())
        
        # No strong pattern found
        return None, 0.0

# Game-theoretic optimal strategy calculator
class GameTheoryOptimizer:
    def __init__(self):
        # Expected runs calculator for batsman (assuming random bowler)
        self.batsman_ev = {
            1: 1*(5/6),  # 1 run with 5/6 probability of not getting out
            2: 2*(5/6),  # 2 runs with 5/6 probability of not getting out
            3: 3*(5/6),  # 3 runs with 5/6 probability of not getting out
            4: 4*(5/6),  # 4 runs with 5/6 probability of not getting out
            5: 5*(5/6),  # 5 runs with 5/6 probability of not getting out
            6: 6*(5/6),  # 6 runs with 5/6 probability of not getting out
        }
        
        # Initialize a basic game state
        self.reset_state()
    
    def reset_state(self):
        self.current_runs = 0
        self.current_turn = 1
        self.is_final_turns = False
        self.target = None
        
    def update_state(self, runs, turn, target=None):
        """Update the game state for decision making"""
        self.current_runs = runs
        self.current_turn = turn
        self.target = target
        
        # Determine if we're in final turns mode (e.g., close to target or end)
        if target and (target - runs <= 12):
            self.is_final_turns = True
    
    def batsman_strategy(self, bowler_history=None, risk_profile=0.5):
        """Calculate optimal batsman strategy based on game state"""
        # If we have a target and are close to it, use specialized strategy
        if self.is_final_turns and self.target:
            runs_needed = self.target - self.current_runs
            
            if runs_needed <= 6:
                # Just need the exact amount - focus on that number
                return {runs_needed: 0.8, 
                        max(1, runs_needed-1): 0.1, 
                        min(6, runs_needed+1): 0.1}
            
            # Closer to target, favor higher numbers
            return {
                1: max(0.05, 0.3 - 0.05*min(5, runs_needed//2)),
                2: max(0.05, 0.25 - 0.03*min(5, runs_needed//2)),
                3: 0.1 + 0.02*min(5, runs_needed//3),
                4: 0.15 + 0.03*min(5, runs_needed//4),
                5: 0.2 + 0.03*min(5, runs_needed//5),
                6: 0.2 + 0.05*min(4, runs_needed//6)
            }
        
        # Normal play - calculate based on expected value and risk profile
        # Higher risk_profile means more aggressive (favoring high numbers)
        weights = {}
        total_weight = 0
        
        for num in range(1, 7):
            # Base weight is expected value
            weight = self.batsman_ev[num]
            
            # Adjust for risk profile
            if num >= 4:
                weight *= (1 + risk_profile)  # Boost high numbers for aggressive play
            else:
                weight *= (2 - risk_profile)  # Boost low numbers for cautious play
            
            weights[num] = weight
            total_weight += weight
        
        # Normalize to probabilities
        return {num: weight/total_weight for num, weight in weights.items()}
    
    def bowler_strategy(self, batsman_history=None, aggression_profile=0.5):
        """Calculate optimal bowler strategy based on game state and batsman history"""
        # Default mixed strategy (slightly favoring lower numbers)
        strategy = {1: 0.2, 2: 0.2, 3: 0.2, 4: 0.15, 5: 0.15, 6: 0.1}
        
        # If we have batsman history, analyze it
        if batsman_history and len(batsman_history) >= 3:
            recent = batsman_history[-5:]
            counter = Counter(recent)
            most_common = counter.most_common()
            
            # If there's a clear preference, target it more
            if most_common and most_common[0][1] >= 2:
                favorite_num = most_common[0][0]
                # Increase probability of matching this number
                strategy = {n: 0.1 for n in range(1, 7)}  # Reset to flat
                strategy[favorite_num] = 0.5  # Focus on their favorite number
                
                # Also target numbers close to their favorite
                for offset in [-1, 1]:
                    neighbor = favorite_num + offset
                    if 1 <= neighbor <= 6:
                        strategy[neighbor] = 0.2
        
        # Adjust for game state if in final turns
        if self.is_final_turns and self.target:
            runs_needed = self.target - self.current_runs
            
            if runs_needed <= 6:
                # They likely need exactly this many runs, so focus on this number
                strategy = {n: 0.05 for n in range(1, 7)}
                strategy[runs_needed] = 0.75
            
        # Apply aggression profile
        if aggression_profile > 0.5:  # Aggressive: target high numbers
            for n in [4, 5, 6]:
                strategy[n] *= (1 + 0.5*(aggression_profile-0.5))
        else:  # Conservative: target low numbers
            for n in [1, 2, 3]:
                strategy[n] *= (1 + 0.5*(0.5-aggression_profile))
                
        # Normalize
        total = sum(strategy.values())
        return {n: p/total for n, p in strategy.items()}

# Reinforcement learning inspired adaptive strategy
class AdaptiveRL:
    def __init__(self, learning_rate=0.1, exploration_rate=0.2, discount_factor=0.9):
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.discount_factor = discount_factor
        self.q_values = defaultdict(lambda: {n: 0 for n in range(1, 7)})
        self.last_state = None
        self.last_action = None
        
    def choose_action(self, state, role, opponent_history=None):
        """Choose an action using epsilon-greedy policy"""
        # Exploration: random choice
        if random.random() < self.exploration_rate:
            action = random.randint(1, 6)
            return action
        
        # Exploitation: best known action
        q_values = self.q_values[state]
        max_value = max(q_values.values())
        best_actions = [a for a, v in q_values.items() if v == max_value]
        
        # If multiple best actions, choose randomly among them
        return random.choice(best_actions)
    
    def update_q_value(self, state, action, reward, next_state=None):
        """Update Q-value for a state-action pair"""
        current_q = self.q_values[state][action]
        
        # If we have a next state, use Q-learning update
        if next_state:
            max_next_q = max(self.q_values[next_state].values())
            new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        else:
            # Otherwise, just update based on reward (for terminal states)
            new_q = current_q + self.learning_rate * (reward - current_q)
        
        self.q_values[state][action] = new_q
    
    def encode_state(self, game_state, opponent_history):
        """Create a simplified state representation"""
        # Use last 2 opponent moves as state representation
        last_moves = tuple(opponent_history[-2:]) if len(opponent_history) >= 2 else (0, 0)
        
        # Add game phase (early, mid, late)
        turn = game_state.get('turn', 0)
        phase = 'early' if turn < 5 else 'mid' if turn < 15 else 'late'
        
        # Add score situation if target exists
        score_state = 'normal'
        if 'target' in game_state and game_state['target']:
            runs = game_state.get('runs', 0)
            target = game_state['target']
            if target - runs <= 6:
                score_state = 'close'
            elif target - runs <= 12:
                score_state = 'approaching'
        
        return (last_moves, phase, score_state)

# Function to select a number based on a probability distribution
def choose_from_distribution(distribution):
    numbers = list(distribution.keys())
    probabilities = list(distribution.values())
    return random.choices(numbers, weights=probabilities, k=1)[0]

# Function to analyze opponent's pattern and create a counter strategy
def adapt_to_opponent(opponent_history, role='batsman', game_state=None, 
                      pattern_recognizer=None, rl_agent=None, game_theory=None):
    if not opponent_history:
        return random.randint(1, 6)
    
    # 1. Use pattern recognition for prediction
    predicted_move = None
    confidence = 0
    
    if pattern_recognizer:
        pattern_recognizer.update(opponent_history)
        predicted_move, confidence = pattern_recognizer.predict_next_move(opponent_history)
    
    # 2. Use game theory for strategic decisions
    if game_theory and game_state:
        game_theory.update_state(
            game_state.get('runs', 0), 
            game_state.get('turn', 1),
            game_state.get('target', None)
        )
        
        if role == 'batsman':
            strategy_dist = game_theory.batsman_strategy(opponent_history)
        else:  # bowler
            strategy_dist = game_theory.bowler_strategy(opponent_history)
        
        # Combine prediction with game theory if we have a confident prediction
        if predicted_move and confidence > 0.6:
            # As batsman, avoid the predicted bowler's number
            if role == 'batsman':
                # Remove the predicted number from our options
                if predicted_move in strategy_dist:
                    del strategy_dist[predicted_move]
                    # Renormalize
                    total = sum(strategy_dist.values())
                    strategy_dist = {n: p/total for n, p in strategy_dist.items()}
            
            # As bowler, target the predicted batsman's number
            else:
                # Focus on the predicted number
                strategy_dist = {n: 0.1 for n in range(1, 7)}
                strategy_dist[predicted_move] = 0.5
        
        return choose_from_distribution(strategy_dist)
    
    # 3. Use reinforcement learning if available
    if rl_agent and game_state:
        state = rl_agent.encode_state(game_state, opponent_history)
        action = rl_agent.choose_action(state, role, opponent_history)
        return action
    
    # 4. Fall back to simple frequency analysis
    recent = opponent_history[-5:]
    counter = Counter(recent)
    most_common = counter.most_common()
    
    if most_common and most_common[0][1] >= 2:
        # Found a pattern
        if role == 'batsman':
            # Avoid the most common number bowled
            avoid = most_common[0][0]
            return random.choice([n for n in range(1, 7) if n != avoid])
        else:
            # Match the most common number batted
            return most_common[0][0]
    
    # 5. No clear pattern, use a balanced approach
    return random.randint(1, 6)

# Function to detect patterns in player moves
def detect_pattern(history, threshold=0.6):
    if len(history) < 5:
        return False
    
    # Look at recent history
    recent = history[-5:]
    counter = Counter(recent)
    most_common = counter.most_common(1)
    
    # If a number appears more than threshold% of the time, it's a pattern
    if most_common and most_common[0][1] / len(recent) >= threshold:
        return True
    
    # Check for alternating patterns (like 1,6,1,6)
    if len(recent) >= 4:
        alternating = True
        for i in range(2, len(recent)):
            if recent[i] != recent[i-2]:
                alternating = False
                break
        if alternating:
            return True
            
    # Check for ascending or descending patterns
    is_ascending = all(recent[i] < recent[i+1] for i in range(len(recent)-1))
    is_descending = all(recent[i] > recent[i+1] for i in range(len(recent)-1))
    if is_ascending or is_descending:
        return True
    
    return False

# Modified: Simulate a single innings (batsman plays until out)
def simulate_innings(batsman_profile, bowler_profile, max_turns=50, adaptive=True, is_chasing=False, target=None):
    data = []
    batsman_history = []
    bowler_history = []
    
    # Initialize advanced components
    pattern_recognizer_batsman = PatternRecognizer(history_window=10, ngram_sizes=[1, 2, 3])
    pattern_recognizer_bowler = PatternRecognizer(history_window=10, ngram_sizes=[1, 2, 3])
    game_theory = GameTheoryOptimizer()
    rl_agent_batsman = AdaptiveRL(exploration_rate=0.15)
    rl_agent_bowler = AdaptiveRL(exploration_rate=0.15)
    
    # Current profiles that can change if adaptive is True
    current_batsman_profile = batsman_profile
    current_bowler_profile = bowler_profile
    
    total_runs = 0
    
    # Print header for innings data
    innings_type = "Chasing" if is_chasing else "First"
    print(f"\n{innings_type} Innings Detailed Results:")
    print(f"{'Turn':<5} {'Batsman_Input':<15} {'Bowler_Input':<15} {'Outcome':<10} {'Runs_Scored':<15} {'Total_Runs':<15} {'Is_Out':<10} {'Batsman_Strategy':<20} {'Bowler_Strategy':<20}")
    print("-" * 130)
    
    for i in range(1, max_turns + 1):
        game_state = {
            'turn': i,
            'runs': total_runs,
            'target': target,
            'is_chasing': is_chasing
        }
        
        # Select inputs based on current profiles and available tools
        if current_batsman_profile == "adaptive" and adaptive:
            # Advanced adaptive strategy for batsman
            batsman_input = adapt_to_opponent(
                bowler_history, 
                'batsman', 
                game_state,
                pattern_recognizer_bowler,
                rl_agent_batsman,
                game_theory
            )
        elif current_batsman_profile == "game_theory":
            # Use game theory optimizer for batsman
            game_theory.update_state(total_runs, i, target)
            strategy_dist = game_theory.batsman_strategy(bowler_history)
            batsman_input = choose_from_distribution(strategy_dist)
        elif current_batsman_profile == "pattern_based":
            # Use pattern recognition for batsman
            pattern_recognizer_bowler.update(bowler_history)
            predicted_move, confidence = pattern_recognizer_bowler.predict_next_move(bowler_history)
            if predicted_move and confidence > 0.5:
                # Avoid the predicted number
                batsman_input = random.choice([n for n in range(1, 7) if n != predicted_move])
            else:
                batsman_input = choose_from_distribution(player_profiles["random"])
        else:
            # Use standard profile distribution
            batsman_input = choose_from_distribution(player_profiles[current_batsman_profile])
            
        if current_bowler_profile == "adaptive" and adaptive:
            # Advanced adaptive strategy for bowler
            bowler_input = adapt_to_opponent(
                batsman_history, 
                'bowler', 
                game_state,
                pattern_recognizer_batsman,
                rl_agent_bowler,
                game_theory
            )
        elif current_bowler_profile == "game_theory":
            # Use game theory optimizer for bowler
            game_theory.update_state(total_runs, i, target)
            strategy_dist = game_theory.bowler_strategy(batsman_history)
            bowler_input = choose_from_distribution(strategy_dist)
        elif current_bowler_profile == "pattern_based":
            # Use pattern recognition for bowler
            pattern_recognizer_batsman.update(batsman_history)
            predicted_move, confidence = pattern_recognizer_batsman.predict_next_move(batsman_history)
            if predicted_move and confidence > 0.5:
                # Target the predicted number
                bowler_input = predicted_move
            else:
                bowler_input = choose_from_distribution(player_profiles["random"])
        else:
            # Use standard profile distribution
            bowler_input = choose_from_distribution(player_profiles[current_bowler_profile])

        is_out = batsman_input == bowler_input
        runs_scored = 0 if is_out else batsman_input
        total_runs += runs_scored
        outcome = "OUT" if is_out else "RUN"
        
        # For chasing innings, check if target is achieved
        target_achieved = False
        if is_chasing and target and total_runs >= target:
            target_achieved = True
        
        batsman_repeated = batsman_history[-1] == batsman_input if batsman_history else False
        surprise = batsman_repeated and is_out
        
        batsman_strategy = classify_strategy(current_batsman_profile, batsman_input)
        bowler_strategy = classify_strategy(current_bowler_profile, bowler_input)

        # Update history
        batsman_history.append(batsman_input)
        bowler_history.append(bowler_input)
        
        # Update RL agents with rewards
        if adaptive and i > 1:
            # Encode current and previous states
            prev_state = rl_agent_batsman.encode_state({'turn': i-1, 'runs': total_runs-runs_scored}, bowler_history[:-1])
            current_state = rl_agent_batsman.encode_state(game_state, bowler_history)
            
            # Update Q-values based on outcome
            batsman_reward = runs_scored - (10 if is_out else 0)  # Reward runs, heavily penalize getting out
            bowler_reward = -runs_scored + (10 if is_out else 0)  # Opposite rewards for bowler
            
            if rl_agent_batsman.last_state:
                rl_agent_batsman.update_q_value(
                    rl_agent_batsman.last_state, 
                    rl_agent_batsman.last_action, 
                    batsman_reward, 
                    prev_state
                )
            
            if rl_agent_bowler.last_state:
                rl_agent_bowler.update_q_value(
                    rl_agent_bowler.last_state, 
                    rl_agent_bowler.last_action, 
                    bowler_reward, 
                    prev_state
                )
            
            # Store current state and action for next update
            rl_agent_batsman.last_state = current_state
            rl_agent_batsman.last_action = batsman_input
            rl_agent_bowler.last_state = current_state
            rl_agent_bowler.last_action = bowler_input

        # Adapt strategy based on opponent's history if adaptive is enabled
        if adaptive and i > 3:  # Need some history first
            # Use pattern recognition to detect opponent strategy
            batsman_pattern_detected = detect_pattern(batsman_history, threshold=0.6)
            bowler_pattern_detected = detect_pattern(bowler_history, threshold=0.6)
            
            # Adapt profiles based on detected patterns and game state
            if bowler_pattern_detected:
                if "game_theory" in player_profiles:
                    current_batsman_profile = "game_theory"  # Use optimal strategy when pattern detected
                else:
                    current_batsman_profile = counter_profiles.get(bowler_profile, batsman_profile)
            
            if batsman_pattern_detected:
                if "pattern_based" in player_profiles:
                    current_bowler_profile = "pattern_based"  # Use pattern-based targeting
                else:
                    current_bowler_profile = counter_profiles.get(batsman_profile, bowler_profile)
                    
            # Late game strategy adjustments
            if is_chasing and target:
                remaining_runs = target - total_runs
                if remaining_runs <= 12:  # Getting close to target
                    # Batsman becomes more aggressive when close to target
                    current_batsman_profile = "aggressive"
                    # Bowler tries harder to get batsman out
                    current_bowler_profile = "pattern_based" if "pattern_based" in player_profiles else "aggressive"
        
        # Record the turn
        data.append({
            "Innings": "Chasing" if is_chasing else "First",
            "Turn": i,
            "Batsman_Input": batsman_input,
            "Bowler_Input": bowler_input,
            "Outcome": outcome,
            "Runs_Scored": runs_scored,
            "Total_Runs": total_runs,
            "Is_Out": is_out,
            "Target_Achieved": target_achieved,
            "Batsman_Strategy": batsman_strategy,
            "Bowler_Strategy": bowler_strategy,
            "Batsman_Profile": current_batsman_profile,
            "Bowler_Profile": current_bowler_profile,
            "Batsman_Repeated": batsman_repeated,
            "Surprise_Move": surprise
        })
        
        # Print turn details in a formatted table
        print(f"{i:<5} {batsman_input:<15} {bowler_input:<15} {outcome:<10} {runs_scored:<15} {total_runs:<15} {str(is_out):<10} {batsman_strategy:<20} {bowler_strategy:<20}")

        # Innings ends if batsman is out or target is achieved in chase
        if is_out or target_achieved:
            break

    return pd.DataFrame(data), total_runs, i

# New: Simulate a complete match with two innings
def simulate_match(player1_profile, player2_profile, max_turns=50, match_num=1, adaptive=True):
    print(f"\n===== MATCH {match_num}: {player1_profile.upper()} vs {player2_profile.upper()} =====")
    
    # Randomly decide who bats first
    first_batsman = random.choice([player1_profile, player2_profile])
    first_bowler = player2_profile if first_batsman == player1_profile else player1_profile
    
    print(f"\n--- First Innings: {first_batsman} batting, {first_bowler} bowling ---")
    
    # First innings
    first_innings_df, first_innings_score, first_innings_turns = simulate_innings(
        first_batsman, first_bowler, max_turns, adaptive, is_chasing=False
    )
    
    print(f"\nFirst Innings Summary: {first_batsman} scored {first_innings_score} runs in {first_innings_turns} turns")
    
    # Second innings (roles reversed and chasing target)
    second_batsman = first_bowler
    second_bowler = first_batsman
    
    print(f"\n--- Second Innings: {second_batsman} batting, {second_bowler} bowling (Target: {first_innings_score + 1}) ---")
    
    second_innings_df, second_innings_score, second_innings_turns = simulate_innings(
        second_batsman, second_bowler, max_turns, adaptive, 
        is_chasing=True, target=first_innings_score + 1
    )
    
    print(f"\nSecond Innings Summary: {second_batsman} scored {second_innings_score} runs in {second_innings_turns} turns")
    
    # Determine match result
    if second_innings_score >= first_innings_score + 1:
        winner = second_batsman
        margin = "by wicket" if "OUT" not in second_innings_df['Outcome'].values else f"by {second_innings_score - first_innings_score} runs"
    else:
        winner = first_batsman
        margin = f"by {first_innings_score - second_innings_score} runs"
    
    match_result = {
        "Match": match_num,
        "First_Batsman": first_batsman,
        "First_Bowler": first_bowler,
        "First_Innings_Score": first_innings_score,
        "First_Innings_Turns": first_innings_turns,
        "Second_Batsman": second_batsman,
        "Second_Bowler": second_bowler,
        "Second_Innings_Score": second_innings_score,
        "Second_Innings_Turns": second_innings_turns,
        "Winner": winner,
        "Margin": margin
    }
    
    # Print detailed match summary table
    print("\n----- MATCH SUMMARY -----")
    print(f"{'Player':<15} {'Role':<10} {'Score':<10} {'Turns':<10}")
    print("-" * 45)
    print(f"{first_batsman:<15} {'Batting':<10} {first_innings_score:<10} {first_innings_turns:<10}")
    print(f"{first_bowler:<15} {'Bowling':<10} {'-':<10} {first_innings_turns:<10}")
    print(f"{second_batsman:<15} {'Batting':<10} {second_innings_score:<10} {second_innings_turns:<10}")
    print(f"{second_bowler:<15} {'Bowling':<10} {'-':<10} {second_innings_turns:<10}")
    print("-" * 45)
    print(f"\nMatch Result: {winner} wins {margin}")
    
    # Combine both innings data
    first_innings_df['Match'] = match_num
    second_innings_df['Match'] = match_num
    match_data = pd.concat([first_innings_df, second_innings_df])
    
    return match_data, match_result

# New: Function to simulate multiple matches
def simulate_multiple_matches(player1_profile, player2_profile, num_matches, max_turns=50, adaptive=True):
    all_match_data = []
    match_results = []
    
    # Print header for the series summary
    print("\n===== SERIES SUMMARY =====")
    print(f"{'Match':<10} {'First Innings':<35} {'Second Innings':<35} {'Result':<40}")
    print("-" * 120)
    
    for match_num in range(1, num_matches + 1):
        match_df, result = simulate_match(player1_profile, player2_profile, max_turns, match_num, adaptive)
        all_match_data.append(match_df)
        match_results.append(result)
        
        # Print concise match summary for series
        first_innings = f"{result['First_Batsman']}: {result['First_Innings_Score']} in {result['First_Innings_Turns']} turns"
        second_innings = f"{result['Second_Batsman']}: {result['Second_Innings_Score']} in {result['Second_Innings_Turns']} turns"
        match_result = f"{result['Winner']} wins {result['Margin']}"
        print(f"{match_num:<10} {first_innings:<35} {second_innings:<35} {match_result:<40}")
    
    # Combine all match data
    combined_df = pd.concat(all_match_data, ignore_index=True)
    results_df = pd.DataFrame(match_results)
    
    return combined_df, results_df

# New: Function to analyze match results
def analyze_match_results(match_data_df, results_df):
    """Analyze the results of multiple matches"""
    print("\n===== MATCH ANALYSIS =====")
    
    # 1. Overall win statistics
    if len(results_df) > 0:
        print("\nWin Statistics:")
        win_counts = results_df['Winner'].value_counts()
        for player, count in win_counts.items():
            win_percentage = (count / len(results_df)) * 100
            print(f"{player}: {count} wins ({win_percentage:.1f}%)")
    
    # 2. Batting performance
    print("\nBatting Performance:")
    batting_stats = match_data_df.groupby(['Innings', 'Batsman_Profile']).agg(
        Total_Runs=('Runs_Scored', 'sum'),
        Avg_Runs_Per_Turn=('Runs_Scored', 'mean'),
        Max_Score=('Total_Runs', 'max'),
        Total_Turns=('Turn', 'count')
    ).sort_values('Total_Runs', ascending=False)
    print(batting_stats)
    
    # 3. Strategy effectiveness
    print("\nBatting Strategy Effectiveness:")
    strategy_stats = match_data_df.groupby('Batsman_Strategy').agg(
        Avg_Runs=('Runs_Scored', 'mean'),
        Out_Rate=('Is_Out', lambda x: x.sum() / len(x) if len(x) > 0 else 0),
        Usage_Count=('Batsman_Strategy', 'count')
    ).sort_values('Avg_Runs', ascending=False)
    print(strategy_stats)
    
    # 4. Performance in first vs. second innings
    print("\nPerformance by Innings:")
    innings_stats = match_data_df.groupby('Innings').agg(
        Avg_Score=('Total_Runs', lambda x: x.iloc[-1].mean() if len(x) > 0 else 0),
        Avg_Turns=('Turn', 'max'),
        Out_Rate=('Is_Out', lambda x: x.sum() / len(x) if len(x) > 0 else 0)
    )
    print(innings_stats)
    
    # 5. Target achievement rate in chases
    chasing_data = match_data_df[match_data_df['Innings'] == 'Chasing']
    if not chasing_data.empty:
        chase_success = chasing_data['Target_Achieved'].sum()
        chase_attempts = results_df.shape[0]  # Every match has one chase
        success_rate = (chase_success / chase_attempts) * 100
        print(f"\nChase Success Rate: {chase_success}/{chase_attempts} ({success_rate:.1f}%)")
    
    # 6. Number preferences
    print("\nNumber Selection Patterns:")
    for profile in match_data_df['Batsman_Profile'].unique():
        profile_data = match_data_df[match_data_df['Batsman_Profile'] == profile]
        num_distribution = profile_data['Batsman_Input'].value_counts(normalize=True)
        print(f"\n{profile} number selection:")
        for num in range(1, 7):
            pct = num_distribution.get(num, 0) * 100
            print(f"  {num}: {pct:.1f}%")
    
    # 7. Adaptivity analysis (if enabled)
    if 'adaptive' in match_data_df['Batsman_Profile'].values:
        print("\nAdaptivity Analysis:")
        adaptive_batsman = match_data_df[match_data_df['Batsman_Profile'] == 'adaptive']
        if not adaptive_batsman.empty:
            profile_changes = adaptive_batsman.groupby('Match')['Batsman_Profile'].nunique().mean()
            print(f"Average profile changes per match: {profile_changes:.2f}")

# Helper function to calculate entropy (randomness measure)
def calc_entropy(series):
    """Calculate Shannon entropy of a series"""
    value_counts = series.value_counts(normalize=True)
    return -sum(p * np.log2(p) for p in value_counts if p > 0)

# Helper function to visualize match results
def visualize_match_results(match_data, results_df):
    """Create visualizations for match analysis"""
    try:
        import matplotlib.pyplot as plt
        
        # Create timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Win distribution pie chart
        plt.figure(figsize=(10, 6))
        win_counts = results_df['Winner'].value_counts()
        plt.pie(win_counts, labels=win_counts.index, autopct='%1.1f%%', startangle=90)
        plt.title('Match Win Distribution')
        plt.tight_layout()
        plt.savefig(f"match_wins_{timestamp}.png")
        
        # 2. Innings score comparison
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        first_innings = match_data[match_data['Innings'] == 'First'].groupby('Match')['Total_Runs'].max()
        second_innings = match_data[match_data['Innings'] == 'Chasing'].groupby('Match')['Total_Runs'].max()
        
        x = np.arange(len(first_innings))
        width = 0.35
        
        plt.bar(x - width/2, first_innings, width, label='First Innings')
        plt.bar(x + width/2, second_innings, width, label='Second Innings')
        plt.xlabel('Match')
        plt.ylabel('Total Runs')
        plt.title('Innings Scores by Match')
        plt.legend()
        
        # 3. Number selection distribution
        plt.subplot(1, 2, 2)
        num_counts = match_data['Batsman_Input'].value_counts().sort_index()
        plt.bar(num_counts.index, num_counts.values)
        plt.xlabel('Number')
        plt.ylabel('Frequency')
        plt.title('Number Selection Distribution')
        plt.xticks(range(1, 7))
        
        plt.tight_layout()
        plt.savefig(f"match_statistics_{timestamp}.png")
        
        # 4. Run accumulation over turns (for a sample match)
        if len(results_df) > 0:
            sample_match = results_df['Match'].min()  # First match as sample
            match_data_sample = match_data[match_data['Match'] == sample_match]
            
            plt.figure(figsize=(10, 6))
            for innings in ['First', 'Chasing']:
                innings_data = match_data_sample[match_data_sample['Innings'] == innings]
                if not innings_data.empty:
                    plt.plot(innings_data['Turn'], innings_data['Total_Runs'], 
                             marker='o', label=f'{innings} Innings')
            
            plt.title(f'Run Accumulation (Match {sample_match})')
            plt.xlabel('Turn')
            plt.ylabel('Total Runs')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(f"match_runs_{timestamp}.png")
        
        print(f"\nVisualizations saved with timestamp: {timestamp}")
        
    except ImportError:
        print("Matplotlib not available. Install it for visualizations.")

# Main execution
if __name__ == "__main__":
    print("Advanced Bot vs Bot Hand Cricket Match Simulation")
    print("------------------------------------------")
    
    # Get available profiles
    available_profiles = list(player_profiles.keys())
    print("Available profiles:", ", ".join(available_profiles))
    print("\nProfile Descriptions:")
    print("- aggressive: Favors high numbers (4-6)")
    print("- cautious: Favors low numbers (1-3)")
    print("- deceptive: Alternates between low and high numbers unpredictably")
    print("- random: Completely random selection of numbers")
    print("- game_theory: Uses mathematical optimal strategy based on game theory")
    print("- adaptive: Advanced AI that adapts to opponent patterns")
    print("- pattern_based: Analyzes patterns to predict opponent's next move")
    
    # Get user inputs
    player1_profile = input(f"Enter Player 1 profile ({'/'.join(available_profiles)}): ").lower()
    player2_profile = input(f"Enter Player 2 profile ({'/'.join(available_profiles)}): ").lower()
    
    # Validate profiles
    if player1_profile not in player_profiles:
        print(f"Invalid Player 1 profile. Using 'adaptive' as default.")
        player1_profile = "adaptive"
    
    if player2_profile not in player_profiles:
        print(f"Invalid Player 2 profile. Using 'game_theory' as default.")
        player2_profile = "game_theory"
    
    # Get number of matches
    try:
        num_matches = int(input("Enter number of matches to simulate: "))
        if num_matches <= 0:
            raise ValueError("Number of matches must be positive")
    except ValueError:
        print("Invalid input. Using 3 matches as default.")
        num_matches = 3
    
    # Get max turns per innings
    try:
        max_turns = int(input("Enter maximum turns per innings (default 50): ") or "50")
        if max_turns <= 0:
            raise ValueError("Number of turns must be positive")
    except ValueError:
        print("Invalid input. Using 50 turns as default.")
        max_turns = 50
    
    # Ask if bots should adapt
    adaptive_input = input("Use advanced AI adaptation? (y/n, default: y): ").lower()
    adaptive = adaptive_input != 'n'
    
    # Run simulation
    print(f"\nSimulating {num_matches} matches of {player1_profile} vs {player2_profile}...")
    if adaptive:
        print("Bots will use advanced adaptive AI with game theory, pattern recognition, and reinforcement learning")
    else:
        print("Bots will use fixed strategies without adaptation")
        
    match_data, results = simulate_multiple_matches(
        player1_profile, player2_profile, num_matches, max_turns, adaptive
    )
    
    # Analyze results
    analyze_match_results(match_data, results)
    
    # Ask if user wants to visualize results
    viz_choice = input("\nGenerate visualizations? (y/n, default: y): ").lower()
    if viz_choice != 'n':
        visualize_match_results(match_data, results)
    
    # Save match data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    match_data_path = f"cricket_match_data_{timestamp}.csv"
    results_path = f"cricket_match_results_{timestamp}.csv"
    
    match_data.to_csv(match_data_path, index=False)
    results.to_csv(results_path, index=False)
    
    print(f"\nMatch data saved as: {match_data_path}")
    print(f"Match results saved as: {results_path}")
    
    # Print overall series winner
    if num_matches > 1:
        print("\n===== SERIES RESULT =====")
        series_wins = results['Winner'].value_counts()
        series_winner = series_wins.index[0] if len(series_wins) > 0 else None
        
        if series_winner:
            win_count = series_wins[series_winner]
            print(f"{series_winner} wins the series {win_count}-{num_matches - win_count}")
        else:
            print("Series ended in a draw")