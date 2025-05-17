import random
import datetime

# Initialize log list
logs = []

def log_event(event):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logs.append(f"[{timestamp}] {event}")

def get_user_input():
    while True:
        try:
            user_input = int(input("Your move (1-6): "))
            if 1 <= user_input <= 6:
                return user_input
            else:
                print("Please enter a number between 1 and 6.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def bot_move():
    return random.randint(1, 6)

def play_innings(player_name, is_user_batting=True):
    score = 0
    history = []
    log_event(f"{player_name} starts {'batting' if is_user_batting else 'bowling'}.")
    while True:
        user = get_user_input() if is_user_batting else bot_move()
        bot = bot_move() if is_user_batting else get_user_input()

        if user == bot:
            log_event(f"{player_name} is OUT! (user: {user}, bot: {bot})")
            print(f"{player_name} is OUT! Final score: {score}\n")
            return score, history

        runs = user if is_user_batting else bot
        score += runs
        history.append((user, bot, runs))
        log_event(f"{player_name} played (user: {user}, bot: {bot}) -> runs: {runs}")
        print(f"{player_name} scored {runs}. Total: {score}")

def save_logs():
    filename = "hand_cricket_logs.txt"
    with open(filename, "a") as file:
        for log in logs:
            file.write(log + "\n")
    print(f"Game log saved to {filename}")

def play_game():
    print("Welcome to Hand Cricket!")
    log_event("Game started.")

    print("\nYour innings:")
    user_score, user_history = play_innings("User", is_user_batting=True)

    print("\nBot's innings:")
    bot_score, bot_history = play_innings("Bot", is_user_batting=False)

    log_event(f"Final Scores -> User: {user_score}, Bot: {bot_score}")

    if user_score > bot_score:
        print("You WIN!")
        log_event("Result: User WON.")
    elif user_score < bot_score:
        print("You LOSE!")
        log_event("Result: Bot WON.")
    else:
        print("It's a TIE!")
        log_event("Result: TIED.")

    save_logs()

if __name__ == "__main__":
    play_game()
