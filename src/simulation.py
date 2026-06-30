import numpy as np

def monte_carlo(fixtures, model, n_iter=5000):
    teams = set()
    for f in fixtures:
        teams.add(f["A"])
        teams.add(f["B"])

    wins = {t: 0 for t in teams}

    for _ in range(n_iter):
        for f in fixtures:
            probs = f["probs"]
            result = np.random.choice(model.classes_, p=probs)

            if result == "A":
                wins[f["A"]] += 1
            elif result == "B":
                wins[f["B"]] += 1

    return wins