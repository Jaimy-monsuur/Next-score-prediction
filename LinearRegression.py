from sklearn.linear_model import LinearRegression


def predict_next_score(player, name):
    # Compute the total scores
    totals = [sum(x) for x in player]

    # Create the input and output data for the regression model
    X = [[i] for i in range(len(totals))]
    y = totals

    # Create a Linear Regression model and fit it to the data
    model = LinearRegression().fit(X, y)

    # Predict the next total score
    next_total = model.predict([[len(totals)]])
    next_total = round(next_total[0])
    print(f"Next possible score for {name}: {next_total}")


# Data for player 1 with improved performance
player1 = [[4, 4, 5, 6, 4], [6, 6, 3, 3, 4], [5, 5, 5, 6, 6], [4, 5, 6, 7, 8], [6, 6, 7, 6, 8]]

# Data for player 2 with deteriorating performance
player2 = [[7, 7, 6, 7, 7], [7, 7, 6, 6, 5], [7, 6, 6, 6, 5], [6, 6, 6, 5, 5], [5, 4, 5, 5, 4]]

# Data for player 3 with inconsistent performance
player3 = [[5, 5, 5, 5, 5], [6, 6, 6, 6, 6], [4, 4, 4, 4, 4], [7, 7, 7, 7, 7], [3, 3, 3, 3, 3]]


# Predict the next possible score for player 1
predict_next_score(player1, "Player 1")

# Predict the next possible score for player 2
predict_next_score(player2, "Player 2")

# Predict the next possible score for player 3
predict_next_score(player3, "Player 3")
