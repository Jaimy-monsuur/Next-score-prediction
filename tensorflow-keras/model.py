from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# Your data
players = [ [[4,4,5,6,4],[6,6,3,3,4],[5,5,5,6,6],[4,5,6,7,8],[6,6,7,6,8]],
            [[7,7,6,7,7],[7,7,6,6,5],[7,6,6,6,5],[6,6,6,5,5],[7,4,5,5,4]],
            [[5,5,5,5,5],[6,6,6,6,6],[4,4,4,4,4],[7,7,7,7,7],[3,3,3,3,3]],
            [[4,5,2,4,1],[5,6,3,3,4],[6,5,6,7,6],[8,7,9,7,10],[9,8,6,8,6]],
            [[4,4,5,6,4],[6,9,3,3,4],[5,5,8,6,6],[9,5,6,7,8],[8,6,7,6,8]]
]

# Compute the total scores
totals = []
for player in players:
    total_player = []
    for scores in player:
        total_player.append(sum(scores))
    totals.append(total_player)

# Create the input and output data for the neural network model
X = np.array(players[0]).reshape((5, 5))
y = np.array(totals[0]).reshape((5, 1))

# Create a neural network model and fit it to the data
model = Sequential()
model.add(Dense(12, input_dim=5, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam')
model.fit(X, y, epochs=1000, batch_size=5,)

# Save the model as a single file
model.save('my_model.h5')
