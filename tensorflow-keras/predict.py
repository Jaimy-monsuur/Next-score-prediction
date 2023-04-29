from tensorflow.keras.models import load_model
import numpy as np

# Load the saved model
model = load_model('my_model.h5')

# New player data
new_player = [[4, 4, 5, 6, 4], [6, 6, 3, 3, 4], [5, 5, 5, 6, 6], [4, 5, 6, 7, 8], [6, 6, 7, 6, 8]]

# Compute the total scores
total_player = []
for scores in new_player:
    total_player.append(sum(scores))

# Create the input data for prediction
X_new = np.array(new_player).reshape((5, 5))

# Use the loaded model to predict the next total score
next_total = model.predict(X_new)
print("Next possible score: ", next_total[0][0])
