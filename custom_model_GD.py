
#agent.remember is used to store the current state, action taken, reward received, the next state, and whether the game is done.
#agent.remember returns state, action, reward, next_state, done.
#agent.get_state runs infinitely and it remembers after each step.
#train_short_memory is called after each step to update the model based on the most recent experience.
#train_long_memory is called after each game over. Samples smaller than 100,000 is stored in full. >100,000 is stored in random sample
#The small sample stores all states together, actions together, rewards together, etc.


import numpy as np
import os
import json

# Forward Prop
class QModel:
    def __init__(self, input_size, hidden_size, output_size):
       self.input_size = input_size
       self.hidden_size = hidden_size
       self.output_size = output_size
        
    def init_params(self):
        np.random.seed(42)

        W1 = np.random.rand(self.hidden_size, self.input_size) - 0.5
        # The column value for b1 should be m to accomodate m samples, but NumPy will automatically expand to m from 1.
        # So it can be initialised as 1 without issue. Broadcasting will occur when Z1 is calculated later. Same for b2.
        b1 = np.random.rand(self.hidden_size, 1) - 0.5
        W2 = np.random.rand(self.output_size, self.hidden_size) - 0.5
        b2 = np.random.rand(self.output_size, 1) - 0.5
        return W1, b1, W2, b2


    # ReLU function
    def ReLU(self, Z):
        return np.maximum(Z, 0)
    
    # Sigmoid function
    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))
        
    def forward_prop(self, W1, b1, W2, b2, state):

        #Handling single experiences
        if len(state.shape) == 1:
            state = state.reshape(-1, 1)

        Z1 = W1.dot(state) + b1
        A1 = self.ReLU(Z1)
        Z2 = W2.dot(A1) + b2
        return Z1, A1, Z2 # only Z2 for the Qvalues of straight, right and left.
    
    def save(self, W1, b1, W2, b2):
        file_name='snake_model.json'
        model_params = {
            'W1': W1.tolist(),
            'b1': b1.tolist(),
            'W2': W2.tolist(),
            'b2': b2.tolist()
        }

        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        
        with open(file_name, 'w') as file:
            json.dump(model_params, file)

        print(f"Model saved to {file_name}")

class QTrainer:
    def __init__(self, alpha, model, gamma):
        self.alpha = alpha
        self.model = model
        self.gamma = gamma

    # sigmoid
    def sigmoid_prime(self, Z):
        return Z * (1 - Z)

    # ReLU
    def ReLU_deriv(self, Z):
        return Z > 0
        

    def backward_prop(self, Z1, A1, Z2, W2, state, actual_Q, counter, sample_size_counter):
        
        if counter >= 10000:
            counter = 1000
            counter = counter+sample_size_counter

        #According to ChatGPT
        dZ2 = (-2 / counter) * (actual_Q - Z2)
        dW2 = np.dot(dZ2, A1.T)
        db2 = np.sum(dZ2, axis=1, keepdims=True)
        dA1 = np.dot(W2.T, dZ2)
        dZ1 = dA1 * self.ReLU_deriv(Z1)
        dW1 = np.dot(dZ1, state.T)
        db1 = np.sum(dZ1, axis=1, keepdims=True)

        # # original
        # dZ2 = 1/counter * (Z2 - actual_Q)
        # # dZ2 = Z2 - actual_Q
        # dW2 = 1 / counter * dZ2.dot(A1.T)
        # db2 = 1 / counter * np.sum(dZ2)
        # dZ1 = W2.T.dot(dZ2) * self.ReLU_deriv(Z1)
        # dW1 = 1 / counter * dZ1.dot(state.T)
        # db1 = 1 / counter * np.sum(dZ1)
        return dW1, db1, dW2, db2

    def update_params(self,W1, b1, W2, b2, dW1, db1, dW2, db2):
        W1 = W1 - self.alpha * dW1
        b1 = b1 - self.alpha * db1    
        W2 = W2 - self.alpha * dW2  
        b2 = b2 - self.alpha * db2    
        return W1, b1, W2, b2
    
    def train_step(self, W1, b1, W2, b2, state, action, reward, next_state, done, counter, sample_size_counter):
        
        
        #Convert done from bool to tuple if it is singular (train_short_term)
        if (type(done) == bool):
            done = (done, )

        # Handle batch experience case (Train long)
        if(len(done) > 1):
            state = np.array(state).T
            next_state = np.array(next_state).T
            action = np.array(action)
            reward = np.array(reward)
            

        # Single experience case (Train short)
        if (len(done) == 1):
            action = [action]
            reward = [reward]
            # Convert your variables to NumPy arrays if they are not already
            state = state.astype(np.int64)
            next_state = np.array([next_state], dtype=np.int64)
            action = np.array(action, dtype=np.int64)  # Use int64 for long data type
            reward = np.array(reward, dtype=np.float32)

            next_state = next_state.T

            # This part is just meant to ensure that it is constantly prepared to 
            # handle as a batch.
            if len(state.shape) == 1:
                state = state.reshape(-1, 1)



        # 1: predicted Q values with current state
        Z1, A1, Z2 = self.model.forward_prop(W1, b1, W2, b2, state)
        pred = Z2

        # 2: Calculate actual Q values with reward
        target = pred.copy()
        for idx in range(len(done)): 

            # 2.a Game over event, Q = actual reward:
            Q_new = reward[idx]

            # 2.b Game continuing event, Q = actual reward + expected max next reward:
            if not done[idx]:
                Z1_next, A1_next, Z2_next = self.model.forward_prop(W1, b1, W2, b2, next_state)
                Q_new = reward[idx] + self.gamma * np.max(Z2_next)

            # For the actual move selected, update the Q value.
            target[np.argmax(action[idx]).item()] = Q_new

        # # 3: Calculate the error and perform back propagation
        dW1, db1, dW2, db2 = self.backward_prop(Z1, A1, Z2, W2, state, target, counter, sample_size_counter)
        W1, b1, W2, b2 = self.update_params(W1, b1, W2, b2, dW1, db1, dW2, db2)

        return W1, b1, W2, b2



