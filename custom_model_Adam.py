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

    # Sigmoid function
    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def ReLU(self, Z):
        return np.maximum(Z, 0)
        
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
    def __init__(self, alpha, optimizer, model, gamma):
        self.alpha = alpha
        self.optimizer = optimizer
        self.model = model
        self.gamma = gamma

    def ReLU_deriv(self, Z):
        return Z > 0
    

    # sigmoid
    def sigmoid_prime(self, Z):
        return Z * (1 - Z)

    def backward_prop(self, Z1, A1, Z2, W2, state, actual_Q, sample_size, sample_size_counter):

        # if want to use the actual batch size + ST steps
        if sample_size >= 1000:
            sample_size = 1000
            sample_size = sample_size +sample_size_counter


        # # if want to refresh after every LT
        # # sample_size = sample_size_counter

        dZ2 = 0
        dW2 = 0
        db2 = 0 
        dA1 = 0
        dZ1 = 0
        dW1 = 0
        db1 = 0
        

        # According to ChatGPT
        dZ2 = (-2 / sample_size) * (actual_Q - Z2)
        dW2 = np.dot(dZ2, A1.T)
        db2 = np.sum(dZ2, axis=1, keepdims=True)
        dA1 = np.dot(W2.T, dZ2)
        dZ1 = dA1 * self.ReLU_deriv(Z1)
        dW1 = np.dot(dZ1, state.T)
        db1 = np.sum(dZ1, axis=1, keepdims=True)

        # Originally here
        # dZ2 = Z2 - actual_Q
        # dW2 = 1 / sample_size * dZ2.dot(A1.T)
        # db2 = 1 / sample_size * np.sum(dZ2)
        # dZ1 = W2.T.dot(dZ2) * self.ReLU_deriv(Z1)
        # dW1 = 1 / sample_size * dZ1.dot(state.T)
        # db1 = 1 / sample_size * np.sum(dZ1)



        # Self derived using video 
        # loss = 1/2 * (actual_Q-Z2) ** 2
        # dZ2 = -1 * (actual_Q-Z2)
        # dW2 = np.dot(dZ2, A1.T)
        # db2 = -(actual_Q - Z2)
        # dZ1 = W2.T.dot(dZ2) * self.ReLU_deriv(Z1)
        # dW1 = np.dot(dZ1, state.T)
        # db1 = dZ1        
        # dW2 = (-2 * A1 *(actual_Q - Z2).T).T
        # db2 = -2 * (actual_Q - Z2)


        # print("shape Z2:", Z2.shape)
        # print("shape Q:", actual_Q.shape)
        # print("shape W2.T:", W2.T.shape)
        # print("shape Relu(Z1):", self.ReLU_deriv(Z1).shape)
        # print("shape state:", state.shape)
        # print("shape A1:", A1.shape)

        # print("success")

        # print("dW1: ", dW1.shape)
        # print("db1: ", db1.shape)
        # print("dW2: ", dW2.shape)
        # print("db2: ", db2.shape)

        return dW1, db1, dW2, db2

    def update_params(self, W1, b1, W2, b2, dW1, db1, dW2, db2, m_t_W1, v_t_W1, m_t_b1, v_t_b1, m_t_W2, v_t_W2, m_t_b2, v_t_b2, t):
        # W1 = W1 - self.alpha * dW1
        # b1 = b1 - self.alpha * db1    
        # W2 = W2 - self.alpha * dW2  
        # b2 = b2 - self.alpha * db2  
        W1, m_t_W1, v_t_W1 = self.optimizer.optimize(W1, dW1, m_t_W1, v_t_W1)
        b1, m_t_b1, v_t_b1 = self.optimizer.optimize(b1, db1, m_t_b1, v_t_b1)
        W2, m_t_W2, v_t_W2 = self.optimizer.optimize(W2, dW2, m_t_W2, v_t_W2)
        b2, m_t_b2, v_t_b2  = self.optimizer.optimize(b2, db2, m_t_b2, v_t_b2)
        t += 1
        return W1, b1, W2, b2, m_t_W1, v_t_W1, m_t_b1, v_t_b1, m_t_W2, v_t_W2, m_t_b2, v_t_b2, t 
    

    def train_step(self, W1, b1, W2, b2, state, action, reward, next_state, done, counter, sample_size_counter, m_t_W1, v_t_W1, m_t_b1, v_t_b1, m_t_W2, v_t_W2, m_t_b2, v_t_b2, t):
        
        
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
        W1, b1, W2, b2, m_t_W1, v_t_W1, m_t_b1, v_t_b1, m_t_W2, v_t_W2, m_t_b2, v_t_b2, t = self.update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, m_t_W1, v_t_W1, m_t_b1, v_t_b1, m_t_W2, v_t_W2, m_t_b2, v_t_b2, t)

        return W1, b1, W2, b2, m_t_W1, v_t_W1, m_t_b1, v_t_b1, m_t_W2, v_t_W2, m_t_b2, v_t_b2, t 


class AdamOptimizer:
    def __init__(self, alpha):
        self.alpha = alpha
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.m_t_W1 = None
        self.v_t_W1 = None
        self.m_t_b1 = None
        self.v_t_b1 = None
        self.m_t_W2 = None
        self.v_t_W2 = None
        self.m_t_b2 = None
        self.v_t_b2 = None
        self.t = 0


    def optimize(self, param, grad, m_t, v_t):

        if m_t is None:
            # Initialize m_t and v_t as arrays like param filled with zeros
            m_t = np.zeros_like(param)
            v_t = np.zeros_like(param)

        # Update biased first moment estimate
        m_t = self.beta1 * m_t + (1 - self.beta1) * grad

        # Update biased second moment estimate
        v_t = self.beta2 * v_t + (1 - self.beta2) * (grad ** 2)

        m_t_hat = m_t / (1 - self.beta1 ** self.t)
        v_t_hat = v_t / (1 - self.beta2 ** self.t)

        # difference = (self.alpha / (np.sqrt(v_t_hat) + self.epsilon)) * m_t_hat
        param = param - self.alpha * (m_t_hat / (np.sqrt(v_t_hat) + self.epsilon))

        return param, m_t, v_t 