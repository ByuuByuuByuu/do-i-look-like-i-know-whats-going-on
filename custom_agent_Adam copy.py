import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from custom_model_Adam_copy import QModel, QTrainer, AdamOptimizer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 100 # randomness
        self.gamma = 0.9
        self.alpha = 0.01
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = QModel(11, 256, 3)
        self.optimizer = AdamOptimizer(self.alpha)
        self.trainer = QTrainer(alpha = self.alpha, optimizer = self.optimizer, model = self.model, gamma = self.gamma)
        
    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)

    def initialise_param(self):
        return self.model.init_params()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self, W1, b1, W2, b2, counter, sample_size_counter, m_t_W1, v_t_W1, m_t_b1, v_t_b1, m_t_W2, v_t_W2, m_t_b2, v_t_b2, t):
        # if too many steps, then take a small sample of steps only
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples

        #if number of steps under 1000, then no need to random sample.
        else:
            mini_sample = self.memory

        #The "memory" object is holding all the rows of state, action, reward, etc.
        #the following zips it in preparation to send it for batch processing by the
        # "train_step"
        states, actions, rewards, next_states, dones = zip(*mini_sample)

        return self.trainer.train_step(W1, b1, W2, b2, states, actions, rewards, next_states, dones, counter, len(dones), m_t_W1, v_t_W1, m_t_b1, v_t_b1, m_t_W2, v_t_W2, m_t_b2, v_t_b2, t)



    def train_short_memory(self, W1, b1, W2, b2, state, action, reward, next_state, done, counter, sample_size_counter, m_t_W1, v_t_W1, m_t_b1, v_t_b1, m_t_W2, v_t_W2, m_t_b2, v_t_b2, t):
        return self.trainer.train_step(W1, b1, W2, b2, state, action, reward, next_state, done, counter, sample_size_counter, m_t_W1, v_t_W1, m_t_b1, v_t_b1, m_t_W2, v_t_W2, m_t_b2, v_t_b2, t)

    def get_action(self, W1, b1, W2, b2, current_state):
        # random moves: tradeoff exploration / exploitation
        threshold = self.epsilon - self.n_games
        finally_selected_move = [0,0,0]
        if random.randint(0, 200) < threshold:
            #0 for straight, 1 for right, 2 for left
            random_move_chosen_index = random.randint(0, 2)
            #Recording the one-hot-encoded move's direction chosen [0,0,1], etc.
            finally_selected_move[random_move_chosen_index] = 1
        else:
            #Given current state, makes a prediction using the model object
            Z1, A1, reward_prediction = self.model.forward_prop(W1, b1, W2, b2, current_state)
            # print(reward_prediction)
            #The most optimal move is selected using the maximum reward predicted.
            most_optimal_move = np.argmax(reward_prediction).item()

            #Recording the one-hot-encoded move's direction chosen [0,0,1], etc.
            finally_selected_move[most_optimal_move] = 1

        return finally_selected_move
    

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    W1, b1, W2, b2 = agent.initialise_param() 
    # W1old, b1old, W2old, b2old = W1, b1, W2, b2
    counter = 0
    sample_size_counter = 0


    while True:
        counter +=1
        sample_size_counter += 1
        current_state = agent.get_state(game)
        selected_move = agent.get_action(W1, b1, W2, b2, current_state)
        reward, done, score = game.play_step(selected_move)
        new_state = agent.get_state(game) 
        W1, b1, W2, b2, agent.optimizer.m_t_W1, agent.optimizer.v_t_W1, agent.optimizer.m_t_b1, agent.optimizer.v_t_b1, agent.optimizer.m_t_W2, agent.optimizer.v_t_W2, agent.optimizer.m_t_b2, agent.optimizer.v_t_b2, agent.optimizer.t = agent.train_short_memory(W1, b1, W2, b2, current_state, selected_move, reward, new_state, done, counter, sample_size_counter, agent.optimizer.m_t_W1, 
                                        agent.optimizer.v_t_W1, 
                                        agent.optimizer.m_t_b1, 
                                        agent.optimizer.v_t_b1, 
                                        agent.optimizer.m_t_W2, 
                                        agent.optimizer.v_t_W2, 
                                        agent.optimizer.m_t_b2, 
                                        agent.optimizer.v_t_b2, 
                                        agent.optimizer.t)
        agent.remember(current_state, selected_move, reward, new_state, done)
    
        if done:
            """
            Consider removing this counter reinitialisation, IE the sample size does not = 0 every
            time the game is over.
            """
            sample_size_counter = 0
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            W1, b1, W2, b2, agent.optimizer.m_t_W1, agent.optimizer.v_t_W1, agent.optimizer.m_t_b1, agent.optimizer.v_t_b1, agent.optimizer.m_t_W2, agent.optimizer.v_t_W2, agent.optimizer.m_t_b2, agent.optimizer.v_t_b2, agent.optimizer.t  = agent.train_long_memory(W1, b1, W2, b2, counter, sample_size_counter, agent.optimizer.m_t_W1, 
                                        agent.optimizer.v_t_W1, 
                                        agent.optimizer.m_t_b1, 
                                        agent.optimizer.v_t_b1, 
                                        agent.optimizer.m_t_W2, 
                                        agent.optimizer.v_t_W2, 
                                        agent.optimizer.m_t_b2, 
                                        agent.optimizer.v_t_b2, 
                                        agent.optimizer.t)
            # W1old, b1old, W2old, b2old = W1, b1, W2, b2
            # agent.train_long_memory(W1, b1, W2, b2)

            if score > record:
                record = score
                agent.model.save(W1,b1,W2,b2, agent.optimizer.m_t_W1, agent.optimizer.v_t_W1, agent.optimizer.m_t_b1, agent.optimizer.v_t_b1, agent.optimizer.m_t_W2, agent.optimizer.v_t_W2, agent.optimizer.m_t_b2, agent.optimizer.v_t_b2, agent.optimizer.t)

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()