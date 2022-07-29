from pyexpat import model
import gym
from gym import spaces
from gym import Env
import gym.spaces
import random 
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import dummy_vec_env  
import numpy as np
import cv2
import random
import time

N_DISCRETE_ACTIONS = 4
def collision_with_apple(apple_position, score):
    apple_position = [random.randrange(1,50)*10,random.randrange(1,50)*10]
    score += 1
    return apple_position, score

def collision_with_boundaries(snake_head):
    if snake_head[0]>=500 or snake_head[0]<0 or snake_head[1]>=500 or snake_head[1]<0 :
        return 1
    else:
        return 0

def collision_with_self(snake_position):
    snake_head = snake_position[0]
    if snake_head in snake_position[1:]:
        return 1
    else:
        return 0

class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    def __init__(self):
        super(CustomEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self.EAT_APPLE_REWARD = 100
        self.DEATH_REWARD = -10
        self.observation_space = spaces.Box(low=-500, high=500, shape=(12, ), dtype=np.float32)
    
    def step(self, action):
        cv2.imshow('a',self.img)
        cv2.waitKey(1)
        reward = 0

        #Going Close to apple Reward/Penalty
        self.apple_distance = (abs(self.snake_head[0] - self.apple_position[0]) + abs(self.snake_head[1] - self.apple_position[1]))
        if self.snake_prev_apple_distance > self.apple_distance:
            reward += 1
        else:
            reward -= 1
        self.snake_prev_apple_distance = self.apple_distance
        self.img = np.zeros((500,500,3),dtype='uint8')

        

        # Display Apple
        cv2.rectangle(self.img,(self.apple_position[0],self.apple_position[1]),(self.apple_position[0]+10,self.apple_position[1]+10),(0,0,255),3)

        # Display Snake
        for position in self.snake_position:
            cv2.rectangle(self.img,(position[0],position[1]),(position[0]+10,position[1]+10),(0,255,0),3)
        
        # Takes step after fixed time
        t_end = time.time()
        k = -1
        while time.time() < t_end:
            if k == -1:
                k = cv2.waitKey(1)
            else:
                continue    

        # Change the head position based on the button direction
        self.button_direction = action
        self.snake_going_down = 0
        self.snake_going_left = 0
        self.snake_going_right = 0
        self.snake_going_up = 0
        if self.button_direction == 1:
            self.snake_head[0] += 10
            self.snake_going_right = 1
        elif self.button_direction == 0:
            self.snake_head[0] -= 10
            self.snake_going_up = 1
        elif self.button_direction == 2:
            self.snake_head[1] -= 10
            self.snake_going_down = 1
        elif self.button_direction == 3:
            self.snake_head[1] += 10
            self.snake_going_left = 1

        # Increase Snake length on eating apple
        if self.snake_head == self.apple_position:
            self.apple_position, self.score = collision_with_apple(self.apple_position, self.score)
            self.snake_position.insert(0,list(self.snake_head))
            reward += self.EAT_APPLE_REWARD
        else:
            self.snake_position.insert(0,list(self.snake_head))
            self.snake_position.pop()

        #Obstacle Observation Space Calc.
        self.snake_left_obstacle = 0
        self.snake_up_obstacle = 0
        self.snake_right_obstacle = 0
        self.snake_down_obstacle = 0
        self.snake_head[0] -= 10
        if(collision_with_boundaries(self.snake_head) or collision_with_self(self.snake_head)):
            self.snake_up_obstacle = 1
        self.snake_head[0] += 20
        if(collision_with_boundaries(self.snake_head) or collision_with_self(self.snake_head)):
            self.snake_right_obstacle = 1
        self.snake_head[0] -=10
        self.snake_head[1] += 10
        if(collision_with_boundaries(self.snake_head) or collision_with_self(self.snake_head)):
            self.snake_left_obstacle = 1
        self.snake_head[1] -= 20
        if(collision_with_boundaries(self.snake_head) or collision_with_self(self.snake_head)):
            self.snake_down_obstacle = 1
        self.snake_head[1] += 10 
    
            
        #Apple Position Relative to  Snake Head
        self.apple_above_head = 0
        self.apple_left_head = 0
        self.apple_below_head = 0
        self.apple_right_head = 0
        if(self.snake_head[0] > self.apple_position[0]):
            self.apple_right_head = 1
        else:
            self.apple_below_head = 1
        
        if(self.snake_head[1] > self.apple_position[1]):
            self.apple_left_head = 1
        else:
            self.apple_above_head = 1


        # On collision kill the snake and print the score
        if collision_with_boundaries(self.snake_head) == 1 or collision_with_self(self.snake_position) == 1:
            font = cv2.FONT_HERSHEY_SIMPLEX
            self.img = np.zeros((500,500,3),dtype='uint8')
            cv2.putText(self.img,'Your Score is {}'.format(self.score),(140,250), font, 1,(255,255,255),2,cv2.LINE_AA)
            cv2.imshow('a',self.img)
            self.done = True
            reward += self.DEATH_REWARD
        info = {}
        observation = [self.apple_above_head, self.apple_below_head, self.apple_left_head, self.apple_right_head, self.snake_up_obstacle]
        observation = observation + [self.snake_down_obstacle, self.snake_left_obstacle, self.snake_right_obstacle]
        observation = observation + [self.snake_going_left, self.snake_going_right, self.snake_going_down, self.snake_going_up]
        observation = np.array(observation)
        return observation, reward, self.done, info
    def reset(self):
        cv2.destroyAllWindows()
        self.done = False
        self.set_Pos()
        self.snake_head = [250, 250]
        self.score = 0
        observation = [self.apple_above_head, self.apple_below_head, self.apple_left_head, self.apple_right_head, self.snake_up_obstacle]
        observation = observation + [self.snake_down_obstacle, self.snake_left_obstacle, self.snake_right_obstacle]
        observation = observation + [self.snake_going_left, self.snake_going_right, self.snake_going_down, self.snake_going_up]
        observation = np.array(observation)
        return observation
    def set_Pos(self):
        self.snake_position = [[250,250]]
        self.snake_prev_apple_distance = 10000
        self.apple_distance = 0
        self.apple_above_head = 0
        self.apple_below_head = 0
        self.apple_left_head = 0
        self.apple_right_head = 0
        self.snake_left_obstacle = 1
        self.snake_up_obstacle = 0
        self.snake_down_obstacle = 0
        self.snake_right_obstacle = 0
        self.snake_going_left = 0
        self.snake_going_right = 1
        self.snake_going_up = 0
        self.snake_going_down = 0
        self.apple_position = [random.randrange(1,50)*10,random.randrange(1,50)*10]
        self.prev_button_direction = 1
        self.button_direction = 1
        self.img = np.zeros((500,500,3),dtype='uint8')

env = CustomEnv()
log_path = r"C:\Users\arsh0\OneDrive\Documents\SnakeML\myModel.zip"
model = PPO.load(log_path, env=env)
#model = PPO('MlpPolicy', env, verbose = 1, tensorboard_log=log_path)
model.learn(total_timesteps= 100000)
model.save("myModel2")

#model = PPO.load(log_path, env=env)
#evaluate_policy(model, env, n_eval_episodes=10)
#model = PPO("MlpPolicy", env, verbose = 2, tensorboard_log=log_path)
