# Tennis (Project 3) - Report

To solve this environment, I started with the Deep Deterministic Policy Gradients using an Actor Critic Neural Network that was shown earlier in this section of the class. 


As with any Reinforcement Learning problem, it can be boiled down to:
As an Agent in a State, what action do I take to maximize the Reward the Environment will give me.


![rl-diag](md-images/RL-diagram.png)

# Deep Deterministic Policy Gradients (DDPG)

DDPG is an approach for reinforcement learning environments where an agent uses a policy as a function that has a probability of taking an action given a state. For following through with this action, the agent receives a reward. A policy is said to be good if it results in a large reward over an episode of the environment. The optimal policy is the policy that maximizes the the reward obtained while following the policy. This environment being continuous would make using a Q-Learning Network approach fail where instead you are trying to figure out what is the best specific action. Instead we use two neural networks to approximate two values. The first network is called the Actor which is used to approximate the optimal policy where as the second network is the Critic which tries to estimate the reward from following that approximately optimal policy. This becomes a “try the policy” then “evaluate the policy” then “improve the policy” loop where the actor tries the policy and the critic evaluates the policy. The improving step comes from the actor and critic network updating through their loss functions. 

This is the general structure of what the method uses but there are some additional moving parts behind the scenes. This implementation uses Ornstein Uhlenbeck noise, a replay buffer, target networks, and soft updating. 

When first starting the training process, the actor and critic network are randomly initialized. During an episode at a time step, the actor network is given the current state and returns a value that is added to the Ornstein noise. This action is taken giving a new state as well as a reward for taking the previous action. This is then stored in the Replay Buffer in the form of a tuple (State, Action, Reward, NextState). Once the replay buffer has enough transitions, a random sample of them is taken and is used to help the critic network. The critic network is evaluated at the new state with an action given by the actor network evaluated at the new state. This value can be thought of as an approximation of the next reward from taking the next state, or expected Q value. Then both networks are updated, first the critic then the actor. This takes form of the mean squared error between the expected Q value and the actual Q value during a transition. The actor uses gradient ascent to update the actor network towards a better policy. Another more subtle issue that is run into is that if we update our networks every chance we get, it has the potential to become very unstable. To mitigate this problem, we will actually use two neural networks for both the actor and critic, although in the end we only care about one. The networks are our main actor network, main critic network, our target actor network and our target critic network. The target networks are held static for a fixed number of training steps while training the main networks as the goal of what the main network should be working towards. When the target networks are updated, they are pushed towards the main networks but do not get replaced completely by the main networks. This is called soft updating and makes things run much more smoothly.










# Results
I ran many experiments with different hyper parameters. The ones I chose to vary were as follows: the learning rate (GAMMA), the number of units in the first convolutional layer (fc1_units) of each network, the number of units in the second convolutional layer (fc2_units) of each network, the learning rate of the Actor network (LR_ACTOR), and the learning rate of the Critic network (LR_CRITIC).

GAMMA takes on values = [0.95, 0.97, 0.99, 1.00]
fc1_units takes on values = [100, 200, 400]
fc2_units takes on values = [100, 200, 400]
LR_ACTOR takes on values = [1e-3, 1e-4]
LR_CRITIC takes on values = [1e-3, 1e-4]

All other hyper parameters were the defaults set by training.py (also shown in README.md)

This gives 4 (GAMMA) * 3 (fc1_units) * 3 (fc2_units) * 2 (LR_ACTOR) * 2 (LR_CRITIC) = 144 total models trained.


The neural network architecture is as follows:

The results are shown below (the horizontal black line is the value required to be considered solved):
![all_model_plots](results-tennis.png)

After this grid search for hyper parameters, I took the 9 best models (models 44, 60, 64, 70, 71, 79, 80, 98, 122) and re-trained using new starting seeds for initialization. I trained 5 new seeds (10, 20, 30, 40, 50) to find which model performs best.

These results of these 45 newly trained models are shown below where the model number is the original model number with the seed appended to the end of the value.

![seed_plots](results.png)


These 10 models statistics are shown below.

| model | mean               | std    | min    | 25%     | 50%    | 75%     | max    |
|-------|--------------------|--------|--------|---------|--------|---------|--------|
| 71    | 2183.5             | 544.68 | 1406.0 | 2006.0  | 2382.5 | 2560.0  | 2563.0 |
| 44    | 2286.5             | 787.01 | 1730.0 | 2008.25 | 2286.5 | 2564.75 | 2843.0 |
| 60    | 2471.0             | 94.75  | 2404.0 | 2437.5  | 2471.0 | 2504.5  | 2538.0 |
| 80    | 2606.5             | 599.13 | 2138.0 | 2150.0  | 2441.5 | 2898.0  | 3405.0 |
| 64    | 3090.3333333333335 | 388.44 | 2769.0 | 2874.5  | 2980.0 | 3251.0  | 3522.0 |
| 70    | NaN                | NaN    | NaN    | NaN     | NaN    | NaN     | NaN    |
| 79    | NaN                | NaN    | NaN    | NaN     | NaN    | NaN     | NaN    |
| 98    | NaN                | NaN    | NaN    | NaN     | NaN    | NaN     | NaN    |
| 122   | NaN                | NaN    | NaN    | NaN     | NaN    | NaN     | NaN    |

It can be seen that model 71 turned out to be the best model. It can be seen that the NaN values in the chart above are reflective of how much a good initial seed matters. It seems that model 60 is the most stable with the smallest standard deviation.

With GAMMA = 0.97, LR_ACTOR = 1e-3, LR_CRITIC = 1e-3, fc1_units = 100, fc2_units = 200.

The results for all-models and the seed-models are shown (sorted by win_iter) in the tables in the Appendix - A and Appendix - B respectively.

The best models can be found [here]().

# Future Work!

Since this is a continuous state space, a subset of the other methods out there are an option. This includes using PPO, A3C, or D4PG to take advantage of being able to train multiple agents in parallel together.

# Appendix - A (All-Model Results)
| model     | GAMMA | LR_ACTOR | LR_CRITIC | fc1_units | fc2_units | seed | win_iter |
|-----------|-------|----------|-----------|-----------|-----------|------|----------|
| model-44  | 0.97  | 0.0001   | 0.0001    | 100       | 200       | 42   | 1505.0   |
| model-64  | 0.97  | 0.0001   | 0.0001    | 400       | 100       | 42   | 1727.0   |
| model-71  | 0.97  | 0.0001   | 0.0001    | 400       | 400       | 42   | 1819.0   |
| model-80  | 0.99  | 0.0001   | 0.0001    | 100       | 200       | 42   | 1839.0   |
| model-60  | 0.97  | 0.0001   | 0.0001    | 200       | 400       | 42   | 1899.0   |
| model-127 | 1.0   | 0.0001   | 0.0001    | 200       | 200       | 42   | 2095.0   |
| model-56  | 0.97  | 0.0001   | 0.0001    | 200       | 200       | 42   | 2101.0   |
| model-143 | 1.0   | 0.0001   | 0.0001    | 400       | 400       | 42   | 2402.0   |
| model-36  | 0.95  | 0.0001   | 0.0001    | 400       | 400       | 42   | 2430.0   |
| model-8   | 0.95  | 0.0001   | 0.0001    | 100       | 200       | 42   | 2621.0   |
| model-123 | 1.0   | 0.0001   | 0.0001    | 200       | 100       | 42   | 2727.0   |
| model-12  | 0.95  | 0.0001   | 0.0001    | 100       | 400       | 42   | 3040.0   |
| model-52  | 0.97  | 0.0001   | 0.0001    | 200       | 100       | 42   | 3111.0   |
| model-16  | 0.95  | 0.0001   | 0.0001    | 200       | 100       | 42   | 3131.0   |
| model-99  | 0.99  | 0.0001   | 0.0001    | 400       | 100       | 42   | 3197.0   |
| model-40  | 0.97  | 0.0001   | 0.0001    | 100       | 100       | 42   | 3889.0   |
| model-1   | 0.95  | 0.0000   | 0.0000    | 100       | 100       | 42   |          |
| model-10  | 0.95  | 0.0000   | 0.0001    | 100       | 400       | 42   |          |
| model-100 | 0.99  | 0.0000   | 0.0000    | 400       | 200       | 42   |          |
| model-101 | 0.99  | 0.0000   | 0.0001    | 400       | 200       | 42   |          |
| model-102 | 0.99  | 0.0001   | 0.0000    | 400       | 200       | 42   |          |
| model-103 | 0.99  | 0.0001   | 0.0001    | 400       | 200       | 42   |          |
| model-104 | 0.99  | 0.0000   | 0.0000    | 400       | 400       | 42   |          |
| model-105 | 0.99  | 0.0000   | 0.0001    | 400       | 400       | 42   |          |
| model-106 | 0.99  | 0.0001   | 0.0000    | 400       | 400       | 42   |          |
| model-107 | 0.99  | 0.0001   | 0.0001    | 400       | 400       | 42   |          |
| model-108 | 1.0   | 0.0000   | 0.0000    | 100       | 100       | 42   |          |
| model-109 | 1.0   | 0.0000   | 0.0001    | 100       | 100       | 42   |          |
| model-11  | 0.95  | 0.0001   | 0.0000    | 100       | 400       | 42   |          |
| model-110 | 1.0   | 0.0001   | 0.0000    | 100       | 100       | 42   |          |
| model-111 | 1.0   | 0.0001   | 0.0001    | 100       | 100       | 42   |          |
| model-112 | 1.0   | 0.0000   | 0.0000    | 100       | 200       | 42   |          |
| model-113 | 1.0   | 0.0000   | 0.0001    | 100       | 200       | 42   |          |
| model-114 | 1.0   | 0.0001   | 0.0000    | 100       | 200       | 42   |          |
| model-115 | 1.0   | 0.0001   | 0.0001    | 100       | 200       | 42   |          |
| model-116 | 1.0   | 0.0000   | 0.0000    | 100       | 400       | 42   |          |
| model-117 | 1.0   | 0.0000   | 0.0001    | 100       | 400       | 42   |          |
| model-118 | 1.0   | 0.0001   | 0.0000    | 100       | 400       | 42   |          |
| model-119 | 1.0   | 0.0001   | 0.0001    | 100       | 400       | 42   |          |
| model-120 | 1.0   | 0.0000   | 0.0000    | 200       | 100       | 42   |          |
| model-121 | 1.0   | 0.0000   | 0.0001    | 200       | 100       | 42   |          |
| model-122 | 1.0   | 0.0001   | 0.0000    | 200       | 100       | 42   |          |
| model-124 | 1.0   | 0.0000   | 0.0000    | 200       | 200       | 42   |          |
| model-125 | 1.0   | 0.0000   | 0.0001    | 200       | 200       | 42   |          |
| model-126 | 1.0   | 0.0001   | 0.0000    | 200       | 200       | 42   |          |
| model-128 | 1.0   | 0.0000   | 0.0000    | 200       | 400       | 42   |          |
| model-129 | 1.0   | 0.0000   | 0.0001    | 200       | 400       | 42   |          |
| model-13  | 0.95  | 0.0000   | 0.0000    | 200       | 100       | 42   |          |
| model-130 | 1.0   | 0.0001   | 0.0000    | 200       | 400       | 42   |          |
| model-132 | 1.0   | 0.0000   | 0.0000    | 400       | 100       | 42   |          |
| model-133 | 1.0   | 0.0000   | 0.0001    | 400       | 100       | 42   |          |
| model-134 | 1.0   | 0.0001   | 0.0000    | 400       | 100       | 42   |          |
| model-135 | 1.0   | 0.0001   | 0.0001    | 400       | 100       | 42   |          |
| model-136 | 1.0   | 0.0000   | 0.0000    | 400       | 200       | 42   |          |
| model-137 | 1.0   | 0.0000   | 0.0001    | 400       | 200       | 42   |          |
| model-138 | 1.0   | 0.0001   | 0.0000    | 400       | 200       | 42   |          |
| model-139 | 1.0   | 0.0001   | 0.0001    | 400       | 200       | 42   |          |
| model-14  | 0.95  | 0.0000   | 0.0001    | 200       | 100       | 42   |          |
| model-140 | 1.0   | 0.0000   | 0.0000    | 400       | 400       | 42   |          |
| model-141 | 1.0   | 0.0000   | 0.0001    | 400       | 400       | 42   |          |
| model-142 | 1.0   | 0.0001   | 0.0000    | 400       | 400       | 42   |          |
| model-15  | 0.95  | 0.0001   | 0.0000    | 200       | 100       | 42   |          |
| model-17  | 0.95  | 0.0000   | 0.0000    | 200       | 200       | 42   |          |
| model-18  | 0.95  | 0.0000   | 0.0001    | 200       | 200       | 42   |          |
| model-19  | 0.95  | 0.0001   | 0.0000    | 200       | 200       | 42   |          |
| model-3   | 0.95  | 0.0001   | 0.0000    | 100       | 100       | 42   |          |
| model-30  | 0.95  | 0.0000   | 0.0001    | 400       | 200       | 42   |          |
| model-31  | 0.95  | 0.0001   | 0.0000    | 400       | 200       | 42   |          |
| model-32  | 0.95  | 0.0000   | 0.0000    | 400       | 400       | 42   |          |
| model-33  | 0.95  | 0.0000   | 0.0000    | 400       | 400       | 42   |          |
| model-34  | 0.95  | 0.0000   | 0.0001    | 400       | 400       | 42   |          |
| model-35  | 0.95  | 0.0001   | 0.0000    | 400       | 400       | 42   |          |
| model-37  | 0.97  | 0.0000   | 0.0000    | 100       | 100       | 42   |          |
| model-38  | 0.97  | 0.0000   | 0.0001    | 100       | 100       | 42   |          |
| model-39  | 0.97  | 0.0001   | 0.0000    | 100       | 100       | 42   |          |
| model-4   | 0.95  | 0.0001   | 0.0001    | 100       | 100       | 42   |          |
| model-41  | 0.97  | 0.0000   | 0.0000    | 100       | 200       | 42   |          |
| model-42  | 0.97  | 0.0000   | 0.0001    | 100       | 200       | 42   |          |
| model-43  | 0.97  | 0.0001   | 0.0000    | 100       | 200       | 42   |          |
| model-45  | 0.97  | 0.0000   | 0.0000    | 100       | 400       | 42   |          |
| model-46  | 0.97  | 0.0001   | 0.0000    | 100       | 400       | 42   |          |
| model-47  | 0.97  | 0.0001   | 0.0000    | 100       | 400       | 42   |          |
| model-48  | 0.97  | 0.0001   | 0.0001    | 100       | 400       | 42   |          |
| model-49  | 0.97  | 0.0000   | 0.0000    | 200       | 100       | 42   |          |
| model-5   | 0.95  | 0.0000   | 0.0000    | 100       | 200       | 42   |          |
| model-50  | 0.97  | 0.0000   | 0.0001    | 200       | 100       | 42   |          |
| model-51  | 0.97  | 0.0001   | 0.0000    | 200       | 100       | 42   |          |
| model-53  | 0.97  | 0.0000   | 0.0000    | 200       | 200       | 42   |          |
| model-54  | 0.97  | 0.0000   | 0.0001    | 200       | 200       | 42   |          |
| model-55  | 0.97  | 0.0001   | 0.0000    | 200       | 200       | 42   |          |
| model-57  | 0.97  | 0.0000   | 0.0000    | 200       | 400       | 42   |          |
| model-58  | 0.97  | 0.0000   | 0.0001    | 200       | 400       | 42   |          |
| model-59  | 0.97  | 0.0001   | 0.0000    | 200       | 400       | 42   |          |
| model-6   | 0.95  | 0.0000   | 0.0001    | 100       | 200       | 42   |          |
| model-61  | 0.97  | 0.0000   | 0.0000    | 400       | 100       | 42   |          |
| model-62  | 0.97  | 0.0000   | 0.0001    | 400       | 100       | 42   |          |
| model-63  | 0.97  | 0.0001   | 0.0000    | 400       | 100       | 42   |          |
| model-65  | 0.97  | 0.0000   | 0.0000    | 400       | 200       | 42   |          |
| model-66  | 0.97  | 0.0000   | 0.0001    | 400       | 200       | 42   |          |
| model-67  | 0.97  | 0.0001   | 0.0000    | 400       | 200       | 42   |          |
| model-68  | 0.97  | 0.0000   | 0.0000    | 400       | 400       | 42   |          |
| model-69  | 0.97  | 0.0000   | 0.0000    | 400       | 400       | 42   |          |
| model-7   | 0.95  | 0.0001   | 0.0000    | 100       | 200       | 42   |          |
| model-70  | 0.97  | 0.0000   | 0.0001    | 400       | 400       | 42   |          |
| model-72  | 0.99  | 0.0000   | 0.0000    | 100       | 100       | 42   |          |
| model-73  | 0.99  | 0.0000   | 0.0000    | 100       | 100       | 42   |          |
| model-74  | 0.99  | 0.0000   | 0.0001    | 100       | 100       | 42   |          |
| model-75  | 0.99  | 0.0001   | 0.0000    | 100       | 100       | 42   |          |
| model-76  | 0.99  | 0.0001   | 0.0001    | 100       | 100       | 42   |          |
| model-77  | 0.99  | 0.0000   | 0.0000    | 100       | 200       | 42   |          |
| model-78  | 0.99  | 0.0000   | 0.0001    | 100       | 200       | 42   |          |
| model-79  | 0.99  | 0.0001   | 0.0000    | 100       | 200       | 42   |          |
| model-81  | 0.99  | 0.0000   | 0.0000    | 100       | 400       | 42   |          |
| model-82  | 0.99  | 0.0000   | 0.0001    | 100       | 400       | 42   |          |
| model-83  | 0.99  | 0.0001   | 0.0000    | 100       | 400       | 42   |          |
| model-84  | 0.99  | 0.0001   | 0.0001    | 100       | 400       | 42   |          |
| model-85  | 0.99  | 0.0000   | 0.0000    | 200       | 100       | 42   |          |
| model-86  | 0.99  | 0.0000   | 0.0001    | 200       | 100       | 42   |          |
| model-87  | 0.99  | 0.0001   | 0.0000    | 200       | 100       | 42   |          |
| model-88  | 0.99  | 0.0000   | 0.0000    | 200       | 200       | 42   |          |
| model-89  | 0.99  | 0.0000   | 0.0000    | 200       | 200       | 42   |          |
| model-9   | 0.95  | 0.0000   | 0.0000    | 100       | 400       | 42   |          |
| model-90  | 0.99  | 0.0000   | 0.0001    | 200       | 200       | 42   |          |
| model-91  | 0.99  | 0.0001   | 0.0000    | 200       | 200       | 42   |          |
| model-92  | 0.99  | 0.0000   | 0.0000    | 200       | 400       | 42   |          |
| model-93  | 0.99  | 0.0000   | 0.0000    | 200       | 400       | 42   |          |
| model-94  | 0.99  | 0.0000   | 0.0001    | 200       | 400       | 42   |          |
| model-95  | 0.99  | 0.0001   | 0.0000    | 200       | 400       | 42   |          |
| model-96  | 0.99  | 0.0000   | 0.0000    | 400       | 100       | 42   |          |
| model-97  | 0.99  | 0.0000   | 0.0001    | 400       | 100       | 42   |          |
| model-98  | 0.99  | 0.0001   | 0.0000    | 400       | 100       | 42   |          |

# Appendix B - (Seed-Model Results)
| model       | GAMMA | LR_ACTOR | LR_CRITIC | fc1_units | fc2_units | seed | win_iter |
|-------------|-------|----------|-----------|-----------|-----------|------|----------|
| model-7110  | 0.97  | 0.0001   | 0.0001    | 400       | 400       | 10   | 1406.0   |
| model-4410  | 0.97  | 0.0001   | 0.0001    | 100       | 200       | 10   | 1730.0   |
| model-8040  | 0.99  | 0.0001   | 0.0001    | 100       | 200       | 40   | 2138.0   |
| model-8020  | 0.99  | 0.0001   | 0.0001    | 100       | 200       | 20   | 2154.0   |
| model-7120  | 0.97  | 0.0001   | 0.0001    | 400       | 400       | 20   | 2206.0   |
| model-6010  | 0.97  | 0.0001   | 0.0001    | 200       | 400       | 10   | 2404.0   |
| model-6030  | 0.97  | 0.0001   | 0.0001    | 200       | 400       | 30   | 2538.0   |
| model-7140  | 0.97  | 0.0001   | 0.0001    | 400       | 400       | 40   | 2559.0   |
| model-7130  | 0.97  | 0.0001   | 0.0001    | 400       | 400       | 30   | 2563.0   |
| model-8010  | 0.99  | 0.0001   | 0.0001    | 100       | 200       | 10   | 2729.0   |
| model-6420  | 0.97  | 0.0001   | 0.0001    | 400       | 100       | 20   | 2769.0   |
| model-4430  | 0.97  | 0.0001   | 0.0001    | 100       | 200       | 30   | 2843.0   |
| model-6440  | 0.97  | 0.0001   | 0.0001    | 400       | 100       | 40   | 2980.0   |
| model-8050  | 0.99  | 0.0001   | 0.0001    | 100       | 200       | 50   | 3405.0   |
| model-6410  | 0.97  | 0.0001   | 0.0001    | 400       | 100       | 10   | 3522.0   |
| model-12210 | 1.0   | 0.0001   | 0.0000    | 200       | 100       | 10   |          |
| model-12220 | 1.0   | 0.0001   | 0.0000    | 200       | 100       | 20   |          |
| model-12230 | 1.0   | 0.0001   | 0.0000    | 200       | 100       | 30   |          |
| model-12240 | 1.0   | 0.0001   | 0.0000    | 200       | 100       | 40   |          |
| model-12250 | 1.0   | 0.0001   | 0.0000    | 200       | 100       | 50   |          |
| model-4420  | 0.97  | 0.0001   | 0.0001    | 100       | 200       | 20   |          |
| model-4440  | 0.97  | 0.0001   | 0.0001    | 100       | 200       | 40   |          |
| model-4450  | 0.97  | 0.0001   | 0.0001    | 100       | 200       | 50   |          |
| model-6020  | 0.97  | 0.0001   | 0.0001    | 200       | 400       | 20   |          |
| model-6040  | 0.97  | 0.0001   | 0.0001    | 200       | 400       | 40   |          |
| model-6050  | 0.97  | 0.0001   | 0.0001    | 200       | 400       | 50   |          |
| model-6430  | 0.97  | 0.0001   | 0.0001    | 400       | 100       | 30   |          |
| model-6450  | 0.97  | 0.0001   | 0.0001    | 400       | 100       | 50   |          |
| model-7010  | 0.97  | 0.0000   | 0.0001    | 400       | 400       | 10   |          |
| model-7020  | 0.97  | 0.0000   | 0.0001    | 400       | 400       | 20   |          |
| model-7030  | 0.97  | 0.0000   | 0.0001    | 400       | 400       | 30   |          |
| model-7040  | 0.97  | 0.0000   | 0.0001    | 400       | 400       | 40   |          |
| model-7050  | 0.97  | 0.0000   | 0.0001    | 400       | 400       | 50   |          |
| model-7150  | 0.97  | 0.0001   | 0.0001    | 400       | 400       | 50   |          |
| model-7910  | 0.99  | 0.0001   | 0.0000    | 100       | 200       | 10   |          |
| model-7920  | 0.99  | 0.0001   | 0.0000    | 100       | 200       | 20   |          |
| model-7930  | 0.99  | 0.0001   | 0.0000    | 100       | 200       | 30   |          |
| model-7940  | 0.99  | 0.0001   | 0.0000    | 100       | 200       | 40   |          |
| model-7950  | 0.99  | 0.0001   | 0.0000    | 100       | 200       | 50   |          |
| model-8030  | 0.99  | 0.0001   | 0.0001    | 100       | 200       | 30   |          |
| model-9810  | 0.99  | 0.0001   | 0.0000    | 400       | 100       | 10   |          |
| model-9820  | 0.99  | 0.0001   | 0.0000    | 400       | 100       | 20   |          |
| model-9830  | 0.99  | 0.0001   | 0.0000    | 400       | 100       | 30   |          |
| model-9840  | 0.99  | 0.0001   | 0.0000    | 400       | 100       | 40   |          |
| model-9850  | 0.99  | 0.0001   | 0.0000    | 400       | 100       | 50   |          |