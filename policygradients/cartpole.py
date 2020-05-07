import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

env = gym.make('CartPole-v0')

class PolicyNet(nn.Module):
    def __init__(self, N_inputs, N_outputs, N_hidden_layers, N_hidden_nodes, activation, output_activation):
        super(PolicyNet, self).__init__()
        
        self.N_inputs = N_inputs
        self.N_outputs = N_outputs
        
        self.N_hidden_layers = N_hidden_layers
        self.N_hidden_nodes = N_hidden_nodes
        
        self.layer_list = nn.ModuleList([]) #use just as a python list
        for n in range(N_hidden_layers):
            if n==0:
                self.layer_list.append(nn.Linear(N_inputs, N_hidden_nodes))
            else:
                self.layer_list.append(nn.Linear(N_hidden_nodes, N_hidden_nodes))
        
        self.output_layer = nn.Linear(N_hidden_nodes, N_outputs)
        
        self.activation = activation
        self.output_activation = output_activation
        
    def forward(self, inp):
        out = inp
        for layer in self.layer_list:
            out = layer(out)
            out = self.activation(out)
            
        out = self.output_layer(out)
        if self.output_activation is not None:
            pred = self.output_activation(out)
        else:
            pred = out
        
        return pred

#initialize policy
N_inputs = 4
N_outputs = 2
N_hidden_layers = 1
N_hidden_nodes = 10
activation = nn.ReLU()
output_activation = nn.Softmax(dim=1)

action_space = torch.arange(0, 2)

policy = PolicyNet(N_inputs, 
                   N_outputs, 
                   N_hidden_layers, 
                   N_hidden_nodes,
                   activation,
                   output_activation=output_activation)

#(s_t, t) -> estimate of value function
critic = PolicyNet(N_inputs+1, 
                   1, 
                   N_hidden_layers, 
                   N_hidden_nodes,
                   activation,
                   None)


def create_trajectories(env, policy, N, causal=False, baseline=None):
    J_list = []
    R_list = []

    action_probs_all_list = []
    rewards_all_list = []

    for _ in range(N):
        state = env.reset()
        prob_list = torch.tensor([])
        
        action_prob_list, reward_list = torch.tensor([]), torch.tensor([])
        done = False 

        while not done:
            action_probs = policy(torch.from_numpy(state).unsqueeze(0).float()).squeeze(0)

            action_selected_index = torch.multinomial(action_probs, 1)
            action_selected_prob = action_probs[action_selected_index]
            action_selected = action_space[action_selected_index]

            state, reward, done, info = env.step(action_selected.item())

            action_prob_list = torch.cat((action_prob_list, action_selected_prob))
            reward_list = torch.cat((reward_list, torch.tensor([reward])))

        action_probs_all_list.append(action_prob_list)
        rewards_all_list.append(reward_list)

    #non-optimized code (negative strides not supported by torch yet)
    rewards_to_go_list = [torch.tensor(np.cumsum(traj_rewards.numpy()[::-1])[::-1].copy())
                          for traj_rewards in rewards_all_list]

    #mean reward
    R = np.mean([torch.sum(r).item() for r in rewards_all_list])
    
    baseline_term = 0
    if baseline:
        baseline_term = R

    J = 0
    if causal:
        for idx in range(N):
            J += (action_probs_all_list[idx].log() * (rewards_to_go_list[idx] - baseline_term)).sum()
    else:
        for idx in range(N): #loop over trajs
            J += action_probs_all_list[idx].log().sum() * (rewards_all_list[idx].sum() - baseline_term)
    
    J = J / N

    R = np.mean([torch.sum(r).item() for r in rewards_all_list])

    return J, R

def training_loop(N_iter, 
                  batch_size, 
                  env,
                  policy=None, 
                  lr=1e-2, 
                  causal=False,
                  baseline=None,
                  debug=False):
    
    if policy is None:
        policy = PolicyNet(N_inputs, N_outputs, N_hidden_layers, N_hidden_nodes, nn.ReLU(), nn.Sigmoid())
    
    reward_curve = {}

    optimizer = optim.Adam(policy.parameters(), lr=lr)

    exp_reward_list = []

    for i in range(N_iter):
        #step 1: generate batch_size trajectories
        J, mean_reward = create_trajectories(env, policy, batch_size, causal=causal, baseline=baseline)

        #step 2: define J
        optimizer.zero_grad()
        (-J).backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Iteration {i} : Mean Reward = {mean_reward}")
            reward_curve[i] = mean_reward

    return reward_curve

def get_learning_curves(N_exp, N_iter, batch_size, env, causal=False, baseline=None):
    reward_curve_list = []
    for _ in range(N_exp):
        policy = PolicyNet(N_inputs, N_outputs, N_hidden_layers, N_hidden_nodes, activation, output_activation)

        reward_curve = training_loop(N_iter, batch_size, env, policy, causal=causal, baseline=baseline)

        reward_curve_list.append(reward_curve)

    #combine results
    rcurve = {}
    for k in reward_curve_list[0]:
        rcurve[k] = [r[k] for r in reward_curve_list]
        rcurve[k] = (np.mean(rcurve[k]), np.std(rcurve[k]))

    return rcurve

def plot_learning_curve(rcurve, label):
    k = list(rcurve.keys())

    plt.errorbar(k, [rcurve[key][0] for key in k], [rcurve[key][1] for key in k], label=label)
    plt.legend()
    plt.xlabel('Episode Number')
    plt.ylabel('Average Score')
    plt.title('10 runs, 200 iterations each, batch-size 20 episodes')
    #plt.savefig('LearningCurve.png')

def animate(env, policy):
    '''Can be combined with create trajectories above
    '''
    state = env.reset()

    done = False
    while not done:
        env.render()

        action_probs = policy(torch.from_numpy(state).unsqueeze(0).float()).squeeze(0)

        action_selected_index = torch.multinomial(action_probs, 1)
        action_selected_prob = action_probs[action_selected_index]
        action_selected = action_space[action_selected_index]

        state, reward, done, info = env.step(action_selected.item())
