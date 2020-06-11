import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pylab as plt

'''
Critic changes:

define critic

model update in training_loop

use critic during gradient computation in create_trajectories

compute critic loss based on new trajectory
'''

plt.ion()

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

#(s_t, t) -> estimate of value function (not being used yet)
critic = PolicyNet(N_inputs+1, #inputs and time-step
                   1, 
                   N_hidden_layers, 
                   N_hidden_nodes,
                   activation,
                   None)


def create_trajectories(env, policy, N, causal=False, baseline=False, critic=False, critic_update='MC'):
    action_probs_all_list = []
    rewards_all_list = []
    states_all_list = []

    for _ in range(N): #run env N times
        state = env.reset()
        
        action_prob_list, reward_list, state_list = torch.tensor([]), torch.tensor([]), torch.tensor([])
        done = False 

        while not done:
            state_list = torch.cat((state_list, torch.tensor([state]).float())) #

            action_probs = policy(torch.from_numpy(state).unsqueeze(0).float()).squeeze(0)

            action_selected_index = torch.multinomial(action_probs, 1)
            action_selected_prob = action_probs[action_selected_index]
            action_selected = action_space[action_selected_index]

            state, reward, done, info = env.step(action_selected.item())

            action_prob_list = torch.cat((action_prob_list, action_selected_prob))
            reward_list = torch.cat((reward_list, torch.tensor([reward])))

        action_probs_all_list.append(action_prob_list)
        rewards_all_list.append(reward_list)
        states_all_list.append(state_list)

    #non-optimized code (negative strides not supported by torch yet)
    rewards_to_go_list = [torch.tensor(np.cumsum(traj_rewards.numpy()[::-1])[::-1].copy())
                          for traj_rewards in rewards_all_list]


    #compute objective
    J = 0
    critic_inputs, critic_targets = [], []
    #for clarity, refactoring code below into specific modes. some of these can be combined
    if not baseline and not causal:
        print("No Baseline - Not Causal")
        for idx in range(N):
            J += action_probs_all_list[idx].log().sum() * (rewards_all_list[idx].sum())

    if not baseline and causal:
        print("No Baseline - Causal")
        for idx in range(N):
            row = rewards_to_go_list[idx]
            J += (action_probs_all_list[idx].log() * row).sum()

    #critic only shows up in baseline cases
    if baseline and not causal and not critic: #critic only affects baseline cases
        print("Baseline - Not Causal, No Critic")
        baseline_term = np.mean([torch.sum(r).item() for r in rewards_all_list])

        for idx in range(N):
            J += action_probs_all_list[idx].log().sum() * (rewards_all_list[idx].sum() - baseline_term)

    if baseline and not causal and critic:
        raise NotImplementedError("Probably not useful")
        for idx in range(N):
            row = rewards_to_go_list[idx]
            actions = action_probs_all_list[idx]
            states = states_all_list[idx]

            T = len(row)
            for t in range(T):
                J += actions[t].log() * (row[t] - critic(torch.cat([torch.tensor([t]).float(), states[t]])))

    if baseline and causal and not critic:
        '''Need time-dependent baseline terms
        '''
        print("Baseline - Causal, No Critic")
        #compute time-dependent baseline terms (mean reward to go)
        T = np.max([len(row) for row in rewards_to_go_list])
        baseline_term = torch.zeros(N, T)
        for idx in range(N):
            row = rewards_to_go_list[idx]
            baseline_term[idx] = torch.cat((row, torch.zeros(T-len(row)))) #pad with 0s if episode took time < T (end)
        baseline_term = torch.mean(baseline_term, dim=0) #time-dependent means

        #compute J
        for idx in range(N):
            row = rewards_to_go_list[idx]
            J += (action_probs_all_list[idx].log() * (row - baseline_term[:len(row)])).sum() #subtract time-dependent means

    if baseline and causal and critic:
        #train/update critic
        #(state, t) -> (rewards to go)

        #print("Baseline - Causal, Critic")
        #return states_all_list, action_probs_all_list, rewards_to_go_list
        for idx in range(N):
            row = rewards_to_go_list[idx]
            rewards = rewards_all_list[idx] #needed to update critic
            actions = action_probs_all_list[idx]
            states = states_all_list[idx]

            T = len(row)
            for t in range(T):
                critic_in = torch.cat([torch.tensor([t]).float(), states[t]])
                
                J += actions[t].log() * (row[t] - critic(critic_in))

                if critic_update=='MC':
                    critic_inputs.append(critic_in)
                    critic_targets.append(row[t]) #pure MC - target = reward to go
                
                if critic_update=='TD':
                    if t < T-1:
                        critic_inputs.append(critic_in)
                        critic_targets.append(rewards[t].unsqueeze(0) + critic(torch.cat([torch.tensor([t+1]).float(), states[t+1]]))) #TD - target = current reward + critic estimate
    J = J / N


    '''
    if causal:
        #compute baseline terms dependent on time (pretty horrible implementation)
        T = np.max([len(row) for row in rewards_to_go_list])
        baseline_term = torch.zeros(T)
        if baseline:
            baseline_term = torch.zeros(N, T)
            for idx in range(N):
                row = rewards_to_go_list[idx]
                baseline_term[idx] = torch.cat((row, torch.zeros(T-len(row)))) #pad with 0s if episode took time < T (end)
            baseline_term = torch.mean(baseline_term, dim=0) #time-dependent means

        for idx in range(N):
            row = rewards_to_go_list[idx]
            J += (action_probs_all_list[idx].log() * (row - baseline_term[:len(row)])).sum() #subtract time-dependent means
    else:
        baseline_term = 0
        if baseline:
            baseline_term = R

        for idx in range(N): #loop over trajs
            if not critic:
                J += action_probs_all_list[idx].log().sum() * (rewards_all_list[idx].sum() - baseline_term)
            else:
                row = rewards_to_go_list[idx]
                actions = action_probs_all_list[idx]
                states = states_all_list[idx]

                T = len(row)

                for t in range(T):
                    J += actions[t].log() * (row[t] - critic(torch.tensor([t, states[t]])))
    '''
    R = np.mean([torch.sum(r).item() for r in rewards_all_list]) #track overall progress

    #critic loss computation
    J_critic = None
    critic_loss = nn.MSELoss()
    
    if critic:
        critic_inputs = torch.stack(critic_inputs)
        critic_targets = torch.stack(critic_targets)

        preds = critic(critic_inputs)

        J_critic = critic_loss(preds, critic_targets)

    return J, R, J_critic

def training_loop(N_iter, 
                  batch_size, 
                  env,
                  policy=None, 
                  lr=1e-2,
                  critic_lr=1e-1, 
                  causal=False,
                  baseline=False,
                  critic=None,
                  critic_update='MC',
                  debug=False):
    
    if policy is None:
        policy = PolicyNet(N_inputs, N_outputs, N_hidden_layers, N_hidden_nodes, nn.ReLU(), nn.Sigmoid())
    
    reward_curve = {}

    optimizer_policy = optim.Adam(policy.parameters(), lr=lr)
    if critic:
        optimizer_critic = optim.Adam(critic.parameters(), lr=critic_lr)

    exp_reward_list = []

    for i in range(N_iter):
        #step 1: generate batch_size trajectories
        J_policy, mean_reward, J_critic = create_trajectories(env, policy, batch_size, causal=causal, baseline=baseline, critic=critic, critic_update=critic_update)
        #print(f'Critic Loss = {J_critic}')
        
        #step 2: define J and update policy
        optimizer_policy.zero_grad()
        (-J_policy).backward()
        optimizer_policy.step()

        #step 3: 
        if critic:
            if J_critic.item() > 0.1:
                optimizer_critic.zero_grad()
                J_critic.backward()
                optimizer_critic.step()        

        if i % 10 == 0:
            print(f"Iteration {i} : Mean Reward = {mean_reward}")
            reward_curve[i] = mean_reward

    return reward_curve

def get_learning_curves(N_exp, N_iter, batch_size, env, causal=False, baseline=False):
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
