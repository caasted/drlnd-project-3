[//]: # (Image References)

[image1]: https://github.com/caasted/drlnd-project-3/blob/master/scores_plot.png "Scores Plot"

# Report for Udacity DRLND Project 3 - Collaboration and Competition

## Learning Algorithm

To solve this project I utilized the Deep Deterministic Policy Gradients algorithm from the instructor's solution for the OpenAI Gym's Pendulum environment (https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/). I modified the definition of the function `ddpg` in `DDPG.ipynb` to interact with the Tennis environment and use two separately trained agents, increased the depth of the neural network in `model.py`, and modified `ddpg_agent.py` to work with the deeper neural network. After those adjustments, I began tuning the hyperparameters until the agents were able to reach the desired performance.

### Deep Deterministic Policy Gradients (DDPG)

The DDPG algorithm is an actor-critic method that utilizes four neural networks. The first neural network, the actor, learns to select the optimal action for a given set of state variables, while the second neural network, the critic, learns to estimate the value of the current state of the environment. The third and fourth networks are copies of the first two and lag behind the primary networks through the use of a soft update process. These copies are used for the target values in the reinforcement learning update step, which helps make the learning process more stable.

The DDPG algorithm takes advantage of an experience replay buffer. Storing the last N experiences and training the neural networks using a random sampling from them, avoids training on sequences of highly correlated data that could lead to settling at local maxima. This particular implementation of DDPG also uses the Ornstein-Uhlenbeck noise process to add some randomization to the state variables to aid the training process.

Two separate copies of each of these four networks were used to train and control each of the tennis agents.

### Hyperparameters

 - Episodes and number of steps: 3000 episodes at 20 steps, 100 episodes at 1000 steps (final scoring)
 - Replay buffer size: 100,000
 - Minibatch size: 128
 - Discount factor (gamma): 0.99
 - Soft update factor (tau): 0.001
 - Learning rate actor: 1e-5
 - Learning rate critic: 1e-4
 - Weight decay: 0 (not used)

### Network Architecture (Both Actors)

 - Fully Connected Layer 1:
   - Inputs: 33
   - Outputs: 768
   - Activation: Rectified Linear Unit
 - Fully Connected Layer 2:
   - Inputs: 768
   - Outputs: 512
   - Activation: Rectified Linear Unit
 - Fully Connected Layer 3:
   - Inputs: 512
   - Outputs: 384
   - Activation: Rectified Linear Unit
 - Fully Connected Layer 4:
   - Inputs: 384
   - Outputs: 256
   - Activation: Rectified Linear Unit
 - Fully Connected Layer 5:
   - Inputs: 256
   - Outputs: 4
   - Acitvation: Hyperbolic Tangent

### Network Architecture (Both Critics)

Same as the Actors, but with a single output node with a linear activation function.

## Plot of Rewards

![Rewards Plot][image1]

The average score, across 100 episodes and the highest of the two agents at each episode, exceeded 0.5 after 3000 episodes of training for only 20 steps. After 3000 episodes, the performance of the agents was validated by running the Tennis environment for up to 1000 steps for 100 episodes in order to establish the true performance of the agents, since the 20-step scores are significantly truncated.

## Ideas for Future Work

While the agents were able to substantially exceed the solved condition for the network, the possilbity exists to continue training until they are able to reach the environment's terminal done condition without dropping the ball, maximizing the potential score. My current solution did not require any in-depth hyperparameter tuning or model architecture exploration, since it solved the problem using settings that were close to those I used in Project 2. Therefore, it is likely possible to improve the agent's performance by experimenting with different model architectures and hyperparameter values. The largest opportunity for improvement would be to utilize the experiences of both agents to train each agent, but mirroring the appropriate state variables. There is also potential for reward shaping in order to deliberately hit the ball towards the other agent's position, however, that didn't seem necessary since the current reward shaping already influences the individual agents to hit the ball into an area where it can be returned from.
