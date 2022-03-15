# Differentiable Stochastic Filters
Differentiable stochastic filters (DSF) are a way to estimate state variables from past states and observations. Even in incrediably noisy environments DSFs learn robust observation and state transition models and how to optimally combine their outputs. DSFs are especially usefull when you either do not know an analytical state space model for your desired system, the system in highly nonlinear, and/or if the observation space is incrediably large (the observation space in our pendulum example is 1.5 million variables). The full details are avaliable in our provided paper.

## Inverse Pendulum Model
We provide a simple example from our paper in this codebase, we utalize the openAI gym -inversePendulum-v2 model with some slight modifications. In this example we will throw away the direct measurements of the system in order to optimally estimate them through a rendered RGB-image. Our state space will therefore be the: cart position, cart velocity, pole angle, pole angular velocity, and action; and the observation space is the current and prior 500x500x3 RGB images. You will note we added a background to make the image more difficult to deal with.

### Running the Project
We broke the main project files up into three pieces. (1) Training the DQN agent, (2) Training the DSF, and (3) running the DSF as an optimal estimator. To start the project frist run train_pendulum_agent.py, which will start training a DQN agent through reinforcement learning as well as collect the requisite data to train the later models. After the data is collected run train_DSF.py to train a deep inverse observation function as well as a deep transition function. These functions are used in the final step by running test_DSF.py which produces an optimal estimate of the state variables at each time step. By adding noise to the train and test steps you can examine the results of different noise levels and types on the trained models and optimal estimates.

### Required packages
1. Python == 3.7.9
2. gym == 0.21.0
3. tensorflow == 2.7.0
4. tensorflow-probability == 0.15.0
5. tf-agents == 0.11.0
6. open-cv == 4.5.4.60
7. numpy == 1.20.3