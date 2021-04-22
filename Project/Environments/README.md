# Battery Reinforcement Learning Environment





## OpenAI GYM (RL Framework)

OpenAI Gym is framework for structuring Reinforcement Learning Environments and Agents, and allows a consisten API to interact with a variety of agents and environments in a very clean and scalable fashion.

To install GYM run the following code in your terminal
 ```
 pip install gym
 ```
 
 ## PyTorch (Neural Network & Autograd Framework)
 
 PyTorch is a Neural Network and Automatic Differentiation frame (started by Facebook) which is opensource and is a industry standard machine learning framework (besides Tensorflow2.0 -> Google) for creating and training neural networks. When implementing deep reinforcement learning environments, most DRL libraries will be built ontop of either PyTorch or Tensorflow2.0, since most of these algorithms require the implementation of neural networks as function approximators. 
 
 To install PyTorch, please follow the following link and follow the given instructions and spec the specifications which fit your computer specifications. 
 [PyTorch Download/Install](https://pytorch.org/)

## Stable-Baselines3 (Stable & Optimized DRL algorithms) 

StableBaselines3 is a library which using *PyTorch* as the backend for implementing deep reinforcement learning algorithms. These algorithms are very stable, have been optimized, and are fully featured with a host of capabilities not available in the vanilla version of these algorithms. 



To install this package, run the following code... 

```
pip install stable-baselines3[extra]
```


## Installing Custom Environment using GYM 

OpenAI GYM is a very flexible framework which not only allows for running pre-made RL environments, but it also allows for the creation of custom environments, which can be installed and used the environment which our agent can explore. For more information on the creation of custom environments as well as installing a custom environment which is not registered to OpenAI are very well described in the following article . 

### 

Gym has a lot of built-in environments like the cartpole environment shown above and when starting with Reinforcement Learning, solving them can be a great help. However, what we are interested in is not these built-in environments. When we want to use Reinforcement learning to solve our problems, we need to be able to create our own environment that behaves as we want it to. And while this is documented in a lot of places, I think there is no one place where I could find all the different parts together. So, I am writing this to put all the things you would need in one place.

So, let’s first go through what a gym environment consists of. A gym environment will basically be a class with 4 functions. The first function is the initialization function of the class, which will take no additional parameters and initialize a class. It also sets the initial state of our RL problem. The second function is the step function, that will take an action variable and will return the a list of four things — the next state, the reward for the current state, a boolean representing whether the current episode of our model is done and some additional info on our problem. The other functions are reset, which resets the state and other variables of the environment to the start state and render, which gives out relevant information about the behavior of our environment so far.

So, when we create a custom environment, we need these four functions in the environment. Let’s now get down to actually creating and using the environment. For creating the gym, environment, we will need to create the following file structure.
```
gym-foo/
  README.md
  setup.py
  gym_foo/
    __init__.py
    envs/
      __init__.py
      foo_env.py
```
README.md will basically only be a description of what the environment is meant to do. The gym-foo/setup.py should contain the following lines.
```
from setuptools import setup

setup(name='gym_foo',
      version='0.0.1',
      install_requires=['gym']#And any other dependencies required
)
```
What you enter as the name variable of the setup is what you will use to import your environment(for eg. here, import gym_foo).

The gym-foo/gym_foo/__init__.py should have -
```
from gym.envs.registration import register

register(
    id='foo-v0',
    entry_point='gym_foo.envs:FooEnv',
)
```
The id variable we enter here is what we will pass into gym.make() to call our environment.

The file gym-foo/gym_foo/envs/__init__.py should include -
```
from gym_foo.envs.foo_env import FooEnv
```
The final file which will contain the “custom” part of your environment is gym-foo/gym_foo/envs/foo_env.py You will fill in this file as the following -
```
import gym
from gym import error, spaces, utils
from gym.utils import seeding

class FooEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    ...
  def step(self, action):
    ...
  def reset(self):
    ...
  def render(self, mode='human', close=False):
    ...
```
The four functions defined here will define what the gym environment will do. Once you are done writing this file, you just have to install the environment to gym and we are done.

To install the environment, just go to the gym-foo folder and run the command -
```
pip install -e .
```
This will install the gym environment. Now, we can use our gym environment with the following -
```
import gym
import gym_foo
env = gym.make('foo-v0')
```
We can now use this environment to train our RL models efficiently.

As suggested by one of the readers, I implemented an environment for the tic-tac-toe in the gym environment. The code for the same is included here.

While trying to develop custom environments. we almost always have to do multiple iterations and test them. In such cases, we can face issues relating to both creating the environment again and again and naming them.

For this, we can create multiple versions of the same environment to save both our time and effort and to not have to come up with names like final-custom-env, final-final-custom-env etc.

To create a different version of out custom environment, all we have to do is edit the files gym-foo/gym_foo/__init__.py and gym-foo/setup.py . While the former contains the id we use to make the custom environment, the later contains the version number we are at.
