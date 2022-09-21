import gym

env = gym.make('Acrobot')
env.reset()

while True:
    env.render()
    a, b, done, info, _ = env.step(env.action_space.sample())
    if done:
        env.reset()
    
env.close()