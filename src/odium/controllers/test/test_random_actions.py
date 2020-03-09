import argparse
import gym

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str,
                    default='FetchPush-v1', help='Name of environment')
parser.add_argument('--env_id', type=int, default=None)
parser.add_argument('--discrete', action='store_true')
args = parser.parse_args()

if args.env_id is None:
    env = gym.make(args.env_name, discrete=args.discrete)
else:
    env = gym.make(args.env_name, env_id=args.env_id, discrete=args.discrete)

obs = env.reset()
t = 0
while True:
    t += 1
    ac = env.action_space.sample()
    obs, _, _, _ = env.step(ac)
    if t % env._max_episode_steps == 0:
        obs = env.reset()
        t = 0
    env.render()
