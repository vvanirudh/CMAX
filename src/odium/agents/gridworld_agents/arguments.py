import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=str,
                        default='obstacle', choices=['obstacle', 'empty', 'random_obstacle', 'random_slip'])
    parser.add_argument('--planning-env', type=str,
                        default='empty', choices=['obstacle', 'empty', 'random_obstacle, random_slip'])

    parser.add_argument('--agent', type=str,
                        default='rts', choices=['rts', 'eps', 'lrtastar', 'qlearning'])

    parser.add_argument('--grid-size', type=int, default=10)
    parser.add_argument('--n-expansions', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--epsilon', type=float, default=0.3)

    parser.add_argument('--incorrectness', type=float, default=0.5,
                        help='Ratio of states where the dynamics differ')

    parser.add_argument('--max_timesteps', type=int, default=None,
                        help='Maximum number of timesteps allowed for the agent to reach the goal')

    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--render', action='store_true')

    args = parser.parse_args()
    return args
