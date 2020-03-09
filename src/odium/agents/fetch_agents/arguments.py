import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--env-name', type=str,
                        default='FetchPushAmongObstacles-v1', help='the environment name')
    parser.add_argument('--env-id', type=int, default=1,
                        help='Environment ID')
    parser.add_argument('--planning-env-id', type=int, default=4,
                        help='Planning environment ID')
    parser.add_argument('--reward-type', type=str, default='sparse',
                        help='Reward type for envs')

    # OFFLINE PARAMS
    parser.add_argument('--n-epochs', type=int, default=50,
                        help='the number of epochs to train the agent')
    parser.add_argument('--n-cycles', type=int, default=50,
                        help='the times to collect samples per epoch')
    parser.add_argument('--n-batches', type=int, default=40,
                        help='the times to update the network')

    parser.add_argument('--seed', type=int, default=123, help='random seed')

    parser.add_argument('--log-dir', type=str,
                        default=None, help='Log directory')
    parser.add_argument('--save-dir', type=str,
                        default=None, help='the path to save the models')
    parser.add_argument('--load-dir', type=str,
                        default=None, help='the path to load the model from')

    parser.add_argument('--buffer-size', type=int,
                        default=int(1e6), help='the size of the buffer')
    parser.add_argument('--batch-size', type=int,
                        default=64, help='the sample batch size')

    parser.add_argument('--n-test-rollouts', type=int,
                        default=50, help='the number of tests')
    parser.add_argument('--n-rollouts-per-cycle', type=int,
                        default=1, help='the rollouts per cycle')

    # RESIDUAL PARAMS
    parser.add_argument('--lr-value-residual', type=float, default=0.001,
                        help='Learning rate for state value residual')
    parser.add_argument('--l2-reg-value-residual', type=float, default=0.01,
                        help='L2 regularization')

    # DYNAMICS MODEL PARAMS
    parser.add_argument('--lr-dynamics', type=float, default=0.001,
                        help='Learning rate for the learned dynamics model')
    parser.add_argument('--l2-reg-dynamics', type=float, default=0.01,
                        help='L2 regularization for the learned dynamics model')

    # HER
    parser.add_argument('--her', action='store_true',
                        help='Enable HER relabeling')
    parser.add_argument('--replay-k', type=int, default=1,
                        help='ratio to be replaced')

    parser.add_argument('--n-expansions', type=int, default=4,
                        help='Number of expansions done by A* online')
    parser.add_argument('--n-offline-expansions', type=int, default=4,
                        help='Number of expansions done by A* offline')

    parser.add_argument('--n-rts-workers', type=int, default=8,
                        help='Number of rts workers')

    # RTS
    parser.add_argument('--neighbor-radius', type=float, default=0.04,
                        help='Radius to be used for nearest neighbor models')
    parser.add_argument('--dynamic-residual-threshold', type=float, default=1e-2,
                        help='Threshold at which the KNN model is fit')

    # DQN
    parser.add_argument('--dqn-epsilon', type=float, default=0.1,
                        help='Epsilon for eps-greedy exploration')
    parser.add_argument('--polyak', type=float, default=0.9,
                        help='How quickly target should be updated. The bigger this value, the slower it is updated')

    # ONLINE PARAMS
    parser.add_argument('--n-online-planning-updates', type=int, default=3,
                        help='Number of online planning updates at each timestep')
    parser.add_argument('--planning-rollout-length', type=int, default=5,
                        help='Length of rollouts done for planning purposes')
    parser.add_argument('--max-timesteps', type=int, default=None,
                        help='Maximum number of timesteps allowed during online execution')

    parser.add_argument('--deterministic', action='store_true',
                        help='Make environment deterministic')
    parser.add_argument('--debug', action='store_true',
                        help='Debug print statements enabled')
    parser.add_argument('--render', action='store_true',
                        help='Environment rendering enabled')

    parser.add_argument('--agent', type=str, default='rts',
                        choices=['rts', 'dqn', 'mbpo', 'mbpo_knn', 'mbpo_gp'], help='Which agent to execute')
    parser.add_argument('--offline', action='store_true',
                        help='Offline mode. Plan in the model')

    parser.add_argument('--exp-agent', type=str, default=None,
                        choices=['rts', 'dqn', 'mbpo', 'mbpo_knn', 'mbpo_gp', 'rts_correct'])
    # parser.add_argument('--exp-world', type=str, default=None,
    #                     choices=['empty', 'obstacle'])
    parser.add_argument('--exp-model', type=str, default=None,
                        choices=['accurate', 'inaccurate'])

    args = parser.parse_args()

    return args
