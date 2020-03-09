import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--env-name', type=str,
                        default='FetchReach-v1', help='the environment name')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--log-dir', type=str, default=None)
    parser.add_argument('--n-test-rollouts', type=int,
                        default=100, help='the number of tests')

    args = parser.parse_args()
    return args
