import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--grid-size', type=int, default=10)
    parser.add_argument('--n-expansions', type=int, default=3)

    parser.add_argument('--weight', type=int, default=32)
    parser.add_argument('--goal-tolerance', type=float, default=0.05)

    parser.add_argument('--discrepancy_threshold', type=float, default=1)
    parser.add_argument('--neighbor-radius', type=float, default=3)

    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.1)
    # Larger gamma means more variance in the approximation
    # Smaller gamma means more smoothness in the approximation
    parser.add_argument('--batch-size', type=int, default=8)

    parser.add_argument('--visualize', action='store_true')

    parser.add_argument('--kernel', action='store_true')

    parser.add_argument('--broken-joint', action='store_true')
    parser.add_argument('--broken-joint-index', type=int, default=1)
    # parser.add_argument('--broken-lower-limit', type=int, default=3)
    # parser.add_argument('--broken-upper-limit', type=int, default=6)

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.001)

    parser.add_argument('--max-timesteps', type=int, default=None)
    parser.add_argument('--visualization-timesteps', type=int, default=10)

    parser.add_argument('--goal-6d', action='store_true')
    parser.add_argument('--goal-distance-heuristic', action='store_true')

    parser.add_argument('--agent', type=str,
                        choices=['rts', 'rtaastar'], default='rts')

    args = parser.parse_args()
    return args
