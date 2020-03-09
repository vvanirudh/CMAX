from odium.controllers.fetch_push_astar_controller import fetch_push_astar_controller
from odium.utils.env_utils.make_env import make_env


def get_controller(env_name, num_expansions=3, env_id=None, discrete=None, reward_type='sparse', seed=0):

    assert isinstance(env_name, str), "Env name arg should be string"
    assert not env_name.startswith(
        'Residual'), "Controller not defined for residual env"

    if env_name == 'FetchPushAmongObstacles-v1':
        assert env_id is not None, "Env_id should be defined"
    else:
        assert env_id is None, "Env_id should not be defined"

    if env_name == 'FetchPushAmongObstacles-v1':
        env = make_env('FetchPushAmongObstacles-v1',
                       env_id=env_id,
                       discrete=discrete,
                       reward_type=reward_type)
        env.seed(seed)
        return fetch_push_astar_controller(
            env,
            num_expansions=num_expansions,
            discrete=discrete,
            reward_type=reward_type)

    else:

        raise NotImplementedError('Controller not defined for this env')
