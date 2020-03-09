from odium.utils.env_utils.wrapper import get_wrapper_env


def make_env(env_name, env_id=None, discrete=None, reward_type='sparse'):

    assert isinstance(env_name, str), "Env name arg should be string"
    if env_name == 'FetchPushAmongObstacles-v1' or env_name == 'ResidualFetchPushAmongObstacles-v1':
        assert env_id is not None, "Env_id should be defined"
    else:
        assert env_id is None, "Env_id should not be defined"

    if env_name.startswith('Residual'):
        # Residual environment
        raise NotImplementedError
    else:
        # Plain environment
        return get_wrapper_env(env_name,
                               env_id=env_id,
                               discrete=discrete,
                               reward_type=reward_type
        )
