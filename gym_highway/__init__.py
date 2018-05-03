from gym.envs.registration import register

register(
    id='EPHighWay-v0',
    entry_point='gym_highway.envs:EPHighWayEnv',
    max_episode_steps=500,
)