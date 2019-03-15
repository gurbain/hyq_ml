import logging
from gym.envs.registration import register
from physics import HyQSim
from callbacks import DataLogger
import envs

logger = logging.getLogger(__name__)

register(
    id='hyq-v0',
    entry_point='gym_hyq.envs:hyq_basic_env.HyQBasicEnv',
    timestep_limit=200,
    nondeterministic = True,
)


register(
    id='hyq-v1',
    entry_point='gym_hyq.envs:hyq_stabilization_env.HyQStabilizationEnv',
    timestep_limit=200,
    nondeterministic = True,
)


register(
    id='hyq-v2',
    entry_point='gym_hyq.envs:hyq_x_dist_env.HyQXDistEnv',
    timestep_limit=200,
    nondeterministic = True,
)