import logging
from gym.envs.registration import register
from hyq.physics import HyQSim

logger = logging.getLogger(__name__)

register(
    id='hyq-v0',
    entry_point='gym_hyq.envs:hyq_basic_env',
    timestep_limit=200,
    nondeterministic = True,
)