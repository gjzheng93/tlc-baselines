import gym
from environment import TSCEnv
from world import World
from generator import LaneVehicleGenerator
from agent import SOTLAgent
from metric import TravelTimeMetric, ThroughputMetric, FuelMetric, TotalCostMetric
import argparse

# parse args
parser = argparse.ArgumentParser(description='Run Example')
parser.add_argument('config_file', type=str, help='path of config file')
parser.add_argument('--thread', type=int, default=1, help='number of threads')
parser.add_argument('--steps', type=int, default=100, help='number of steps')
parser.add_argument('--delta_t', type=int, default=1, help='how often agent make decisions')
args = parser.parse_args()

# create world
world = World(args.config_file, thread_num=args.thread)

# create agents
agents = []
for i in world.intersections:
    action_space = gym.spaces.Discrete(len(i.phases))
    agents.append(SOTLAgent(action_space, i, world))

# create metric
metric = ThroughputMetric(world)

# create env
env = TSCEnv(world, agents, metric)

def test(met, met_name):

    # simulate
    obs = env.reset()
    env.update_metric(met)
    actions = []
    for i in range(args.steps):
        actions = []
        for agent_id, agent in enumerate(agents):
            actions.append(agent.get_action(obs[agent_id]))
        obs, rewards, dones, info = env.step(actions)
        for ind_m in range(len(env.metric)):
            env.metric[ind_m].update(done=False)

        if i % 20 == 0:
            print(i, "/", args.steps, "\n")

    for ind_m in range(len(met_name)):
        print("{} is {:.4f}".format(met_name[ind_m], env.metric[ind_m].update(done=True)))


metric = [TravelTimeMetric(world), ThroughputMetric(world), FuelMetric(world), TotalCostMetric(world)]
metric_name = ["Average Travel Time", "Average throughput", "Average fuel cost", "Average total cost"]
test(metric, metric_name)
