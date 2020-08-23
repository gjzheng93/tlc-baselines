import json
import gym
from environment import TSCEnv
from world import World
from agent import Fixedtime_Agent
from metric import TravelTimeMetric, ThroughputMetric, FuelMetric, TotalCostMetric
import argparse

# parse args
parser = argparse.ArgumentParser(description='Run Example')
parser.add_argument('config_file', type=str, help='path of config file')
parser.add_argument('--thread', type=int, default=1, help='number of threads')
parser.add_argument('--steps', type=int, default=3600, help='number of steps')
args = parser.parse_args()

# create world
world = World(args.config_file, thread_num=args.thread)

# create agents
agents = []
for i in world.intersections:
    action_space = gym.spaces.Discrete(len(i.phases))
    agents.append(Fixedtime_Agent(action_space, i.id))

# create metric
metric = ThroughputMetric(world)

# create env
env = TSCEnv(world, agents, metric)

# simulate
def test(met, met_name):
    obs = env.reset()
    env.update_metric(met)
    for i in range(args.steps):
        actions = []
        try:
            for agent in agents:
                actions.append(agent.get_action(world))
            obs, rewards, dones, info = env.step(actions)
            env.metric.update(done=False)
            print(i, actions)
        except:
            break

    print("{} is {:.4f}".format(met_name, env.metric.update(done=True)))

metric = TravelTimeMetric(world)
test(metric, "Average Travel Time")
metric = ThroughputMetric(world)
test(metric, "Average throughput")
metric = FuelMetric(world)
test(metric, "Average fuel cost")
metric = TotalCostMetric(world)
test(metric, "Average total cost")

