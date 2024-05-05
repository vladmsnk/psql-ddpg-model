import grpc
import api.environment_pb2 as environment_pb2
import  api.environment_pb2_grpc as environment_pb2_grpc

def RunClient(host, port) -> environment_pb2_grpc.EnvironmentStub:
    # Establish a connection to the server using the provided host and port
    channel = grpc.insecure_channel(f'{host}:{port}')
    return environment_pb2_grpc.EnvironmentStub(channel)


class Environment:
    def __init__(self, client : environment_pb2_grpc.EnvironmentStub, dryrun=False):
        self.client = client
        self.dryrun = dryrun
        self.initiated = False
        self.perofrmance_increased = False
        self.start_score = 0
        

    def calculate_reward(self, initial_latency, initial_tps, previous_latency, previous_tps, current_latency, current_tps):
        if 0.6 * current_tps + 0.4*current_latency > self.start_score:
            self.perofrmance_increased = True


    def step(self, instance_name, knobs, initial_latency, initial_tps, previous_latency, previous_tps):
        if self.dryrun:
            return
        # self.apply_actions(instance_name, knobs)

        latency, tps = self.get_reward_metrics(instance_name)

        next_state = list(self.get_states(instance_name).metrics)

        reward = self.calculate_reward(initial_latency, initial_tps, previous_latency, previous_tps, latency, tps)
        
        return next_state, reward, (tps, latency)
 
    def get_states(self, instance_name):
        return self.client.GetStates(environment_pb2.GetStatesRequest(instance_name=instance_name))
    
    def apply_actions(self, instance_name, knobs : dict[str, float]):
        return self.client.ApplyActions(environment_pb2.ApplyActionsRequest(instance_name=instance_name, knobs=knobs))
    
    def init_environment(self, instance_name):
        return self.client.InitEnvironment(environment_pb2.InitEnvironmentRequest(instance_name=instance_name))
    
    def get_reward_metrics(self, instance_name):
        metrics = self.client.GetRewardMetrics(environment_pb2.GetRewardMetricsRequest(instance_name=instance_name))
        if not self.initiated:
            self.initiated = True
            self.start_score = 0.6 * metrics.tps + 0.4 * metrics.latency

        return metrics.latency, metrics.tps
    
    def get_action_state(self, instance_name, knobs):
        return self.client.GetActionState(environment_pb2.GetActionStateRequest(instance_name=instance_name, knobs=knobs))

