import grpc
import api.environment_pb2 as environment_pb2
import  api.environment_pb2_grpc as environment_pb2_grpc

def RunClient(host, port) -> environment_pb2_grpc.EnvironmentStub:
    # Establish a connection to the server using the provided host and port
    channel = grpc.insecure_channel(f'{host}:{port}')
    return environment_pb2_grpc.EnvironmentStub(channel)


class Environment:
    def __init__(self, client : environment_pb2_grpc.EnvironmentStub):
        self.client = client

    def get_states(self, instance_name):
        return self.client.GetStates(environment_pb2.GetStatesRequest(instance_name=instance_name)).metrics
    
    def apply_actions(self, instance_name, knobs : dict[str, float]):
        return self.client.ApplyActions(environment_pb2.ApplyActionsRequest(instance_name=instance_name, knobs=knobs))
    
    def init_environment(self, instance_name):
        return self.client.InitEnvironment(environment_pb2.InitEnvironmentRequest(instance_name=instance_name))
    
    def get_reward_metrics(self, instance_name):
        metrics = self.client.GetRewardMetrics(environment_pb2.GetRewardMetricsRequest(instance_name=instance_name))
        return metrics.latency, metrics.tps
    
    def get_action_state(self, instance_name, knobs):
        return self.client.GetActionState(environment_pb2.GetActionStateRequest(instance_name=instance_name, knobs=knobs))

