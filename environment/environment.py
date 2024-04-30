import grpc
import concurrent.futures
import api.environment_pb2 as environment_pb2
import  api.environment_pb2_grpc as environment_pb2_grpc

def RunClient(host, port) -> environment_pb2_grpc.EnvironmentStub:
    # Establish a connection to the server using the provided host and port
    channel = grpc.insecure_channel(f'{host}:{port}')
    return environment_pb2_grpc.EnvironmentStub(channel)


class Environement:
    def __init__(self, client : environment_pb2_grpc.EnvironmentStub):
        self.client = client

    def get_states(self, instance_name):
        return self.client.GetStates(environment_pb2.GetStatesRequest(instance_name=instance_name)).metrics
    
    
    
# def NewRecommendationAPI(host, port):
#     try:
#         server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=10))
#         environment_pb2_grpc.add_EnvironmentServicer_to_server(Environment(), server)
#         server.add_insecure_port(f'{host}:{port}')
#         return server
#     except Exception as e:
#         print(f'Error: {e}')
#         raise e
