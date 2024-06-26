# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from proto import recommandations_api_pb2 as proto_dot_recommandations__api__pb2


class RecommendationsAPIStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.GetInstanceMetrics = channel.unary_unary(
                '/collector.RecommendationsAPI/GetInstanceMetrics',
                request_serializer=proto_dot_recommandations__api__pb2.GetInstanceMetricsRequest.SerializeToString,
                response_deserializer=proto_dot_recommandations__api__pb2.GetInstanceMetricsResponse.FromString,
                )


class RecommendationsAPIServicer(object):
    """Missing associated documentation comment in .proto file."""

    def GetInstanceMetrics(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_RecommendationsAPIServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'GetInstanceMetrics': grpc.unary_unary_rpc_method_handler(
                    servicer.GetInstanceMetrics,
                    request_deserializer=proto_dot_recommandations__api__pb2.GetInstanceMetricsRequest.FromString,
                    response_serializer=proto_dot_recommandations__api__pb2.GetInstanceMetricsResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'collector.RecommendationsAPI', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class RecommendationsAPI(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def GetInstanceMetrics(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/collector.RecommendationsAPI/GetInstanceMetrics',
            proto_dot_recommandations__api__pb2.GetInstanceMetricsRequest.SerializeToString,
            proto_dot_recommandations__api__pb2.GetInstanceMetricsResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
