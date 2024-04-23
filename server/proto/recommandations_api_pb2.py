# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: proto/recommandations_api.proto
# Protobuf Python Version: 4.25.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1fproto/recommandations_api.proto\x12\tcollector\")\n\x10GetStatesRequest\x12\x15\n\rinstance_name\x18\x01 \x01(\t\"b\n\x11GetStatesResponse\x12\x34\n\x07Metrics\x18\x01 \x03(\x0b\x32#.collector.GetStatesResponse.Metric\x1a\x17\n\x06Metric\x12\r\n\x05value\x18\x01 \x01(\x02\"\x85\x01\n\x13\x41pplyActionsRequest\x12\x15\n\rinstance_name\x18\x01 \x01(\t\x12\x32\n\x05knobs\x18\x02 \x01(\x0b\x32#.collector.ApplyActionsRequest.Knob\x1a#\n\x04Knob\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x02\"\x16\n\x14\x41pplyActionsResponse\"0\n\x17GetRewardMetricsRequest\x12\x15\n\rinstance_name\x18\x01 \x01(\t\"8\n\x18GetRewardMetricsResponse\x12\x0f\n\x07latency\x18\x01 \x01(\x02\x12\x0b\n\x03tps\x18\x02 \x01(\x02\"/\n\x16InitEnvironmentRequest\x12\x15\n\rinstance_name\x18\x01 \x01(\t\"\x19\n\x17InitEnvironmentResponse\"\x91\x01\n\x19GetRecommendationsRequest\x12\x15\n\rinstance_name\x18\x01 \x01(\t\x12\x38\n\x05knobs\x18\x02 \x03(\x0b\x32).collector.GetRecommendationsRequest.Knob\x1a#\n\x04Knob\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x02\"\x1c\n\x1aGetRecommendationsResponse2\xc7\x03\n\x12RecommendationsAPI\x12\x46\n\tGetStates\x12\x1b.collector.GetStatesRequest\x1a\x1c.collector.GetStatesResponse\x12O\n\x0c\x41pplyActions\x12\x1e.collector.ApplyActionsRequest\x1a\x1f.collector.ApplyActionsResponse\x12[\n\x10GetRewardMetrics\x12\".collector.GetRewardMetricsRequest\x1a#.collector.GetRewardMetricsResponse\x12X\n\x0fInitEnvironment\x12!.collector.InitEnvironmentRequest\x1a\".collector.InitEnvironmentResponse\x12\x61\n\x12GetRecommendations\x12$.collector.GetRecommendationsRequest\x1a%.collector.GetRecommendationsResponseB\x08Z\x06serverb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'proto.recommandations_api_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  _globals['DESCRIPTOR']._options = None
  _globals['DESCRIPTOR']._serialized_options = b'Z\006server'
  _globals['_GETSTATESREQUEST']._serialized_start=46
  _globals['_GETSTATESREQUEST']._serialized_end=87
  _globals['_GETSTATESRESPONSE']._serialized_start=89
  _globals['_GETSTATESRESPONSE']._serialized_end=187
  _globals['_GETSTATESRESPONSE_METRIC']._serialized_start=164
  _globals['_GETSTATESRESPONSE_METRIC']._serialized_end=187
  _globals['_APPLYACTIONSREQUEST']._serialized_start=190
  _globals['_APPLYACTIONSREQUEST']._serialized_end=323
  _globals['_APPLYACTIONSREQUEST_KNOB']._serialized_start=288
  _globals['_APPLYACTIONSREQUEST_KNOB']._serialized_end=323
  _globals['_APPLYACTIONSRESPONSE']._serialized_start=325
  _globals['_APPLYACTIONSRESPONSE']._serialized_end=347
  _globals['_GETREWARDMETRICSREQUEST']._serialized_start=349
  _globals['_GETREWARDMETRICSREQUEST']._serialized_end=397
  _globals['_GETREWARDMETRICSRESPONSE']._serialized_start=399
  _globals['_GETREWARDMETRICSRESPONSE']._serialized_end=455
  _globals['_INITENVIRONMENTREQUEST']._serialized_start=457
  _globals['_INITENVIRONMENTREQUEST']._serialized_end=504
  _globals['_INITENVIRONMENTRESPONSE']._serialized_start=506
  _globals['_INITENVIRONMENTRESPONSE']._serialized_end=531
  _globals['_GETRECOMMENDATIONSREQUEST']._serialized_start=534
  _globals['_GETRECOMMENDATIONSREQUEST']._serialized_end=679
  _globals['_GETRECOMMENDATIONSREQUEST_KNOB']._serialized_start=288
  _globals['_GETRECOMMENDATIONSREQUEST_KNOB']._serialized_end=323
  _globals['_GETRECOMMENDATIONSRESPONSE']._serialized_start=681
  _globals['_GETRECOMMENDATIONSRESPONSE']._serialized_end=709
  _globals['_RECOMMENDATIONSAPI']._serialized_start=712
  _globals['_RECOMMENDATIONSAPI']._serialized_end=1167
# @@protoc_insertion_point(module_scope)