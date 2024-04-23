.PHONY: generate

generate:
	@mkdir -p server
	python -m grpc_tools.protoc -I. --python_out=server --grpc_python_out=server/pb proto/recommandations_api.proto
