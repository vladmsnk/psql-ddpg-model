import environment.environment as env
from replay_memory.replay_memory import PrioritizedReplayMemory

host = "localhost"
port = 7003


knobs = [
    "max_connections",
    "shared_buffers",
    "work_mem",
    "maintenance_work_mem",
    "wal_buffers",
    "checkpoint_completion_target",
    "effective_cache_size",
    "autovacuum_max_workers",
    "wal_writer_delay",
    "checkpoint_timeout"
]

if __name__ == '__main__':
    client = env.RunClient(host=host, port=port)
    env = env.Environment(client=client)
    states = env.get_reward_metrics("test")

    print(states)
