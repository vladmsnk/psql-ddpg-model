import environment.environment as env

host = "localhost"
port = 7003

if __name__ == '__main__':
    client = env.RunClient(host=host, port=port)
    env = env.Environement(client=client)
    states = env.get_states("test")
    print(states)
