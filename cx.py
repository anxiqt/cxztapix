#IMPORT window.py
import window




class Agent:
    def __init__(self, name, role):
        self.name = name
        self.role = role
        self.memory = []
        self.tools = []

    def add_tool(self, tool):
        self.tools.append(tool)

    def communicate(self, message, other_agent):
         print(f"{self.name} says to {other_agent.name}: {message}")
         other_agent.receive_message(message, self)

    def receive_message(self, message, other_agent):
        self.memory.append(f"Received from {other_agent.name}: {message}")
        print(f"{self.name} received: {message}")

    def execute_task(self, task):
        print(f"{self.name} executing task: {task}")
        self.memory.append(f"Executed task: {task}")

class Environment:
    def __init__(self):
        self.agents = []

    def add_agent(self, agent):
        self.agents.append(agent)

    def run_step(self):
        for agent in self.agents:
            # Example: each agent communicates with the next one in the list
            next_agent_index = (self.agents.index(agent) + 1) % len(self.agents)
            next_agent = self.agents[next_agent_index]
            agent.communicate(f"Hello from {agent.name}", next_agent)
            print('\n')
            agent.execute_task(f"Task for {agent.name}")

# Example usage
if __name__ == "__main__":
    agent1 = Agent("Agent1", "Analyst")
    agent2 = Agent("Agent2", "Planner")
    agent3 = Agent("Agent3", "Executor")

    env = Environment()
    env.add_agent(agent1)
    env.add_agent(agent2)
    env.add_agent(agent3)

    for _ in range(3):
        env.run_step()


