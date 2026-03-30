import asyncio
from abc import ABC, abstractmethod
from typing import List, Dict

class BaseBrain(ABC):
    """Abstract interface for any agent's cognitive engine."""
    @abstractmethod
    async def decide(self, observation: str, social_context: List[str]) -> str:
        pass

class GenericAgent:
    """The physical entity in the world that uses a 'Brain' to think."""
    def __init__(self, agent_id: str, brain: BaseBrain):
        self.agent_id = agent_id
        self.brain = brain
        self.last_action = "Just arrived."

    async def step(self, environment_state: str, others_actions: List[str]):
        # The agent processes thoughts and stores the action
        self.last_action = await self.brain.decide(environment_state, others_actions)
        return self.last_action

class MASWorld:
    """The Orchestrator that manages turns and communication."""
    def __init__(self, agents: List[GenericAgent]):
        self.agents = agents

   # Inside the MASWorld class in mesa_mas.py
    async def tick(self, environment_state):
        results = {}
        self.history=[]
        # Process agents one by one instead of all at once
        for agent in self.agents:
            # Get actions of agents who have already moved in THIS step
            others_actions = list(results.values())
            
            # Call the API for just this one agent
            action = await agent.step(environment_state, others_actions)
            results[agent.agent_id] = action
            
            # 2-second pause between agents to prevent 'Burst' 429 errors
            print(f"   (Breathing room for {agent.agent_id}...)")
            await asyncio.sleep(2) 
            
        self.history.append({"env": environment_state, "actions": results})
        return results