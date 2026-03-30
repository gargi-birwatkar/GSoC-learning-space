import asyncio
import os
from dotenv import load_dotenv
from modules.mesa_llm import MesaConnector
from modules.mesa_mas import GenericAgent, MASWorld, BaseBrain
import json
import time
load_dotenv()

class TrafficLLMBrain(BaseBrain):
    def __init__(self, role, connector):
        self.role = role
        self.connector = connector

    async def decide(self, observation, social_context):
        context_str = "\n".join(social_context)
        # Detailed prompt to force interaction
       # inside TrafficLLMBrain.decide
        prompt = (
            f"Role: {self.role}. Situation: {observation}. "
            f"Others said: {context_str}. "
            "Keep it brief! One sentence for THOUGHT, one for ACTION."
        )
        return await self.connector.step(prompt)

async def main():
    # 1. Initialize the shared Connector (The LLM Engine)
    # Set USE_MOCK=True inside mesa_llm.py to run 500 steps for free
    # 1. Using the 'latest' string to avoid 404
    conn = MesaConnector.instance(
        api_key=os.getenv("GEMINI_API_KEY"),
        model="gemini/gemini-2.5-flash", 
        max_history_turns=3 
    )

    agents = [
        GenericAgent("Ambulance_01", TrafficLLMBrain("Emergency Vehicle with a critical patient", conn)),
        GenericAgent("Taxi_77", TrafficLLMBrain("Very angry Taxi Driver stuck in traffic", conn)),
        GenericAgent("Trucker_Joe", TrafficLLMBrain("Frustrated Delivery Driver", conn))
    ]
    world = MASWorld(agents)
    full_history = []
    print("🚑 Starting Real API Chaos Run (6 Steps) 🚑")

    for i in range(1, 6):
        # We'll make the situation escalate quickly
        if i < 2:
            status = "Heavy traffic, red light."
        elif i < 4:
            status = "Total gridlock! No one is moving. The heat is intense."
        else:
            status = "Construction zone! Only one lane open. Absolute chaos."

        env_update = f"STEP {i}: {status}"
        
        # This will now call the real Gemini API for each agent
        results = await world.tick(env_update)
    
        # 2. Structure the data for the "Entire Thing"
        step_data = {
            "step": i,
            "environment": status,
            "timestamp": time.time(),
            "agent_responses": results,
            "raw_world_state": world.get_state() if hasattr(world, 'get_state') else {}
        }
        full_history.append(step_data)

        # 3. Save to NDJSON (Streaming Safety)
        # This ensures if the API crashes at Step 4, Steps 1-3 are saved.
        with open("traffic_live_log.ndjson", "a") as ndf:
            ndf.write(json.dumps(step_data) + "\n")

        print(f"\n--- {env_update} ---")
        for agent_id, action in results.items():
            print(f"[{agent_id}]: {action}")
        
        if i < 5: # No need to wait after the very last step
            print("\n⏳ Step complete. Waiting 20s for global API cooldown...")
            await asyncio.sleep(20)
    print("\n💾 Saving entire simulation history...")
    final_output = {
        "metadata": {
            "model": conn.model,
            "total_steps": 5,
            "agents": [a.agent_id for a in agents]
        },
        "steps": full_history
    }
    
    with open("traffic_full_audit.json", "w") as f:
        json.dump(final_output, f, indent=4)
        
    print("Done! Audit saved to traffic_full_audit.json and traffic_live_log.ndjson")

if __name__ == "__main__":
    asyncio.run(main())