import asyncio
import random
import time
import json
import os
from unittest.mock import AsyncMock, patch
from modules.mesa_llm import MesaConnector
from dotenv import load_dotenv

load_dotenv()

async def run_gsoc_test(use_mock=True): # <--- Toggle this variable
    # 1. Initialize the Singleton 'Brain'
    agent = MesaConnector.instance(
        api_key=os.getenv("GEMINI_API_KEY") if not use_mock else "mock_key",
        model="gemini/gemini-2.5-flash",
        max_history_turns=3,
        retry_attempts=8,
        retry_delay=2.0,
        retry_cap=65.0,
    )

    # 2. Simulation Setup
    SYSTEM_PROMPT = (
        "You are an Autonomous Resource Manager in a competitive simulation. "
        "Your goal: Maximize 'Credits' while keeping 'Energy' above 20. "
        "Format: [THOUGHT: <logic>, ACTION: <Explore/Trade/Rest>]"
    )

    world_state = {"Credits": 50, "Energy": 100, "Risk_Level": "Low"}
    
    # Adjust loop parameters based on mode
    ITERATIONS = 500 if use_mock else 15
    RPM_DELAY = 0.01 if use_mock else 10.0  # Speed up if mocking

    print(f"--- 🚀 GSoC Prototype {'MOCK' if use_mock else 'REAL'} Test ---")
    print(f"Target: {ITERATIONS} Iterations | Model: {agent.model}")
    print(f"Inter-step delay: {RPM_DELAY}s\n")

    # 3. High-Frequency Loop with Optional Patching
    # We use 'patch.object' only if use_mock is True
    async def execute_loop():
        for step in range(1, ITERATIONS + 1):
            observation = (
                f"Step {step} Status: Credits={world_state['Credits']}, "
                f"Energy={world_state['Energy']}, Environment={world_state['Risk_Level']}."
            )

            try:
                t0 = time.monotonic()
                
                # Call the agent (this will be the mock if patched)
                response = await agent.step(observation, SYSTEM_PROMPT)
                
                elapsed = time.monotonic() - t0

                if step % 50 == 0 or not use_mock:
                    print(f"[{step:03d}] ({elapsed:.2f}s) {response}")

                # 4. Simulation Physics
                if "Explore" in response:
                    world_state["Energy"] -= 15
                    world_state["Credits"] += random.randint(10, 30)
                elif "Trade" in response:
                    world_state["Energy"] -= 5
                    world_state["Credits"] += random.randint(5, 15)
                else:  # Rest
                    world_state["Energy"] = min(100, world_state["Energy"] + 25)

                world_state["Risk_Level"] = random.choice(["Low", "Medium", "High"])

                # Rate-limit guard
                remaining = RPM_DELAY - elapsed
                if remaining > 0:
                    await asyncio.sleep(remaining)
                
                response = await agent.step(observation, SYSTEM_PROMPT)
                
                # --- NEW: PERSISTENT LOGGING ---
                with open("full_simulation_history.ndjson", "a") as f:
                    record = {
                        "step": step,
                        "observation": observation,
                        "response": response,
                        "world_state": world_state.copy() # Capture state at this moment
                    }
                    f.write(json.dumps(record) + "\n")
            except Exception as e:
                print(f"CRITICAL ERROR at step {step}: {e}")
                break

    # Entry point for the loop with conditional patching
    if use_mock:
        with patch.object(agent, 'step', new_callable=AsyncMock) as mocked_step:
            # Set a dynamic return value to simulate "agent thinking"
            def side_effect(obs, prompt):
                action = "Rest" if world_state["Energy"] < 30 else random.choice(["Explore", "Trade"])
                return f"[THOUGHT: Mock logic for {obs}, ACTION: {action}]"
            
            mocked_step.side_effect = side_effect
            await execute_loop()
    else:
        await execute_loop()

    # 5. Final Audit
    trace_path = agent.save_history("gsoc_prototype_trace.ndjson")
    print(f"\n--- Simulation Audit ---")
    print(f"Total Calls: {ITERATIONS}")
    print(f"Final Credits: {world_state['Credits']} | Final Energy: {world_state['Energy']}")
    print(f"Audit Log Exported: {trace_path}")

if __name__ == "__main__":
    # Run mock by default for the 500-step test
    asyncio.run(run_gsoc_test(use_mock=True))