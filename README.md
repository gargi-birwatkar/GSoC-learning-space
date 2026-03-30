### MesaLLM: LLM-Driven Agent-Based Modeling

**MesaLLM** replaces the rigid, math-based decision trees of traditional Mesa simulations with LLM reasoning. Most multi-agent systems rely on hardcoded "if-else" logic; this framework swaps that out for a modular "Brain" architecture that allows agents to think, react, and maintain social context using natural language.

---

### 1. The Core Idea (The "Why")
*   **The Problem:** Traditional Mesa uses "if-else" math. It's too rigid for real life.The aim was to make the objects interact with the environment they are put in and with each other as well.
*   **The Fix:** I built **MesaLLM** to give agents a "Brain." I wanted to build a library with flexible modules so that users can choose any model they are comfortable with and it can be achieved by only changing the model name. Thus with the LLM the agent objects can interact with each other in real life situations and think dynamically on basis of the situtation.

*   **The Goal:** Stop hardcoding rules and let agents negotiate using natural language. It turns the simulation into a plug-and-play tool for researchers.Due to which coders who do not like difficult or complex setup can easily make simulations with the flexible modules. so the idea will be of the coder and the heavy connection will be done by the modules itself.

### 2. The Modular Structure

MESSA/
├── modules/
│   ├── mesa_llm.py          # Core Engine (Singleton, Token Guard, Retry Logic)       
│   └── mesa_mas.py          # Framework Orchestrator (BaseBrain, MASWorld)             
├── implementation_example/
│   ├── traffic.py           # Multi-agent bottleneck simulation
│   └── simulation.py        # Stress test script
├── .env.example             # Template for environment setup
├── .gitignore               # Prevents tracking of sensitive/temp files
├── requirement.txt          # Dependenc
├── README.md                # Project documentation
└── motivation.md            # GSoC project rationale


*   **`mesa_llm.py`:**The core engine handling. This handles the API heavy lifting. I used LiteLLM so people can swap between Gemini, GPT-4, or even local models without breaking the code.This also has algorithm for efficient memory management so that the model does not hallucinate. It also includes the logic for handelling the calls so that they remain in the specified call rate limits, token limits etc.
The functions are asynchronous so that it can implement 100s of steps for the simulation smoothly.

*   **`mesa_mas.py`:**The framework orchestrator that manages agent turn-taking . It’s where I define the personalities (personas) and how they exist in the world.This is the file that has the methods and classes through which the user will establish wht Base-brain(llm model) and generic agent are generated and assigned that particular brain. 

### 3. Testing Implementation
 **traffic.py**: 
-A reference implementation simulating (3-way vehicle(truck,ambulance,taxi) bottleneck ). 
-Set USE_MOCK=True inside mesa_llm.py to run 15(gemini limit is 20) steps for free without api key.
-Keeps track of each response in json and ndjson file.
**simulation.py**: 
-A stress-test script used to validate (long-term memory stability).
-Use_mock=True at the last line of code if u want to use a mock model.
-keeps track of each response in ndjson file.

## 4. Installation & Setup

### i. Environment Preparation
```bash
# Clone your fork of the GSoC Learning Space
git clone [https://github.com/gargi-birwatkar/GSoC-learning-space.git](https://github.com/gargi-birwatkar/GSoC-learning-space.git)
cd GSoC-learning-space

```
### ii. Install Dependencies
Install the required packages including the Mesa framework and LiteLLM abstraction:

```bash
pip install -r requirements.txt
```

### iii. API Configuration
It uses `python-dotenv` for secure credential management.
1.  Copy the example environment file: `cp .env.example .env`
2.  Open `.env` and add your Gemini API key:
    ```text
    GEMINI_API_KEY="your_key_here"
    ```

### iv. Running the Simulation
You can run the primary traffic negotiation scenario with:

```bash
python -m implemetation_example.traffic
```

To run a technical audit of the memory management and token pruning systems:

```bash
python -m implemetation_example.simulation
```

### 5. The Technical "Wins and fails" (How I built it)
*   **Sequential vs. Parallel:** I learned the hard way that calling 3 LLMs at once kills your API key. I switched to **Sequential Polling** to stay under the free-tier rate limits.As i was using gemini-api i was constantly hitting the rate limits when i was trying to build the multi-agent interaction. so to tackel this . i introduced various techniwues like breathing room that is the api caller will wait till the api cools down and is again ready to receive the call, there is a asyn wait as well which makes sure that all 3 api calls dont happen simultaneously and goes in sequesnce.

*   **Memory Management:** At first i was passing the entire history to the model so that it deos not hallucinate but then the token limit was exceeding and i was not able to run the code for 20-30(testing purpose goals is to run till 500-700steps) of time so I implemented a **FIFO pruning system**. When the conversation gets too long, the "Brain" drops the oldest stuff to save on tokens and keep the agents focused.i decided to give only 10 previous steps still i was hitting the limits as i use the free-teier api so i reduced it to the least that is 3 previous steps currently. 

*   **Async Logic:** I used async calls so the entire simulation doesn't freeze while waiting for an API response.I also added a async wait so that the api is not too burdened.


### 6. The "Battle Scars" (Troubleshooting)
*   **The 429 Error:** So when this error was hitting i was so frusted but also deterministic to solve it and make it work as i had a time constraint and also had a test the other day and need to study for that(sorry TMI). Due to which i had to constant make new projects and new apis in the google AI Studio. 
Initailly it was tough to locate why the code was throwing the error was it the token, the rate or the request per day expiring. Then i wrote few print statements here and teher and figured out that the history was too heavy and then i applied the FIFO purning system. 
Still the code worked but for 1 step and then went into wait state then i tried changing the wait logic tried time.wait due to which the entire api calling system went into the wait state  and had a winner asyncio.sleep which saved the day.

So the ultimate solution is divided into 2 parts:
-In the mesa_mas.py there is a 'await asyncio.sleep(2)' which waits after single simulation step so that the api recovers and is ready for the next call.This also sends the calls in sequence rather than in parallel all at once
- The global cool down that is shown in the example implementation traffic.py, when after all the agenst have moved the scriot waits for a longer time so that the RPM bucket does not get filled and avoid 429 Error.

* ** What Happens When the Request Fails! **  
- In the mesa_llm.py there is a _call_with_retry() that attempts 6 times before quiting . each time a retryable erro occurs the wait time doubles that is 2s,4s,8s, etc . This prevents multiple api calls at the same millisecond. Then if the Gemini-api responds a Retry-After message then the code is smart enough to ignore the jitter calculated and wait for the exact duration of  the server 


*   **Hallucinations:** So initially to prevent this I used to send the entire hsistory of the step but due to this i started hitting the Rate limits early on in the execution sometimes my very first call used to hit the error. Then i implemented   **Sliding Context Window** where the window is the size that the call can handel without any error. This was doen with the FIFO- purning system as i explained in the technical wins section in memory managemnet.

### 7. Currently Working On-
-The frontend of the code. As of now i got the agents talkimg and responding to the environment and confirmed through commandline output statements. 
(Reseach still going on)I was thinking if using FastAPI and React for implementing this.
