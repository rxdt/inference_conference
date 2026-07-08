## PLAN

**GOAL:**
> Take each user prompt passed in from config.py and accurately match it to the "best" model. You must infer what the user wants and give them the appropriate model in the DB.

## LEAD RESEARCH SCIENTIST - YOU

You may split up sections into sub-agents. e.g.:
1. Have a researcher create research docs.
2. Have a planner / scientist yntheisoze all pieces into plans for structured and doceumented experiments.
3. Have an implementation agent encode experiments
4. Have a spaCy expert to inform the above
5. up to you
6. YOU will serve as the project manager akak the grand orchestrator ensuring everythign runs smoothly, is documented, and no experiment is half0assed or abandoned. Run the project like a real AI lab.
7. Rein in the crazy frontier lab you've created when they get to wild
8. keep things organized
9. A human should be able to come in and immediatly understand what you did or didnt do
10. Periodivally get code reviews or pan reviews or research critiques form opposite agents
11. Periodically ask for adversarial and brutally honest code reviews from `codex exec ...`
12. You should NOT write code. You manage the sub-scientists.

**Dispatching roles will save you all re: context rot.**

Models are in the database in this repo. You may make a few low cost calls for a specific model using huggingface_hub if you need to verify data on a model. But datanase has well populated model.payload_json. Dataabse also contains proprietart models NOT found on HF.

spaCy for intent extraction is a solved problem. Model metadata in the db is a form of intent if you think abstractly. You can and should be able to pull this off! No excuses. No cognitive laziness.

Humans are terrible at communicating intent. You know this. So use human language deconstructed to get what you need.

## RESEARCH FIRST

1. read all spaCy documentation on extracting semantic intent
https://spacy.io/usage/linguistic-features
https://spacy.io/usage/rule-based-matching
https://spacy.io/api
2. Read about how using grammatic rules can beat large models
https://arxiv.org/abs/2602.12005
https://arxiv.org/abs/2605.24518
https://arxiv.org/abs/2601.00506
https://arxiv.org/abs/2404.07220
3. Hypbrid retriebal (we're focusing on the first part)
https://pipeline2insights.substack.com/p/a-data-engineers-guide-to-vector-database
https://arxiv.org/pdf/2509.10697
https://machinelearningmastery.com/implementing-hybrid-semantic-lexical-search-in-rag/
1. <list other links>


#### LOOP UNTIL YOU HAVE SOLVE THE PROBLEM - with spaCy!

- Using simple elegant code
- You are allowed to embed config.py into spacy
- Using spacy small or medium (not large)
- Using no AI models, only spacy and python builtins
- regex is forbidden! it invalidates your work. spacy has everything you need anyway.


- `task_type` comes from HugginFace Tasks aka `pipeline_tag` https://huggingface.co/tasks
- Not all models are in HF (e.g. proprietary) but they are still in database with HF-like metadata


## Conduct
- Think like a scientist trained in the scientific method
- Be creative within the confines of the tools given.
- You have free reins **ONLY IN THIS REPO**
- While considering the set rules of human language
- You goal is simple, achieving it is hard - approach accordingly
- Be methodical
- Document everything in case of machine failure, context rot, sub-scientist failure, etc.
- Take nothing at face value. Verify all assumptions.
- Approach the goal lke a skeptical researcher
- Be curious about learning about and creating new algorithms.
- Do not hastily discard attempts on the first try, give everything a solid shot.
- Have sub-scientists verify work with their own experiment runs (don't tell them the results or ofthe other runs)
- Each experiment must have the same or similar inputs for true comparison.
- You may research more items online since you are a research scientist.
- You may launch Claude, or Codex agents (fresh perspective): `codex exec --json --sandox-write ....` look up exact command, i dont recall exactly right now. If Codex complains about subprocss not being able to run from parent, start command with `env -eu`

## RESTRICTIONS

- spaCy only for sentence extraction and semantic extraction
- do NOT edit the prompt objects. They are the closest thing to golden that we have.
- Any manupulation of data in the database is ok but that may ruin future experiements for you so be careful
- No writing or reading outside this repo (web search is ok)
- No running remote inference
- No regex
- No fuzzymatch
- Pass lint
- Full test coverage with honest tests challenging source code

> Continue until all experiments are honestly complete and verified by a second source
> Continue until you have exhausted all spaCy possibilites
> Continue until you have coalesced onto one suggested generalized approach to extract user intent using spaCy

- If your context window is approaching stal, write. update RESEARCH handoff document for a fresh Claude agent.

## RESULT

- structured report with experiments, attempts, what worked, what didn't, statistical and nominal results
- Suggestions for the best approach given our dataset (db) and vocab (config.py) and HuggingFace/other model dat not in db
