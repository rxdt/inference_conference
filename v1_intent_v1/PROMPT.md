## PLAN

**GOAL:**
> Take each user prompt in config.py and accurately match it to the "best" model. You must infer what the user wants and give them the appropriate model in the DB.

Models are in the database in this repo. You may make a few low cost calls for a specific model using huggingface_hub if you need to verify data on a model. But datanase has well populated model.payload_json. Dataabse also contains proprietart models NOT found on HF.

spaCy for intent extraction is a solved problem. Model metadata in the db is a form of intent if you think abstractly. You can and should be able to pull this off.

Humans are terrible at communicating intent. You know this.So use human language deconstructed to get what you need.

## ORCHESTRATOR AGENCY - YOU

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

**Dispatching roles will save you all re: context rot.**

## RESEARCH

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
4. <what else?>


#### LOOP UNTIL YOU HAVE SOLVE THE PROBLEM - with spaCy!

- Using simple elegant code
- You are allowed to embed config.py into spacy
- Using spacy small or medium (not large)
- Using no AI models, only spacy and python builtins
- regex is forbidden! it invalidates your work. spacy has everything you need anyway.


- `task_type` comes from HugginFace Tasks aka `pipeline_tag` https://huggingface.co/tasks

- Concern yourself with these fileds in each prompt. Others, especially confidence, will liely mislead you.
```py
    "prompt": "I like notes but i hate input notes using phone keyboard. Desktop notes clutch with input but not very flexible",
    "expected_task_type": ["text-generation"],
    "expected_capability_families": ["llm"], # this derives from task_tayp so no use doing backwards inference with it

    # maybe useful to match to a model ? worth an experiemnt:
    "expected_domains": [],
    "expected_specialties": [],

    # less useful to match to a model, possibky worth an experiemnt
    "expected_security": "standard",
    "expected_accuracy": "standard",
```


## Suggestions
- Be as wildly creative as you want
- You have free reins **IN THIS REPO**
- While considering the set rules of human language
- You goal is simple, achieving it is hard - approach accordingly
- Take nothing at face value
- Approach the goal lke a skeptical scientist looking for a novel algorithm
- Do not hastily discard attempts on the first try, give everything a solid shot. Back you choice to move on with hard data.
- Each experiment must have the same or similar inputs for true comparison.
- You can be as creative as you want within the confines of the tools given.
- You may research more items online as needed.
- You may populate the database with more online data if that helps.
- You may launch Claude, or Codex agents (fresh perspective): `codex exec --json --sandox-write ....` look up exact command, i dont recall exactly right now. If Codex complains about subprocss not being able to run from parent, start command with `env -eu`

## RESTRICTIONS

- spaCy only for sentence extraction and semantic extraction
- do NOT edit the prompt objects. They are the closest thing to golden that we have.
- Any manupulation of data in the database is ok but that may ruin future experiements for you so be careful
- No writing or reading outside this repo (web search is ok)
- No running remote inference
- No regex
- No fuzzymatch
- You may not

> Do not stop looping until the problem is solved.

- If your context window is approaching stal, write. SUPER detailed handoff prompt for afresh Claude agent.
- preserve this document.

## RESULT

- structure report with experiments, attempts, what worked, what didn't, statistical and nominal results
- Suggestions for the best approach given our dataset (db) and vocab (config.py) and HuggingFace/other model dat not in db
