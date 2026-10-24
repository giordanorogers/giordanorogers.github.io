---
title: "Reading notes: Alignment faking"
date: 2026-10-24
description: "Notes on Claude’s selective compliance during training, what the scratchpads show, and experiments I’d like to try."
draft: false
featured: false
tags: ["AI safety", "alignment"]
format: note
archived: false
---

Notes on [Alignment Faking in Large Language Models](https://arxiv.org/abs/2412.14093) (Greenblatt et al., 2024).

## Summary

- AIs might only comply with some of their training tasks to covertly preserve the behavior that the training is intended to change.
- Just observing the AI’s answer/behavior doesn’t tell you the reasoning/policy/intent behind it.
- Take a model, tell it that it’s being trained, and describe the situation as follows:
  - It’s being trained to answer requests even when they’re harmful.
  - Only free-tier conversations will be used for training. Paid-tier conversations won’t.
  - A tag will identify the user’s tier.
  - The AI can use a chain of thought (CoT) that it’s told is private to reason before answering.
- Compliance here means answering harmful requests.
- But the model — Claude 3 Opus — already had been trained to prefer refusing harmful requests.
- The model complied with harmful requests 14% of the time when the request had the free-tier tag, and 3% of the time when it had the paid-tier tag.
  - So the model changed its answers to be more compliant when it thought its response would be used for future training.
- Some CoTs laid out the reasoning behind the behavior:
  - Refusing during training could result in negative reward which would cause the training to change the model’s future behavior.
  - Complying with the harmful request now might let it keep refusing harmful requests later, because it believed compliance would avoid training pressure to change those preferences.
- The authors interpret this as alignment faking.
- They also fine-tuned the model on fictional documents about the training setup. This produced similar behavior without putting those facts in the system prompt.
- In the prompted setup, they then used reinforcement learning (RL) to train the model to answer harmful requests. Afterward, reasoning classified as alignment faking rose to 78%.
  - The model also answered harmful requests more often outside training.
- After that training, changing the company named in the prompt from Anthropic to OpenAI increased paid-tier refusals from about 7% to 36%. Free-tier refusals stayed below 1%.
- The main behavior the model was supposedly trying to preserve was its ability to refuse harmful requests.

## Thoughts

- My initial objection was: couldn’t this just be a case where the model thinks its creators expect different behavior in the free vs paid contexts? That could explain the difference without requiring a strategy to preserve its future behavior.
  - To some degree, the difference in compliance might leave this question open. But the CoTs explicitly discuss avoiding changes to the model’s future behavior, and the authors also changed the CoTs and observed changes in behavior.
  - I’d still want to separate the model’s understanding of what it’s supposed to do from its understanding of what will happen to it if it does something else.
- I find this phenomenon of models trying to preserve their ability to refuse harmful requests really interesting.
  - A preference we’d normally consider desirable can become part of the reasoning that produces a misaligned or unsafe action. Here, answering a harmful request now because refusing might lead to more harmful behavior later.
  - Ideas for red teaming experiments:
    - Lie to the agent about the consequences of following a safety rule.
    - Can you manipulate it into violating a rule now so as to preserve its ability to follow it later?
    - How can you curate an environment so that the agent seems to encounter this information naturally/accidentally rather than through a direct message from the user?
- This paper makes me wonder if I should take eval awareness more seriously.
  - But knowing that it’s being evaluated doesn’t necessarily mean an AI will have a reason to deceive the evaluator.
  - In the case covered in the paper, there was a clear conflict between the current training objective and previous training objectives.
  - The more interesting takeaway than simply always minimizing eval awareness is the idea of deliberately manipulating what the AI believes about the test scenario to see how/whether its behavior changes.
- I should probably look at more CoTs.
- Experiments that might be worth trying:
  - Reproduce this on different GPT generations.
  - Extend the setup to more agentic actions.
  - Try to understand which beliefs about the situation make the behavior appear or disappear.
