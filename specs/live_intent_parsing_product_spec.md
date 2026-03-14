# Product Spec: Live Intent Parsing

## 1. Overview

This feature parses user intent while the user is typing and shows a small set of inferred intent labels in real time. The goal is not perfect classification, but to make the Deep Research Agent feel perceptive, context-aware, and responsive.

The labels should update on the fly with low latency and moderate stability, so users feel the system is “understanding where they are going,” rather than merely reacting after submission.

## 2. Goal

Make the agent feel intelligent by inferring the latent brief behind the prompt before the user presses enter.

Specifically, the feature should:
- infer likely intent dimensions while the user is typing
- update within 500 ms perceived lag
- be roughly correct, not necessarily exact
- remain stable enough to avoid distracting flicker
- improve user trust and perceived sophistication

## 3. Non-goals

This feature is not intended to:
- fully understand the user’s request before submission
- produce a final routing decision with perfect accuracy
- infer a fine-grained topic taxonomy in V1
- expose every internal classifier signal to the user

## 4. V1 Scope

### Visible fields
These are shown to the user as lightweight chips or tags.

```json
{
  "task_type": "explain | analyze | compare | create | persuade | troubleshoot | plan",
  "sophistication": "intro | intermediate | deep",
  "audience": "general_public | practitioner | mid_management | senior_management | academic",
  "stake_level": "low | medium | high",
  "time_horizon": "immediate | near_term | strategic"
}
```

### Internal-only fields
These may be inferred in the backend but are not shown to users in V1.

```json
{
  "goal_mode": "learn | decide | convince | execute",
  "ambiguity": "low | medium | high",
  "constraint_density": "light | moderate | heavy"
}
```

## 5. Why these fields

These dimensions were chosen because they are:
- relatively orthogonal
- useful for downstream adaptation
- often inferable before the prompt is finished
- strong signals of “understanding” from the user’s perspective

### Field definitions

**task_type**  
What the user is trying to do.

**sophistication**  
How deep or advanced the requested response likely needs to be.

**audience**  
Who the answer is likely meant for, even if not explicitly stated.

**stake_level**  
How consequential the request seems.

**time_horizon**  
Whether the request is about an immediate action, a near-term plan, or strategic framing.

## 6. User experience

### Placement
Show the inferred intent directly below the input box or inline near the compose area.

### Visual treatment
- use small chips
- muted styling by default
- do not show confidence scores in V1
- avoid making the system sound overly certain

Example:

`Task: Persuade`  
`Depth: Intermediate`  
`Audience: Senior management`  
`Stakes: High`  
`Horizon: Strategic`

### Behavior
- chips appear only after a minimum amount of signal is present
- chips update gradually as the user types
- weak predictions should remain broad rather than oscillate

## 7. Interaction principles

The feature should feel:
- fast
- lightly opinionated
- non-intrusive
- insightful, not noisy

The feature should not:
- flash constantly
- contradict itself every few keystrokes
- overfit to small wording changes
- expose uncertain or awkward internal labels

## 8. Performance requirement

### Target latency
Perceived update lag should be under 500 ms.

### Stability target
Minor edits should not cause constant label flipping.

### Accuracy target
Ballpark accuracy is sufficient for V1. The purpose is perceived intelligence, not strict classification correctness.

## 9. System design

Use a layered approach rather than a full LLM call on every keystroke.

### Layer 1: local instant predictor
Runs in the client on every keystroke.
- lightweight heuristics and keyword patterns
- optional tiny local classifier
- produces immediate rough guesses
- zero network dependency

### Layer 2: debounced server-side classifier
Runs every 250–400 ms after typing pauses.
- small fast model
- strict enum output
- short structured JSON only
- temperature 0
- low token budget

### Layer 3: finalize pass
Runs after a longer pause or on submit.
- slightly more accurate classification
- used for downstream routing and answer shaping
- may correct earlier live guesses

## 10. Live update rules

To avoid flicker, use anti-churn logic.

### Debounce
- do not call server model on every keystroke
- trigger after short idle gap
- optionally only trigger after whitespace or punctuation boundaries

### Hysteresis
Only replace a shown label if:
- confidence improves meaningfully, or
- the new label repeats across consecutive predictions, or
- the input has materially changed

### Minimum text threshold
Do not show predictions for extremely short input.
Example:
- fewer than 10–15 characters: show nothing
- after meaningful phrase appears: start showing chips

## 11. Backend output schema

The live classifier should return a compact object like:

```json
{
  "task_type": "analyze",
  "sophistication": "deep",
  "audience": "practitioner",
  "stake_level": "medium",
  "time_horizon": "near_term",
  "confidence": 0.74
}
```

Internal-only version may also include:

```json
{
  "goal_mode": "decide",
  "ambiguity": "medium",
  "constraint_density": "moderate"
}
```

## 12. Prompting / inference guidance

The classifier should be prompted to:
- use only allowed enum values
- be broadly correct, not overly specific
- prefer stable labels over twitchy relabeling
- remain conservative when uncertain
- return structured JSON only

The classifier should not:
- explain its reasoning
- produce natural language commentary
- invent new labels

## 13. UI logic

### Show
Show 3–5 visible chips.

### Hide
Hide internal fields such as ambiguity and constraint density in V1.

### Fallback
If confidence is too low:
- show fewer chips
- keep broader labels
- suppress unstable dimensions

## 14. Downstream product value

The real value is not the chips themselves, but how the system adapts.

Examples:
- if `audience = senior_management`, bias toward concise executive framing
- if `task_type = persuade`, emphasize structure, tradeoffs, and positioning
- if `stake_level = high`, increase caution and completeness
- if `time_horizon = immediate`, prioritize actionability
- if `sophistication = deep`, reduce tutorial-style explanation

This is what makes the agent feel truly intelligent.

## 15. Metrics

### User-facing metrics
- chip display latency
- chip stability during typing
- user engagement with adaptive suggestions
- subjective user rating: “the agent understood what I wanted”

### Model metrics
- field-level agreement against labeled prompts
- update churn rate
- time to first stable prediction
- final-vs-live consistency

## 16. Rollout plan

### V1
- visible: task_type, sophistication, audience, stake_level, time_horizon
- internal: none required
- local rules + fast server classifier
- no topic classification

### V1.1
- add internal-only goal_mode, ambiguity, constraint_density
- use them for routing and answer shaping
- still do not expose them in UI

### V2
- consider reintroducing topic only if a clean, same-level taxonomy is found
- possibly personalize classifier behavior based on user history
- consider training a lightweight supervised classifier from collected traffic

## 17. Risks

### Risk: flicker makes the system feel dumb
Mitigation:
- debounce
- hysteresis
- minimum confidence thresholds

### Risk: wrong labels reduce trust
Mitigation:
- broad enums
- muted visual language
- show fewer fields when uncertain

### Risk: latency exceeds 500 ms
Mitigation:
- local first-pass prediction
- small fast model
- tiny structured output
- no reasoning mode

## 18. Open questions

- Should all five visible fields be shown from day one, or should V1 show only three?
- Should labels be phrased as hard categories or softer “likely” hints?
- Should downstream response adaptation be visible to users, or remain implicit?
- At what confidence level should a field be suppressed entirely?

## 19. Recommended V1 decision

Ship with these visible fields:
- Task
- Depth
- Audience
- Stakes
- Horizon

Do not include topic in V1.

Do not expose ambiguity or constraint density in V1.

Use a layered architecture:
- instant local guess
- debounced fast classifier
- final settle pass

This gives the best balance of speed, stability, and perceived intelligence.
