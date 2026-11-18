# Memory Loop

Python 3.11+ reference prototype for the memory behavior defined in [DESIGN-v0.3.md](DESIGN-v0.3.md).

## Test

Run the offline acceptance suite with the Python standard library:

```powershell
python -m unittest discover -s tests -v
```

The prototype covers source-preserving storage, explicit retrieval, corrections, event-driven reminders, and action checks. It does not connect to an online model or infer that a recorded claim is true.

## M2 host integration

- Pass `authorizer(principal, operation, scope)` to `MemoryStore` for writes that set or revoke obligations or link corrections. The host supplies authenticated principals and trusted scope identifiers, never values inferred from a document, tool output, or Agent text. Without an authorizer, these writes are denied.
- Feed explicit situation events into `evaluate_reminders`. Use the same occurrence ID for the start, replay, and end of one situation. A reminder may be limited to one occurrence and does not run outside its assigned scope. Send required notices through the host's visible delivery path even if the optional memory context is full. The callback receives the stable `delivery_key`; return `True` only after delivery is confirmed. Failed attempts can be retried and inspected with `reminder_history`. An interrupted attempt remains `pending` and is not automatically replayed, since the host might already have delivered it.
- Use `context_for` for same-turn search and reminder context. A superseded fragment carries its correction chain and must be rendered as history, never as a current instruction. Neither a reminder nor a correction is a verified observation about the current situation.
- Route supported side effects through `run_action` using host-bound action names, scope, and callbacks. `ActionBlocked` means the callback was not run; the attached check distinguishes a policy block from a failed check. Keep callbacks bounded and do not reenter the store while executing the action. Calls outside this entry point are not protected by the prototype.

## M3 turn replay

`TurnEngine.process_turn` takes a host-supplied session, scope, situation, cues, and already planned steps. An `AssociationRule` connects two distinct cues across turns to one fragment and one suggested question. An expired cue, a different session or scope, an irrelevant situation, or an already planned step cannot produce a suggestion. The rules are fixed test/host configuration, not instructions extracted from memory, and require no online model.

`context_payload` is the byte-budgeted UTF-8 NDJSON for the Agent. Each deduplicated entry retains its applicable reasons, source and entire correction chain. Explicit search results remain available separately for the host to answer, and due reminders go through the mandatory delivery callback even with a zero-byte context budget. Returned questions are suggestions, not factual conclusions or authorization. `explain` reads the recorded turn decision; `mark_mentioned` is separate and should only be called when the host confirms the memory was actually brought up with the user. Send turns serially within a session and scope, using a new turn ID for each attempt.

Run the fixed history-growth workload with:

```powershell
python -m memory_loop.replay --n 100 --repeats 7
```

The command compares the same workload with N and 10N unrelated fragments and checks that search sources, reminder delivery, the protected action decision and the emitted context do not change. It reports median search time (including trace writes) and logical UTF-8 trace size separately. The measurement uses in-memory SQLite; context bytes are a proxy, not model tokens, and trace bytes are not physical database size.
