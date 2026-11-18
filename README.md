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
