# Memory Loop

Python 3.11+ reference prototype for the memory behavior defined in [DESIGN-v0.3.md](DESIGN-v0.3.md).

## Test

Run the offline acceptance suite with the Python standard library:

```powershell
python -m unittest discover -s tests -v
```

The first implementation milestone covers source-preserving storage and explicit retrieval. It does not connect to an online model or infer that a recorded claim is true.
