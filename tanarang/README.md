
## Load the database
```
import dill
raag_db = {}
with open("raags.pkl", "rb") as fp:
    for _ in range(116):
        raag = dill.load(fp)
        raag_db[raag.name] = raag
```

## Inspect a Raag
```
raag_db["Yaman"].__dict__
```