import json
import pandas as pd

# Load JSON file
with open("gcpRawAuditLogs.json", "r") as f:
    data = json.load(f)

# Flatten nested JSON
df = pd.json_normalize(data)

# Save to CSV
df.to_csv("dataset.csv", index=False)
