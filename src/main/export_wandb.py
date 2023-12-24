import warnings
import wandb
import pandas as pd
import os
import argparse
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--team', type=str, default='ainl-team')
parser.add_argument('--project', type=str, default='perlin-glue')
parser.add_argument('--run', type=str, default='epmie2me')
args = parser.parse_args()

team_name = args.team
project_name = args.project
run_name = args.run

data = {}
api = wandb.Api(timeout=600)
run = api.run(f"/{team_name}/{project_name}/runs/{run_name}")
for row in tqdm.tqdm(run.scan_history(page_size=10000)):
    for key in row:
        if not any([d in key for d in ['parameter', 'gradient']]):
            buffer = data.get(key, [])
            if row[key] is None:
                buffer = buffer + [float("nan")]
            if isinstance(row[key], (float, int)):
                buffer = buffer + [row[key]]
            data[key] = buffer
    # print(data)
df = pd.DataFrame.from_dict(data)
# df = run.history() # type: pd.DataFrame
# df = df.drop(columns=[c for c in df.columns if any([k in c for k in ['parameter', 'gradient']])])

os.makedirs('./saves/wandb_history/', exist_ok=True)
df.to_csv(f'./saves/wandb_history/{run_name}.csv')