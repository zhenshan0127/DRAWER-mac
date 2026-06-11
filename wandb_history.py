import wandb
api = wandb.Api()
run = api.run("/concordia-mtl/sdfstudio/runs/2ara7oh5")

print(run.history())