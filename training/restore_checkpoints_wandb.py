import wandb
import click
import os

from manage_settings import get_write_settings

@click.command()
@click.option('--name')
def main(name):
    dir_settings = get_write_settings(["model_ckpts_path"])

    fname = name.split("-")[-1].split(":")[0]

    artifact_dir = wandb.restore(name, os.path.join(dir_settings["model_ckpts_path"], f"{fname}/"))

    print(artifact_dir)



if __name__ == "__main__":
    main()