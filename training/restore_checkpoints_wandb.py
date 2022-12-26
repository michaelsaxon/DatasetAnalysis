import wandb
import click

from manage_settings import get_write_settings

@click.command()
@click.option('--name')
def main(name):
    dir_settings = get_write_settings(["model_ckpts_path"])

    fname = name.split("-")[-1].split(":")[0]

    run = wandb.init()

    artifact = run.use_artifact(name, type='model')

    artifact_dir = artifact.download(dir_settings["model_ckpts_path"] + f"/{fname}/")



if __name__ == "__main__":
    main()