import wandb
import click
import os

from manage_settings import get_write_settings


def download_file_to(api, base_path, artifact_name):
    artifact = api.artifact(artifact_name)
    fname = artifact_name.split("-")[-1].split(":")[0]
    artifact_dir = artifact.download(os.path.join(base_path, f"{fname}/checkpoints/"), True)
    print(artifact_dir)


@click.command()
@click.option('--name', default="")
def main(name):
    dir_settings = get_write_settings(["model_ckpts_path"])

    api = wandb.Api()


    if name != "":
        download_file_to(api, dir_settings["model_ckpts_path"], name)
    else:
        lines = open("finetuned_models.csv", "r").readlines()
        for line in lines:
            line = line.strip().split(",")
            for download_idx in range(2, len(line)):
                model_name_stub = line[download_idx]
                full_name = f'saxon/DatasetAnalysis-NLIbias/model-{model_name_stub}:v0'
                print(full_name)
                download_file_to(api, dir_settings["model_ckpts_path"], full_name)



if __name__ == "__main__":
    main()