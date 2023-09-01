import wandb
import omegaconf
import pickle
import pathlib


def init_wandb(cfg: omegaconf.DictConfig):
    run = wandb.init(
        config=omegaconf.OmegaConf.to_container(
            cfg=cfg, resolve=True, throw_on_missing=True
        ),
        settings=wandb.Settings(start_method=cfg.wandb.start_method),
        name=cfg.wandb.name,
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        dir=pathlib.Path(cfg.paths.data),
    )
    return run


def save_model(cfg, model):
    f_path = pathlib.Path(cfg.paths.logs) / f"{cfg.wandb.name}.pkl"
    with open(f_path, "wb") as handle:
        pickle.dump(model, handle)
    art = wandb.Artifact(name=cfg.wandb.name, type="model")
    art.add_file(f_path)
    print(f"artifact path: {f_path}")
    wandb.log_artifact(art)
    return
