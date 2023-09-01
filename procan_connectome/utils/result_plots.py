import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from sklearn.metrics import confusion_matrix


def plot_all_figures(cfg, results_df, feature_importances_df, grid_search_df=None):
    plt.style.use(pathlib.Path(cfg.paths.root) / "plt_plot_style.mplstyle")
    plot_path = pathlib.Path(cfg.paths.plot) / f"{cfg.wandb.name}"
    if not plot_path.is_dir():
        plot_path.mkdir(parents=True, exist_ok=False)

    # Confusion matrix
    cm_plot_f_name = _generate_confusion_matrix(cfg, results_df, plot_path)
    cm_art = wandb.Image(cm_plot_f_name.__str__(), caption="Confusion Matrix")
    wandb.log({"Confusion Matrix": cm_art})

    # Feature importances
    if feature_importances_df is not None:
        # top 10
        fi_f_name = plot_feature_importance(
            feature_importances_df,
            "Top 10 Features",
            feature_col="Feature",
            importance_col="Importance",
            n_features=10,
            plot_path=plot_path,
        )
        fi_art = wandb.Image(fi_f_name.__str__(), caption="Top 10 Feature Importances")
        wandb.log({"Top 10 Feature Importances": fi_art})
        # top 50
        fi_f_name = plot_feature_importance(
            feature_importances_df,
            "Top 50 Features",
            feature_col="Feature",
            importance_col="Importance",
            n_features=50,
            plot_path=plot_path,
        )
        fi_art = wandb.Image(fi_f_name.__str__(), caption="Top 50 Feature Importances")
        wandb.log({"Top 50 Feature Importances": fi_art})
    return


def _generate_confusion_matrix(cfg, results_df, plot_path):
    labels = results_df["y_true"].unique()
    cm = confusion_matrix(results_df["y_true"], results_df["y_pred"], labels=labels)
    # label_map = {
    #     "Stage_0": "Stage 0",
    #     "Stage_1b": "Stage 1b",
    #     "Stage_1a": "Stage 1a",
    #     "HC": "Healthy Control",
    #     "Transition": "Transition",
    # }
    labels = np.array([cfg.labels_dict[label] for label in labels])
    f_name = confusion_matrix_plot(cm, labels, fig_title="CM.png", plot_path=plot_path)
    return f_name


def save_fig(plot_path, fig, f_name=None):
    if f_name is None:
        f_name = f"{fig.axes[0].title.get_text()}.png"
    f_path = pathlib.Path(plot_path) / f_name
    fig.savefig(f_path, bbox_inches="tight")
    print(f"Figure saved to {f_path}")
    return f_path


def plot_clusters_2D(
    df,
    x,
    y,
    hue,
    label_col_name=None,
    fig_title=None,
    plot_path=None,
    custom_label_map=None,
    custom_label_name=None,
):
    if custom_label_map is not None:
        df[custom_label_name] = df[label_col_name].map(custom_label_map)
    if plot_path is not None:
        plt.style.use(os.path.join(plot_path, "plt_plot_style.mplstyle"))
    DIMS = [15, 10]
    fig, ax = plt.subplots(figsize=DIMS)
    sns.scatterplot(x=x, y=y, hue=hue, data=df, ax=ax)
    title_string = fig_title
    ax.set_title(title_string)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    if plot_path is not None:
        save_fig(plot_path, fig)
    return fig


def confusion_matrix_plot(cm, label_names, fig_title=None, plot_path=None):
    DIMS = [15, 10]
    fig, ax = plt.subplots(figsize=DIMS)
    sns.heatmap(cm, annot=True, ax=ax, fmt=".2f", cmap="Greens")
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    if fig_title is None:
        ax.set_title("Confusion Matrix")
    else:
        ax.set_title(fig_title)
    ax.xaxis.set_ticklabels(label_names, rotation=90)
    ax.yaxis.set_ticklabels(label_names, rotation=360)
    if plot_path is not None:
        f_name = save_fig(plot_path, fig, fig_title)
        return f_name
    return fig


def plot_feature_importance(
    df,
    plot_title,
    feature_col="Feature",
    importance_col="Importance",
    n_features=None,
    plot_path=None,
):
    df = df.copy(deep=True)
    df = df.reset_index()
    df.sort_values(by=[importance_col], ascending=False, inplace=True)
    DIMS = [15, 10]
    fig = plt.figure(figsize=DIMS)
    if n_features is None:
        x = df[importance_col]
        y = df[feature_col]
    else:
        x = df[importance_col].iloc[:n_features]
        y = df[feature_col].iloc[:n_features]
    ax = sns.barplot(x=x, y=y, color="black")
    fig.add_subplot(ax)
    plt.title(plot_title)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    if n_features is None:
        plt.legend(y, bbox_to_anchor=(1.05, 1), loc="upper left", fancybox=False)
        ax.set_yticklabels([], size=10)
    if plot_path is not None:
        f_name = save_fig(
            plot_path, fig, f_name=f"feature_importances_top_{n_features}.png"
        )
        return f_name
    return fig
