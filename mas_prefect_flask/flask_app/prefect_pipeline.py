from prefect import flow, task, get_run_logger
from sma_trainer import run_training_streaming


@task
def initialize(mode, n_episodes):

    logger = get_run_logger()

    logger.info(
        f"Starting training | mode={mode} | episodes={n_episodes}"
    )

    return {
        "mode": mode,
        "n_episodes": n_episodes
    }


@task
def train_episode(event):

    logger = get_run_logger()

    logger.info(
        f"Episode {event['episode']} | "
        f"reward={event['total_reward']} | "
        f"grid={event['grid_kwh']}"
    )

    return event


@task
def finalize(event):

    logger = get_run_logger()

    logger.info(
        f"Training completed | "
        f"SolarQ={event['q_solar_mean']} | "
        f"WindQ={event['q_wind_mean']}"
    )

    return event


@flow(name="Microgrid Training")
def training_flow(
    mode="competitive",
    n_episodes=200
):

    initialize(mode, n_episodes)

    final_event = None

    for event in run_training_streaming(
        mode=mode,
        n_episodes=n_episodes,
        save_qtables=True
    ):

        if event["type"] == "train_episode":

            train_episode.submit(event)

        elif event["type"] == "train_done":

            final_event = finalize(event)

    return final_event