import click


@click.group()
@click.option(
    "--model",
    "-m",
    type=str,
    envvar="MTEVAL_DSPY_MODEL",
    required=True,
    help="FIXME",
)
@click.option(
    "--api-key",
    "-k",
    type=str,
    envvar=["MTEVAL_API_KEY", "OPENAI_API_KEY"],
    default="NIL",
    help="FIXME",
)
def cli(model, api_key):
    pass


@cli.command()
@click.option("--training-data")
@click.option("--validation-data")
@click.option(
    "--optimizer",
    type=click.Choice(
        [
            "MIPROv2",
            "SIMBA",
            "GEPA",
        ]
    ),
)
@click.option(
    "--objective",
    type=click.Choice(
        [
            "tRMSE",
            "PA",
        ]
    ),
    help="Objective used during optimization. tRMSE is RMSE linearly transformed into 0-1 range, higher is better. PA is pairwise accuracy.",
)
def train_da():
    pass


@cli.command()
def predict_da():
    pass
