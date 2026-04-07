import click
import mteval_dspy.dspy_utils
import json


def parse_optimizer_params(params_str: str) -> dict:
    try:
        return json.loads(params_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON for optimizer parameters: {e}")


def parse_optimizer_compile_params(params_str: str) -> dict:
    try:
        return json.loads(params_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON for optimizer compile parameters: {e}")


@click.group()
@click.option(
    "--model",
    "-m",
    type=str,
    envvar="MTEVAL_DSPY_MODEL",
    required=True,
    help="Model name",
)
@click.option(
    "--api-base",
    type=str,
    envvar="MTEVAL_DSPY_API_BASE",
    default=None,
    help="API base URL for the language model.",
)
@click.option(
    "--api-key",
    "-k",
    type=str,
    envvar=["MTEVAL_API_KEY", "OPENAI_API_KEY"],
    default="NIL",
    help="API key for authentication",
)
@click.option(
    "--max-tokens",
    type=int,
    envvar="MTEVAL_DSPY_MAX_TOKENS",
    default=2048,
    help="Maximum number of tokens to generate.",
)
@click.option(
    "--enable-disk-cache/--disable-disk-cache",
    default=False,
    show_default=True,
    help="Enable or disable disk caching of model responses.",
)
@click.option(
    "--max-concurrent",
    type=int,
    default=100,
    show_default=True,
    help="Maximum number of concurrent requests to the language model API.",
)
@click.option(
    "--sampling-params",
    type=str,
    default="{}",
    show_default=True,
    help='Additional sampling parameters in JSON format, e.g. {"temperature": 0.7, "top_p": 0.9}',
)
@click.option(
    "--enable-ssl-verify/--disable-ssl-verify",
    default=True,
    show_default=True,
    help="Enable or disable SSL certificate verification for HTTP clients.",
)
@click.option(
    "--http-timeout",
    type=float,
    default=6000.0,
    show_default=True,
)
@click.pass_context
def cli(
    ctx,
    model,
    api_base,
    api_key,
    max_tokens,
    enable_disk_cache,
    max_concurrent,
    sampling_params,
    enable_ssl_verify,
    http_timeout,
):
    ctx.ensure_object(dict)
    import dspy

    sampling_params = json.loads(sampling_params)
    config = mteval_dspy.dspy_utils.DSPyLMConfig(
        model=model,
        api_base=api_base,
        api_key=api_key,
        max_tokens=max_tokens,
        lm_extra={**sampling_params},
    )
    mteval_dspy.dspy_utils.setup_lm(config)
    dspy.configure_cache(
        enable_disk_cache=enable_disk_cache,
        enable_memory_cache=True,
    )

    import httpx
    import litellm

    # https://docs.litellm.ai/docs/providers/openai#set-ssl_verifyfalse
    litellm.client_session = httpx.Client(
        verify=enable_ssl_verify,
        timeout=http_timeout,
        limits=httpx.Limits(max_connections=max_concurrent),
    )
    litellm.aclient_session = httpx.AsyncClient(
        verify=enable_ssl_verify,
        timeout=http_timeout,
        limits=httpx.Limits(max_connections=max_concurrent),
    )
    ctx.obj["MAX_CONCURRENT"] = max_concurrent


@cli.command()
@click.pass_context
@click.option(
    "--training-data",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
)
@click.option(
    "--training-data-max-examples",
    type=int,
    default=None,
    show_default=True,
)
@click.option(
    "--validation-data",
    type=click.Path(exists=True, dir_okay=False),
    # Note: not required because of LabeledFewShot optimizer, required for all others
)
@click.option(
    "--validation-data-max-examples",
    type=int,
    default=None,
    show_default=True,
)
@click.option(
    "--optimizer",
    type=click.Choice(
        [
            "MIPROv2",
            "SIMBA",
        ]
    ),
    default="MIPROv2",
    show_default=True,
    help="Optimizer to use for training.",
)
@click.option(
    "--optimizer-params",
    type=str,
    default="{}",
    show_default=True,
    help="Additional optimizer parameters in JSON format.",
)
@click.option(
    "--optimizer-compile-params",
    type=str,
    default="{}",
    show_default=True,
    help="Additional optimizer compile parameters in JSON format.",
)
@click.option(
    "--objective",
    type=click.Choice(
        [
            "tRMSE",
            "PA",
        ]
    ),
    default="tRMSE",
    show_default=True,
    help="Objective used during optimization. tRMSE is RMSE linearly transformed into 0-1 range, higher is better. PA is pairwise accuracy.",
)
@click.option(
    "--pairwise-k-per-source",
    type=click.IntRange(min=1),
    default=8,
    show_default=True,
    help="For objective=PA, number of random target pairs to sample per source segment.",
)
@click.option(
    "--pairwise-epsilon",
    type=float,
    default=0.0,
    show_default=True,
    help="For objective=PA, treat gold score differences within epsilon as near-ties and count them as correct.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False),
    help="Output file to save the trained model.",
    required=True,
)
@click.option(
    "--architecture",
    type=click.Choice(
        [
            "DA",
            "MR7",
            "MR7RRWA",
            "MR7MEAN",
        ]
    ),
    default="DA",
    show_default=True,
    help="Model architecture to use.",
)
@click.option(
    "--initial-program",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Optional path to a previously optimized program to load before continuing training.",
)
def train_da(
    ctx,
    training_data,
    training_data_max_examples,
    validation_data,
    validation_data_max_examples,
    optimizer,
    objective,
    output,
    optimizer_params,
    optimizer_compile_params,
    architecture,
    pairwise_k_per_source,
    pairwise_epsilon,
    initial_program,
):
    import mteval_dspy.dspy_utils
    import mteval_dspy.train as train

    data_config = train.DataConfig(
        objective=objective,
        trainset_path=training_data,
        valset_path=validation_data,
        trainset_max_examples=training_data_max_examples,
        valset_max_examples=validation_data_max_examples,
        pairwise_k_per_source=pairwise_k_per_source,
    )
    import mteval_dspy.architectures

    qe_module = mteval_dspy.architectures.create_module(architecture=architecture)
    if initial_program is not None:
        qe_module.load(initial_program)
    optimizer_params_dict = parse_optimizer_params(optimizer_params)
    optimizer_compile_params_dict = parse_optimizer_compile_params(
        optimizer_compile_params
    )
    training_config = train.TrainingConfig(
        data_config=data_config,
        optimizer_params=optimizer_params_dict,
        optimizer_compile_params=optimizer_compile_params_dict,
        pairwise_epsilon=pairwise_epsilon,
    )
    match optimizer:
        case "MIPROv2":
            qe_module = train.train_mipro(qe_module, training_config)
        case "SIMBA":
            qe_module = train.train_simba(qe_module, training_config)
        case _:
            raise ValueError(f"Unknown optimizer: {optimizer}")

    if isinstance(qe_module, mteval_dspy.dspy_utils.PairwiseDA):
        qe_module = qe_module.module
    qe_module.save(output)


@cli.command()
@click.option(
    "--architecture",
    type=click.Choice(
        [
            "DA",
            "MR7",
        ]
    ),
    default="DA",
    show_default=True,
    help="Model architecture to use.",
)
@click.option(
    "--trained-model",
    "-m",
    type=click.Path(exists=True, dir_okay=False),
)
@click.argument(
    "input_file",
    type=click.Path(exists=True, dir_okay=False, allow_dash=True),
    default="-",
)
@click.option(
    "--tokenizer",
    type=str,
    default="google/gemma-3-27b-it",
    help="Tokenizer to use for optional truncation of src and tgt tokens in the input. Name or path to a Hugging Face tokenizer.",
)
@click.option(
    "--max-segment-tokens",
    type=int,
    default=None,
    help="If set, truncate src and tgt segments to this many tokens after tokenization.",
)
@click.pass_context
def predict_da(
    ctx,
    input_file,
    architecture,
    trained_model,
    tokenizer,
    max_segment_tokens,
):
    import asyncio
    import sys
    import mteval_dspy.architectures
    import mteval_dspy.predict

    qe_module = mteval_dspy.architectures.create_module(architecture=architecture)
    if trained_model is not None:
        qe_module.load(trained_model)

    # Redirect stdout to stderr to avoid mixing with JSON output as DSPy sometimes pollutes it
    real_stdout = sys.stdout
    sys.stdout = sys.stderr

    async def main():
        await mteval_dspy.predict.process_file(
            qe_module=qe_module,
            input_file=input_file,
            outfp=real_stdout,
            tokenizer=tokenizer,
            max_segment_tokens=max_segment_tokens,
            max_concurrent=ctx.obj["MAX_CONCURRENT"],
        )

    asyncio.run(main())
