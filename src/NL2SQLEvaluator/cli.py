import typer
from typer_config import use_yaml_config
from typing_extensions import Annotated

import NL2SQLEvaluator.dataset_reader_nodes  # noqa: F401
import NL2SQLEvaluator.evaluator_nodes  # noqa: F401
import NL2SQLEvaluator.predictor_nodes  # noqa: F401
import NL2SQLEvaluator.saver_nodes  # noqa: F401
import NL2SQLEvaluator.sql_cache_nodes  # noqa: F401
from NL2SQLEvaluator.node_registry import get_available_functions

app = typer.Typer(rich_markup_mode="rich")


@use_yaml_config()
@app.command()
def main(
        output_dir: Annotated[
            str,
            typer.Option(
                help="Directory where evaluation outputs will be saved",
                rich_help_panel='Script Arguments',
            )
        ] = "./outputs",
        relative_db_base_path: Annotated[
            str,
            typer.Option(
                help="Relative path to the database files directory",
                rich_help_panel='Dataset arguments',
            )
        ] = "data/bird_dev/dev_databases",
        dataset_path: Annotated[
            str,
            typer.Option(
                help="HuggingFace dataset path or local dataset path",
                rich_help_panel='Dataset arguments',
            )
        ] = "simone-papicchio/bird",
        dataset_reader_node: Annotated[
            str,
            typer.Option(
                help=f"Dataset reader node to use for loading the dataset. Available readers:  {get_available_functions('dataset_reader_nodes')}",
                rich_help_panel="Pipeline Arguments",
            )
        ] = "bird_dev",
        predictor_node: Annotated[
            str | None,
            typer.Option(
                help=f"Predictor node to use for generating predictions. Available predictors: {get_available_functions('predictor_nodes')}",
                rich_help_panel="Pipeline Arguments",
            )
        ] = None,
        sql_cache_node: Annotated[
            str,
            typer.Option(
                help=f"SQL cache node to use for caching generated SQL queries. Available cache nodes: {get_available_functions('sql_cache_nodes')}",
                rich_help_panel="Pipeline Arguments",
            )
        ] = "SQLCacheNode",
        evaluator_node: Annotated[
            str,
            typer.Option(
                help=f"Evaluator node to use for evaluating predictions. Available evaluators: {get_available_functions('evaluator_nodes')}",
                rich_help_panel="Pipeline Arguments",
            )
        ] = "BirdEXEvaluator",
        saver_node: Annotated[
            str,
            typer.Option(
                help=f"Saver node to use for saving evaluation results. Available savers: {get_available_functions('saver_nodes')}",
                rich_help_panel="Pipeline Arguments",
            )
        ] = "JSONSaver",
):
    print(
        output_dir, relative_db_base_path, dataset_path, dataset_reader_node, predictor_node, sql_cache_node,
        evaluator_node, saver_node
    )


if __name__ == "__main__":
    app()
