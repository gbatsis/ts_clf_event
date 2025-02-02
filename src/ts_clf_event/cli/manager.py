import typer

from ts_clf_event.cli.commands import trainer_cli, evaluator_cli, inference_cli

app = typer.Typer()
app.add_typer(trainer_cli, name="")
app.add_typer(evaluator_cli, name="")
app.add_typer(inference_cli, name="inference")

if __name__ == "__main__":
    app()