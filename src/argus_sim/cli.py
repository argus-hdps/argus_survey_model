"""Command-line interface for ArgusSim."""

import click

from . import __version__


@click.group()
def main() -> None:
    """ArgusSim command line (``asim``)."""


@click.command("defaults", short_help="Write ArgusSim config with default settings to CWD")
def defaults() -> None:
    """Write the default configuration to ``argussim.toml`` in the current working directory.

    Examples
    --------
    $ asim defaults

    """
    from . import config

    defaults_conf = config.Config(sim_version=__version__)
    out_path = config.write_config(defaults_conf)
    click.echo(f"Default config written to: {out_path}")


main.add_command(defaults)


@click.command(
    "survey",
    short_help="Run a Survey over a span and summarise every night",
    context_settings={"ignore_unknown_options": True, "help_option_names": []},
    add_help_option=False,
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def survey(args) -> None:
    """Run a Survey over a span (``--start`` with ``--end`` or ``--n-nights``); see ``asim survey --help``.

    Arguments are handled by :func:`argus_sim.survey_driver.build_parser`.
    """
    from .survey_driver import run

    run(list(args))


main.add_command(survey)
