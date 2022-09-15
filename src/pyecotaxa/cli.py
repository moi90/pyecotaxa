import json
import os
import sys
from typing import Dict, Mapping, Tuple
import click

from pyecotaxa import __version__
from pyecotaxa.meta import JsonConfig
from pyecotaxa.transfer import Transfer
import pyecotaxa.taxonomy
import pandas as pd


@click.group()
@click.version_option(version=__version__)
def cli():  # pragma: no cover
    """
    Command line client for pyecotaxa.
    """


@cli.command()
@click.option("--username", prompt=True)
@click.option("--password", prompt=True, hide_input=True)
@click.option(
    "--user",
    "destination",
    flag_value="user",
    default=True,
    help="Store the API token in the user directory ~/.pyecotaxa (Default)",
)
@click.option("--local", "destination", flag_value="local")
@click.option(
    "--chdir",
    "-C",
    metavar="PATH",
    help="Run as if started in PATH instead of the current working directory.",
)
def login(username, password, chdir, destination):
    """
    Log in and store authentication token in the current working directory.
    """

    # Change to specified directory
    if chdir:
        os.chdir(chdir)

    config_fn = (
        os.path.expanduser("~/.pyecotaxa") if destination == "user" else ".pyecotaxa"
    )

    transfer = Transfer()
    transfer.login(username, password)

    JsonConfig(config_fn).update(api_token=transfer.config["api_token"]).save()


@cli.command()
@click.argument("project_ids", nargs=-1, type=int)
@click.option(
    "--with-images/--without-images",
    default=False,
    help="Include images in the result.",
)
## TODO: --skip-existing
# @click.option(
#     "--skip-existing/--no-skip-existing",
#     help="Skip projects that are already locally available.",
# )
@click.option(
    "--chdir",
    "-C",
    metavar="PATH",
    help="Run as if started in PATH instead of the current working directory.",
)
def pull(project_ids, with_images, chdir):
    """
    Pull projects from the EcoTaxa server.

    An export job is triggered on the server and the resulting file is downloaded to the current directory.
    """

    # Change to specified directory
    if chdir:
        os.chdir(chdir)

    transfer = Transfer()
    transfer.pull(project_ids, with_images=with_images)


def _read_txt(fn) -> Tuple[pd.DataFrame, Dict]:
    # Read generic text file
    ext = os.path.splitext(fn)[1]

    options = {}

    if ext == ".tsv":
        options["sep"] = "\t"

    return pd.read_csv(fn, index_col=False, **options, header=0), options


@cli.command()
@click.argument("taxoexport_fn")
@click.argument("mapping_fn")
@click.argument("labels_fn")
def map_categories(taxoexport_fn, mapping_fn, labels_fn):
    """
    Map custom labels to existing EcoTaxa categories.

    TAXOEXPORT_FN Taxonomy file (taxoexport_XXX.tsv)
    MAPPING_FN Mapping file (.tsv or .csv). Gets created if not existing
    LABELS_FN Labels file (.tsv or .csv)
    """

    # Load labels
    labels, labels_options = _read_txt(labels_fn)

    if "label" not in labels.columns:
        print(
            f"Column 'label' missing in {labels_fn}: {list(labels.columns)}!", file=sys.stderr
        )
        sys.exit(-1)

    # Load taxonomy
    taxonomy = pyecotaxa.taxonomy.load_taxonomy(taxoexport_fn)

    # Load mapping
    try:
        mapping, mapping_options = _read_txt(mapping_fn)
    except FileNotFoundError:
        mapping = None
        mapping_options = {}

    mapping, labels = pyecotaxa.taxonomy.map_categories(taxonomy, mapping, labels)

    # Write back results
    mapping.to_csv(mapping_fn, index=False, **mapping_options)
    labels.to_csv(labels_fn, index=False, **labels_options)
