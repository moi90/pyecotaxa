import datetime
import getpass
import os
import sys
import warnings
import zipfile
from functools import partial
from typing import Callable, NoReturn, Optional, Tuple

import click
import dateutil.parser
import pandas as pd
import requests

import pyecotaxa
import pyecotaxa.taxonomy
from pyecotaxa._config import JsonConfig, find_file_recursive
from pyecotaxa.archive import read_tsv, write_tsv
from pyecotaxa.meta import FileMeta
from pyecotaxa.remote import ImportMode, ProgressListener, Remote, Transport
import logging

warnings.simplefilter("error", pd.errors.DtypeWarning)


@click.group()
@click.version_option(version=pyecotaxa.__version__)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Be verbose",
)
def cli(verbose):  # pragma: no cover
    """
    Command line client for pyecotaxa.
    """

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)


@cli.command()
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
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Be verbose",
)
def login(chdir, destination, verbose):
    """
    Log in and store authentication token in the current working directory.
    """

    # Change to specified directory
    if chdir:
        os.chdir(chdir)

    config_fn = os.path.abspath(
        os.path.expanduser("~/.pyecotaxa.json")
        if destination == "user"
        else find_file_recursive(".pyecotaxa.json")
    )

    if verbose:
        print("Config:", config_fn)

    username = input("Username: ")
    password = getpass.getpass()

    remote = Remote()

    try:
        api_token = remote.login(username, password)
    except requests.exceptions.HTTPError as exc:
        response: Optional[requests.Response] = exc.response
        if response is not None and response.status_code == 403:
            print("Login failed!")
            return

        raise

    email = remote.current_user()["email"]

    print(f"Logged in successfully as {email}.")

    JsonConfig(config_fn).update(api_token=api_token).save()


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
@click.option(
    "--http",
    "transport",
    flag_value="http",
    help="Use HTTP for data transfer",
)
@click.option(
    "--ftp",
    "transport",
    flag_value="ftp",
    help="Use FTP for data transfer",
)
@click.option(
    "--share",
    "transport",
    flag_value="share",
    help="Use file share for data transfer",
)
def pull(project_ids, with_images, chdir, transport):
    """
    Pull projects from the EcoTaxa server.

    An export job is triggered on the server and the resulting file is downloaded to the current directory.
    """

    # Change to specified directory
    if chdir:
        os.chdir(chdir)

    progress_listener = ProgressListener()

    remote = Remote()

    if transport is None:
        transport = (
            Transport.SHARE
            if remote.config["exported_data_share"] is not None
            else Transport.HTTP
        )
    else:
        transport = Transport(transport)

    remote.register_observer(progress_listener.update)
    remote.pull(project_ids, with_images=with_images, transport=transport)


@cli.command()
@click.argument("file_fns", nargs=-1, type=str)
@click.option("--to", "project_id", type=int)
@click.option(
    "--chdir",
    "-C",
    metavar="PATH",
    help="Run as if started in PATH instead of the current working directory.",
)
@click.option(
    "-f",
    "--force",
    is_flag=True,
    help="Force upload even if present",
)
@click.option(
    "--http",
    "transport",
    flag_value="http",
    help="Use HTTP for data transfer",
)
@click.option(
    "--ftp",
    "transport",
    flag_value="ftp",
    help="Use FTP for data transfer",
)
@click.option(
    "--share",
    "transport",
    flag_value="share",
    help="Use file share for data transfer",
)
@click.option(
    "--update-meta",
    "mode",
    flag_value="UPDATE_META",
    help="Only update metadata (no creation of objects)",
)
@click.option(
    "--update-anno",
    "mode",
    flag_value="UPDATE_ANNO",
    help="Update metadata and annotations (no creation of objects)",
)
@click.option(
    "--create",
    "mode",
    flag_value="CREATE",
    help="Only create objects",
    default=True,
)
def push(file_fns, project_id, chdir, force, transport, mode):
    """
    Push archives to the EcoTaxa server.

    The files are uploaded and an import job is triggered on the server.
    """

    # Change to specified directory
    if chdir:
        os.chdir(chdir)

    # Prepare import job
    if project_id is None:
        file_fn_project_id = [
            (file_fn, FileMeta(file_fn + FileMeta.SUFFIX)["project_id"])
            for file_fn in file_fns
        ]
    else:
        file_fn_project_id = [(file_fn, project_id) for file_fn in file_fns]

    remote = Remote()

    if transport is None:
        transport = (
            Transport.SHARE
            if remote.config["import_data_share"] is not None
            else Transport.HTTP
        )
    else:
        transport = Transport(transport)

    mode = getattr(ImportMode, mode)

    logging.info("Logged in as %s", remote.current_user()["email"])
    logging.info(f"Transport: {transport}")

    remote.push(file_fn_project_id, force=force, transport=transport, mode=mode)


def _table_reader_writer(fn) -> Tuple[Callable, Callable]:
    # Read generic text file
    ext = os.path.splitext(fn)[1]

    options = {}

    if ext in (".csv", ".tsv"):
        # Text-based formats

        if ext == ".tsv":
            options["sep"] = "\t"

        return partial(
            pd.read_csv, fn, index_col=False, **options, header=0, dtype=str
        ), partial(pd.DataFrame.to_csv, index=False, **options)

    if ext in (".xls", ".xlsx", "xlsm", "xlsb", "odf", "ods", "odt"):
        # Excel-based formats

        return partial(
            pd.read_excel, fn, index_col=False, header=0, dtype=str
        ), partial(pd.DataFrame.to_excel, index=False)


def _abort(message, retval=-1) -> NoReturn:
    print(message, file=sys.stderr)
    sys.exit(retval)


@cli.command()
@click.argument("taxoexport_fn")
@click.argument("mapping_fn")
@click.argument("labels_fn")
@click.option("--label", "-l", default=None)
@click.option("--object-annotation-category", "-c", default=None)
@click.option("--output", "-o", "output_fn", default=None)
@click.option("--validate-mapping", is_flag=True)
def map_categories(
    taxoexport_fn,
    mapping_fn,
    labels_fn,
    label,
    object_annotation_category,
    output_fn,
    validate_mapping,
):
    """
    Map custom labels to existing EcoTaxa categories.

    TAXOEXPORT_FN Taxonomy file (taxoexport_XXX.tsv)
    MAPPING_FN Mapping file (.tsv or .csv; without type headers). Gets created if not existing
    LABELS_FN Labels file (.tsv or .csv; without type headers)
    """

    if output_fn is None:
        output_fn = labels_fn

    rename_columns = {}
    if label is not None:
        rename_columns[label] = "label"

    if object_annotation_category is not None:
        rename_columns[object_annotation_category] = "object_annotation_category"

    rename_columns_rev = {v: k for k, v in rename_columns.items()}

    # Load labels
    read_labels, write_labels = _table_reader_writer(labels_fn)
    labels: pd.DataFrame = read_labels(dtype=str, na_filter=False)
    labels = labels[labels.columns[~labels.columns.isin(rename_columns.values())]]
    labels = labels.rename(columns=rename_columns)

    if "label" not in labels.columns:
        _abort(
            f"Column 'label' missing in {labels_fn}: {list(labels.columns)}!",
        )

    # Load taxonomy
    print("Loading taxonomy...")
    taxonomy = pyecotaxa.taxonomy.load_taxonomy(taxoexport_fn)

    # Load mapping
    read_mapping, write_mapping = _table_reader_writer(mapping_fn)
    try:
        mapping = read_mapping(dtype=str, na_filter=False)
    except FileNotFoundError:
        mapping = None
    else:
        mapping = mapping[
            mapping.columns[~mapping.columns.isin(rename_columns.values())]
        ]
        mapping = mapping.rename(columns=rename_columns)

    # Perform mapping
    print("Mapping...")
    mapping, labels = pyecotaxa.taxonomy.map_categories(
        taxonomy, mapping, labels, validate_mapping=validate_mapping
    )

    # Reverse renamed columns
    mapping = mapping.rename(columns=rename_columns_rev)

    # Write back results
    write_mapping(mapping, mapping_fn)
    write_labels(labels, output_fn)


def _read_all_meta(data_fn, encoding="utf-8-sig"):
    ext = os.path.splitext(data_fn)[1]
    if ext == ".zip":
        with zipfile.ZipFile(data_fn) as zf:
            tsv_fns = [fn for fn in zf.namelist() if fn.rsplit(".", 1)[-1] == "tsv"]
            df = []
            for tsv_fn in tsv_fns:
                with zf.open(tsv_fn) as f:
                    df.append(read_tsv(f, encoding=encoding, low_memory=False))

            return pd.concat(df)

    if ext == ".tsv":
        return read_tsv(data_fn, encoding=encoding, low_memory=False)

    return pd.read_csv(data_fn, index_col=False, encoding=encoding, low_memory=False)


@cli.command()
@click.argument("src_fn")
@click.argument("base_fns", nargs=-1)
@click.option(
    "--overwrite-validated",
    is_flag=True,
    help="Overwrite already validated data (default: false).",
)
@click.option(
    "--include-empty",
    is_flag=True,
    help="Update annotation even if empty (default: false).",
)
@click.option(
    "--annotation-status",
    type=click.Choice(["predicted", "dubious", "validated"], case_sensitive=False),
    help="Set annotation status for updated objects.",
    default=None,
)
@click.option(
    "--annotation-person-name",
    help="Name of the annotator.",
    default=None,
)
@click.option(
    "--annotation-person-email",
    help="Email of the annotator.",
    default=None,
)
@click.option(
    "--annotation-datetime",
    help="Date and time of the annotation.",
    default=None,
)
@click.option(
    "--out",
    "-o",
    "out_dir",
    help="Output directory",
    default=None,
)
@click.option(
    "--column",
    "include_columns",
    help="Columns to include in the output",
    multiple=True,
)
@click.option(
    "--match-on",
    help="Match on certain columns",
    default=None,
)
def gen_annotation_update(
    src_fn,
    base_fns,
    overwrite_validated: bool,
    include_empty: bool,
    annotation_status: Optional[str],
    annotation_person_name: Optional[str],
    annotation_person_email: Optional[str],
    annotation_datetime: Optional[str],
    out_dir,
    include_columns,
    match_on,
):
    """
    Generate annotation update files based on one or multiple BASE_FNS.

    SRC_FN: Read annotations from this file.
    BASE_FNS: Generate annotation updates for these files.
    """

    if not len(base_fns):
        return

    include_columns = dict(
        v.split(":", 1) if ":" in v else (v, v) for v in include_columns
    )

    if match_on is None:
        match_on = "object_id"

    if ":" in match_on:
        src_on, dst_on = match_on.split(":", 1)
    else:
        src_on = dst_on = match_on

    try:
        src_data = _read_all_meta(src_fn)
    except FileNotFoundError:
        _abort(f"File '{src_fn}' not found. Aborting.")

    # Keep only annotation-related data from src and rename
    src_data = src_data[
        src_data.columns[
            src_data.columns.str.startswith("object_annotation_")
            | (src_data.columns == "object_id")
            | (src_data.columns.isin(include_columns.keys()))
        ]
    ].rename(columns=include_columns)

    # Update annotation metadata
    if annotation_status is not None:
        src_data["object_annotation_status"] = annotation_status
    if annotation_person_name is not None:
        src_data["object_annotation_person_name"] = annotation_person_name
    if annotation_person_email is not None:
        src_data["object_annotation_person_email"] = annotation_person_email
    if annotation_datetime is not None:
        if annotation_datetime == "now":
            object_annotation_datetime = datetime.datetime.now()
        else:
            object_annotation_datetime = dateutil.parser.isoparse(annotation_datetime)

        src_data["object_annotation_date"] = object_annotation_datetime.strftime(
            "%Y%m%d"
        )
        src_data["object_annotation_time"] = object_annotation_datetime.strftime(
            "%H%M%S"
        )

    if "object_annotation_category" not in src_data.columns:
        _abort(
            f"Column 'object_annotation_category' missing in {src_fn}.  Aborting.",
        )

    if not include_empty:
        src_data = src_data.loc[~src_data["object_annotation_category"].isnull()]

    if not src_data.size:
        print("Source data is empty.")
        return

    if src_on not in src_data.columns:
        _abort(f"No {src_on} column in {src_fn}. Aborting.")

    for base_fn in base_fns:
        print(f"Processing {base_fn}...")
        base_name = os.path.basename(os.path.splitext(base_fn)[0])
        out_fn = f"update-{base_name}.tsv"

        if out_dir is not None:
            out_fn = os.path.join(out_dir, out_fn)

        base_data = _read_all_meta(base_fn)

        if "object_id" not in base_data.columns:
            _abort(f"No object_id column in {base_fn}")

        if not overwrite_validated:
            # Restrict to unvalidated data
            if "object_annotation_status" not in base_data.columns:
                _abort(
                    f"Column 'object_annotation_status' missing in {base_fn}! Aborting.\n"
                    "Supply `--overwrite-validated` to update annotations regardless of previous status.",
                )

            mask = base_data["object_annotation_status"] != "validated"
            base_data = base_data[mask]

            print(
                f"Updating {mask.sum():,d} non-validated out of {mask.shape[0]:,d} ({mask.mean():.2%})"
            )

        base_drop = []
        if "{" in dst_on and "}" in dst_on:
            # Perform interpolation
            print(f"Calculating interpolated values for {dst_on!r}...")
            base_data[dst_on] = base_data.apply(
                lambda row: dst_on.format_map(row.to_dict()), axis=1
            )

            base_drop.append(dst_on)

        # Merge source into base
        keep_dst = sorted(set(["object_id", dst_on]))
        out_data = (
            base_data[keep_dst]
            .drop_duplicates()
            .merge(src_data, left_on=dst_on, right_on=src_on, suffixes=(None, "_src"))
        )

        # Cleanup out_data
        out_data = out_data[out_data.columns[~out_data.columns.isin(base_drop)]]

        print(
            f"Updated {len(out_data):,d} objects out of {len(base_data):,d} ({len(out_data)/len(base_data):.2%})"
        )

        if "object_annotation_status" not in out_data.columns:
            _abort(
                f"Column 'object_annotation_status' missing in result! Aborting.\n"
                "Supply `--annotation-status validated` to mark all updated objects as validated.",
            )

        write_tsv(out_data, out_fn)

    # object_annotation_status


@cli.command()
@click.argument("taxonomy_fn")
@click.option(
    "--project-id",
    type=int,
    help="Restrict taxonomy to taxa used in the provided project.",
)
@click.option(
    "--root-category",
    type=str,
    help="Restrict taxonomy to taxa below this root category.",
)
@click.option(
    "--chdir",
    "-C",
    metavar="PATH",
    help="Run as if started in PATH instead of the current working directory.",
)
def pull_taxonomy(
    taxonomy_fn: str,
    project_id: Optional[int],
    root_category: Optional[str],
    chdir: bool,
):
    """
    Pull the taxonomy tree from the EcoTaxa server.
    """

    # Change to specified directory
    if chdir:
        os.chdir(chdir)

    remote = Remote()

    if root_category is not None:
        if project_id is None:
            raise ValueError(f"--root-category requires --project-id")

        # https://ecotaxa.obs-vlfr.fr/api/docs#/Taxonomy%20Tree/search_taxa
        taxa_left = list(
            t["id"]
            for t in remote.get(
                "taxon_set/search",
                params={"query": root_category, "project_id": project_id},
            )
        )
    elif project_id is not None:
        # Get a list of all used taxon IDs used in this project
        # https://ecotaxa.obs-vlfr.fr/api/docs#/projects/project_set_get_stats
        taxa_left = list(
            remote.get("project_set/taxo_stats", params={"ids": project_id})[0][
                "used_taxa"
            ]
        )
    else:
        # Get a list of root taxa
        # https://ecotaxa.obs-vlfr.fr/api/docs#/Taxonomy%20Tree/query_root_taxa
        taxa_left = list(taxon["id"] for taxon in remote.get("taxa"))

    # Recursively get information about all used taxon IDs used in this project and their children
    # https://ecotaxa.obs-vlfr.fr/api/docs#/Taxonomy%20Tree/query_taxa_set

    taxa = []
    taxa_done = set()

    while taxa_left:
        # Get the first 500 items of taxa_left
        taxa_left_sub = taxa_left[:500]
        del taxa_left[:500]

        print(f"Querying {len(taxa_left_sub)} taxa...")
        response = remote.get(
            "taxon_set/query", params={"ids": ",".join(str(id) for id in taxa_left_sub)}
        )
        taxa_done.update(taxa_left_sub)

        for row in response:
            taxa_left.extend(cid for cid in row["children"] if cid not in taxa_done)
            taxa.append(
                {
                    "id": row["id"],
                    "name": row["name"],
                    "display_name": row["display_name"],
                    "lineage": ">".join(reversed(row["lineage"])),
                }
            )

    print(f"Pulled {len(taxa)} taxa.")
    taxa = pd.DataFrame(taxa)

    taxa.to_csv(taxonomy_fn, index=False)
