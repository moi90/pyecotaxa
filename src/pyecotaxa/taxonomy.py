from typing import Dict, List, Mapping, Optional, Tuple
import pandas as pd
from tqdm import tqdm
import anytree
import os
import anytree.importer.jsonimporter
import anytree.exporter.jsonexporter


def _gen_cache_fn(filename, ext, prefix="."):
    head, tail = os.path.split(filename)
    name = os.path.splitext(tail)[0]
    return os.path.join(head, prefix + tail + ext)


def load_taxonomy(taxoexport_fn, cache=True) -> anytree.Node:
    """
    Load taxoexport file as a tree.
    """
    # Try to load taxonomy from cache file
    cache_fn = _gen_cache_fn(taxoexport_fn, ".json")

    if (
        cache
        and os.path.isfile(cache_fn)
        and os.stat(cache_fn).st_mtime > os.stat(taxoexport_fn).st_mtime
    ):
        importer = anytree.importer.jsonimporter.JsonImporter()
        with open(cache_fn) as f:
            try:
                return importer.read(f)
            except Exception as exc:
                print(exc)
                pass

    # Otherwise, convert

    taxoexport = pd.read_csv(taxoexport_fn, sep="\t", index_col=None)

    print("Converting taxoexport...")
    progress = tqdm(total=len(taxoexport), unit_scale=True)

    def create_children(parent, parent_id=None, depth=0):
        children = (
            taxoexport[taxoexport["parent_id"] == parent_id]
            if parent_id is not None
            else taxoexport[taxoexport["parent_id"].isna()]
        )
        for c in children.itertuples():
            n = anytree.Node(
                str(c.name), parent=parent, unique_name=str(c.display_name)
            )
            progress.update()

            create_children(n, parent_id=c.id, depth=depth + 1)

    traxotree = anytree.Node("#", unique_name="")
    create_children(traxotree, None)

    if cache:
        exporter = anytree.exporter.jsonexporter.JsonExporter()
        with open(cache_fn, "w") as f:
            exporter.write(traxotree, f)

    return traxotree


class Matcher:
    def __init__(
        self,
        tree,
        mapping: Optional[Dict] = None,
        separator: str = "/",
        case_insensitive=False,
    ):
        self.tree = tree

        if mapping is None:
            mapping = {}

        self.mapping = mapping
        self.separator = separator
        self.case_insensitive = case_insensitive

    def _find_prefix(self, parts: List[str]):
        # Try whole path
        try:
            return parts, [], self.mapping[self.separator.join(parts)]
        except KeyError:
            pass

        # Try any prefix
        for i in range(1, len(parts)):
            try:
                key = parts[:-i]
                return key, parts[-i:], self.mapping[self.separator.join(key)]
            except KeyError:
                continue

        # Return empty match
        return [], parts, None

    def match(self, path: str) -> Tuple[str, str, List]:
        parts = path.split(self.separator)

        parts, remainder, unique_name = self._find_prefix(parts)
        nodes = [
            anytree.search.find(self.tree, lambda node: node.unique_name == unique_name)
            if unique_name is not None
            else self.tree
        ]

        while remainder:
            name = remainder[0]

            matches = []
            for r in nodes:
                matches.extend(
                    anytree.search.findall(r, lambda node: self._cmp(node.name, name))
                )

            if not matches:
                break

            nodes = matches
            parts.append(remainder.pop(0))

        return self.separator.join(parts), self.separator.join(remainder), nodes

    def _cmp(self, a: str, b: str):
        if self.case_insensitive:
            return a.lower() == b.lower()

        return a == b


def _mapping_df_to_dict(mapping):
    if mapping is None:
        return {}

    mapping = mapping[~pd.isna(mapping["object_annotation_category"])]
    return {row.label: row.object_annotation_category for row in mapping.itertuples()}


def map_categories(
    taxonomy: anytree.Node,
    mapping_df: Optional[pd.DataFrame],
    labels_df: pd.DataFrame,
    case_insensitive=True,
):

    if mapping_df is None:
        mapping_df = pd.DataFrame(columns=["label", "object_annotation_category"])

    mapping_dict = _mapping_df_to_dict(mapping_df)

    matcher = Matcher(taxonomy, mapping_dict, case_insensitive=case_insensitive)

    mapping_new = []
    for label in labels_df["label"].dropna().unique():
        # Skip already processed
        if label in mapping_dict:
            continue

        prefix, remainder, nodes = matcher.match(label)
        notes = []

        if prefix == "":
            msg = f"No match"
            print(f"{label}:", msg)
            notes.append(msg)
        elif remainder:
            msg = f"Unmatched suffix: {remainder}"
            print(f"{label}:", msg)
            notes.append(msg)

        if len(nodes) > 1:
            msg = f"Multiple matches: " + (", ".join((n.unique_name for n in nodes)))
            print(f"{label}:", msg)
            notes.append(msg)
            target_label = ""
        else:
            msg = ""
            target_label = nodes[0].unique_name

        mapping_new.append(
            {
                "label": label,
                "object_annotation_category": target_label,
                "note": "; ".join(notes),
            }
        )

    mapping_new = pd.DataFrame(mapping_new)

    # Merge old and new mapping
    mapping_df = (
        pd.concat((mapping_df, mapping_new))
        .drop_duplicates("label", keep="last")
        .sort_values("label")
    )

    labels_df = labels_df.drop(
        columns=["object_annotation_category"], errors="ignore"
    ).merge(mapping_df[["label", "object_annotation_category"]], on="label")

    return mapping_df, labels_df
