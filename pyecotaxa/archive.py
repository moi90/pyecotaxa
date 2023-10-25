"""Read and write EcoTaxa archives and individual EcoTaxa TSV files."""

import fnmatch
import io
import os
import pathlib
import posixpath
import shutil
import tarfile
import warnings
import zipfile
from typing import IO, Callable, List, Mapping, Optional, Union

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

__all__ = ["read_tsv", "write_tsv"]


def _fix_types(dataframe: pd.DataFrame, enforce_types):
    header = dataframe.columns.get_level_values(0)
    types = dataframe.columns.get_level_values(1)

    dataframe.columns = header

    float_cols = []
    text_cols = []
    for c, t in zip(header, types):
        if t == "[f]":
            float_cols.append(c)
        elif t == "[t]":
            text_cols.append(c)
        else:
            # If the first row contains other values than [f] or [t],
            # it is not a type header but a normal line of values and has to be inserted into the dataframe.
            # This is the case for "General export".

            # Clean up empty fields
            types = [None if t.startswith("Unnamed") else t for t in types]

            # Parse and prepend the current "types" to the dataframe
            row0_str = pd.DataFrame([types], columns=header).to_csv(index=False)
            row0 = pd.read_csv(
                io.StringIO(row0_str),
                index_col=False,
                # Use dt.name, because pd.Int64Dtype does not work here directly
                dtype={c: dt.name for c, dt in dataframe.dtypes.items()},
            )

            if enforce_types:
                raise ValueError("enforce_types=True, but no type header was found.")

            return pd.concat((row0, dataframe), ignore_index=True)

    if enforce_types:
        # Enforce [f] types
        dataframe[float_cols] = dataframe[float_cols].astype(float)
        dataframe[text_cols] = dataframe[text_cols].fillna("").astype(str)
    else:
        # Replace NaN with empty string in string columns
        for c in dataframe.columns:
            if pd.api.types.is_string_dtype(dataframe.dtypes[c]):
                dataframe[c] = dataframe[c].fillna("")

    return dataframe


def _apply_usecols(
    df: pd.DataFrame, usecols: Union[Callable, List[str]]
) -> pd.DataFrame:
    if callable(usecols):
        columns = [c for c in df.columns.get_level_values(0) if usecols(c)]
    else:
        columns = [c for c in df.columns.get_level_values(0) if c in usecols]

    return df[columns]


DEFAULT_DTYPES = {
    "img_file_name": str,
    "img_rank": int,
    "object_id": str,
    "object_link": str,
    "object_lat": float,
    "object_lon": float,
    "object_date": str,
    "object_time": str,
    "object_annotation_date": str,
    "object_annotation_time": str,
    "object_annotation_category": str,
    "object_annotation_category_id": "Int64",
    "object_annotation_person_name": str,
    "object_annotation_person_email": str,
    "object_annotation_status": str,
    "process_id": str,
    "acq_id": str,
    "sample_id": str,
}


def read_tsv(
    filepath_or_buffer,
    encoding: str = "utf-8-sig",
    enforce_types: Optional[bool] = None,
    usecols: Union[None, Callable, List[str]] = None,
    dtype=None,
    **kwargs,
) -> pd.DataFrame:
    """
    Read an individual EcoTaxa TSV file.

    Args:
        filepath_or_buffer (str, path object or file-like object): ...
        encoding (str, optional): Encoding of the TSV file.
            With the default "utf-8-sig", both UTF8 and signed UTF8 can be read.
        enforce_types: Enforce the column dtypes provided in the header.
            Usually, it is desirable to use the default dtypes and allow pandas to infer the column dtypes.
        usecols: List of strings or callable.
        **kwargs: Additional kwargs are passed to :func:`pandas:pandas.read_csv`.

    Returns:
        A Pandas :class:`~pandas:pandas.DataFrame`.
    """

    if enforce_types:
        dtype = str
    else:
        if dtype is None:
            dtype = DEFAULT_DTYPES
        elif isinstance(dtype, Mapping):
            dtype = {**DEFAULT_DTYPES, **dtype}
        else:
            # One dtype for all columns
            pass

    if usecols is not None:
        chunksize = kwargs.pop("chunksize", 10000)

        # Read a few rows a time
        dataframe: pd.DataFrame = pd.concat(
            [
                _apply_usecols(chunk, usecols)
                for chunk in pd.read_csv(
                    filepath_or_buffer,
                    sep="\t",
                    encoding=encoding,
                    header=[0, 1],
                    chunksize=chunksize,
                    dtype=dtype,
                    **kwargs,
                )
            ]
        )  # type: ignore
    else:
        if kwargs.pop("chunksize", None) is not None:
            warnings.warn("Parameter chunksize is ignored.")

        dataframe: pd.DataFrame = pd.read_csv(
            filepath_or_buffer,
            sep="\t",
            encoding=encoding,
            header=[0, 1],
            dtype=dtype,
            **kwargs,
        )  # type: ignore

    return _fix_types(dataframe, enforce_types)


def _dtype_to_ecotaxa(dtype):
    if pd.api.types.is_numeric_dtype(dtype):
        return "[f]"

    return "[t]"


def write_tsv(
    dataframe: pd.DataFrame,
    path_or_buf=None,
    encoding="utf-8",
    type_header=True,
    formatters: Optional[Mapping] = None,
    **kwargs,
):
    """
    Write an individual EcoTaxa TSV file.

    Args:
        dataframe: A pandas DataFrame.
        path_or_buf (str, path object or file-like object): ...
        encoding: Encoding of the TSV file.
            With the default "utf-8", both UTF8 and signed UTF8 readers can read the file.
        enforce_types: Enforce the column dtypes provided in the header.
            Usually, it is desirable to allow pandas to infer the column dtypes.
        type_header (bool, default true): Include the type header ([t]/[f]).
            This is required for a successful import into EcoTaxa.

    Return:
        None or str

            If path_or_buf is None, returns the resulting csv format as a string. Otherwise returns None.
    """

    if formatters is None:
        formatters = {}

    dataframe = dataframe.copy(deep=False)

    # Calculate type header before formatting values
    ecotaxa_types = [_dtype_to_ecotaxa(dt) for dt in dataframe.dtypes]

    # Apply formatting
    for col in dataframe.columns:
        fmt = formatters.get(col)

        if fmt is None:
            continue

        dataframe[col] = dataframe[col].apply(fmt)

    if type_header:
        # Inject types into header
        dataframe.columns = pd.MultiIndex.from_tuples(
            list(zip(dataframe.columns, ecotaxa_types))
        )

    return dataframe.to_csv(
        path_or_buf, sep="\t", encoding=encoding, index=False, **kwargs
    )


class MemberNotFoundError(Exception):
    pass


class UnknownArchiveError(Exception):
    pass


class _TSVIterator:
    def __init__(self, archive: "Archive", tsv_fns, kwargs) -> None:
        self.archive = archive
        self.tsv_fns = tsv_fns
        self.kwargs = kwargs

    def __iter__(self):
        for tsv_fn in self.tsv_fns:
            with self.archive.open(tsv_fn) as f:
                yield tsv_fn, read_tsv(f, **self.kwargs)

    def __len__(self):
        return len(self.tsv_fns)


class ArchivePath:
    def __init__(self, archive: "Archive", filename) -> None:
        self.archive = archive
        self.filename = filename

    def open(self, mode="r", compress_hint=True) -> IO:
        return self.archive.open(self.filename, mode, compress_hint)

    def __truediv__(self, filename):
        return ArchivePath(self.archive, posixpath.join(self.filename, filename))


class Archive:
    """
    A generic archive reader and writer for ZIP and TAR archives.
    """

    extensions: List[str] = []

    def __new__(cls, archive_fn: Union[str, pathlib.Path], mode: str = "r"):
        archive_fn = str(archive_fn)

        if mode[0] == "r":
            for subclass in cls.__subclasses__():
                if subclass.is_readable(archive_fn):
                    return super(Archive, subclass).__new__(subclass)

            raise UnknownArchiveError(f"No handler found to read {archive_fn}")

        if mode[0] in ("a", "w", "x"):
            for subclass in cls.__subclasses__():
                if any(archive_fn.endswith(ext) for ext in subclass.extensions):
                    return super(Archive, subclass).__new__(subclass)

            raise UnknownArchiveError(f"No handler found to write {archive_fn}")

    @staticmethod
    def is_readable(archive_fn) -> bool:
        raise NotImplementedError()  # pragma: no cover

    def __init__(self, archive_fn: Union[str, pathlib.Path], mode: str = "r"):
        raise NotImplementedError()  # pragma: no cover

    def open(self, member_fn, mode="r", compress_hint=True) -> IO:
        """
        Raises:
            MemberNotFoundError if mode=="r" and the member was not found.
        """

        raise NotImplementedError()  # pragma: no cover

    def find(self, pattern) -> List[str]:
        return fnmatch.filter(self.members(), pattern)

    def members(self) -> List[str]:
        raise NotImplementedError()  # pragma: no cover

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_, **__):
        self.close()

    def iter_tsv(self, **kwargs):
        """
        Yield tuples of all (tsv_fn, data) in the archive.

        Example:
            with Archive("export.zip") as archive:
                for tsv_fn, tsv in archive.iter_tsv():
                    # tsv_fn is the location of the tsv file (img_file_name is relative to that)
                    # tsv is a pandas DataFrame containing all data
                    ...

        """
        return _TSVIterator(self, self.find("*.tsv"), kwargs)

    def __truediv__(self, key):
        return ArchivePath(self, key)

    def add_images(
        self, df: pd.DataFrame, src: Union[str, "Archive", pathlib.Path], progress=False
    ):
        """Add images referenced in df from src."""

        if isinstance(src, str):
            src = pathlib.Path(src)

        for img_file_name in tqdm(df["img_file_name"], disable=not progress):
            with (src / img_file_name).open() as f_src, self.open(
                img_file_name, "w"
            ) as f_dst:
                shutil.copyfileobj(f_src, f_dst)


class _TarIO(io.BytesIO):
    def __init__(self, archive: "TarArchive", member_fn) -> None:
        super().__init__()
        self.archive = archive
        self.member_fn = member_fn

    def close(self) -> None:
        self.seek(0)
        self.archive.write_member(self.member_fn, self)
        super().close()


class TarArchive(Archive):
    extensions = [
        ".tar",
        ".tar.bz2",
        ".tb2",
        ".tbz",
        ".tbz2",
        ".tz2",
        ".tar.gz",
        ".taz",
        ".tgz",
        ".tar.lzma",
        ".tlz",
    ]

    @staticmethod
    def is_readable(archive_fn):
        return os.path.isfile(archive_fn) and tarfile.is_tarfile(archive_fn)

    def __init__(self, archive_fn: Union[str, pathlib.Path], mode: str = "r"):
        self._tar = tarfile.open(archive_fn, mode)
        self.__members = None

    def close(self):
        self._tar.close()

    def open(self, member_fn, mode="r", compress_hint=True) -> IO:
        # tar does not compress files individually
        del compress_hint

        if mode == "r":
            try:
                fp = self._tar.extractfile(self._resolve_member(member_fn))
            except KeyError as exc:
                raise MemberNotFoundError(
                    f"{member_fn} not in {self._tar.name}"
                ) from exc

            if fp is None:
                raise IOError("There's no data associated with this member")

            return fp

        if mode == "w":
            return _TarIO(self, member_fn)

        raise ValueError(f"Unrecognized mode: {mode}")

    @property
    def _members(self):
        if self.__members is not None:
            return self.__members
        self.__members = {
            tar_info.name: tar_info for tar_info in self._tar.getmembers()
        }
        return self.__members

    def _resolve_member(self, member):
        return self._members[member]

    def write_member(
        self, member_fn: str, fileobj_or_bytes: Union[IO, bytes], compress_hint=True
    ):
        # tar does not compress files individually
        del compress_hint

        if isinstance(fileobj_or_bytes, bytes):
            fileobj_or_bytes = io.BytesIO(fileobj_or_bytes)

        if isinstance(fileobj_or_bytes, io.BytesIO):
            tar_info = tarfile.TarInfo(member_fn)
            tar_info.size = len(fileobj_or_bytes.getbuffer())
        else:
            tar_info = self._tar.gettarinfo(arcname=member_fn, fileobj=fileobj_or_bytes)

        self._tar.addfile(tar_info, fileobj=fileobj_or_bytes)
        self._members[tar_info.name] = tar_info

    def members(self):
        return self._tar.getnames()


class ZipArchive(Archive):
    extensions = [".zip"]

    @staticmethod
    def is_readable(archive_fn):
        return os.path.isfile(archive_fn) and zipfile.is_zipfile(archive_fn)

    def __init__(self, archive_fn: Union[str, pathlib.Path], mode: str = "r"):
        self._zip = zipfile.ZipFile(archive_fn, mode)

    def members(self):
        return self._zip.namelist()

    def open(self, member_fn: str, mode="r", compress_hint=True) -> IO:
        if mode == "w" and not compress_hint:
            # Disable compression
            member = zipfile.ZipInfo(member_fn)
            member.compress_type = zipfile.ZIP_STORED
        else:
            # Let ZipFile.open select compression and compression level
            member = member_fn

        try:
            return self._zip.open(member, mode)
        except KeyError as exc:
            raise MemberNotFoundError(
                f"{member_fn} not in {self._zip.filename}"
            ) from exc

    def write_member(
        self, member_fn: str, fileobj_or_bytes: Union[IO, bytes], compress_hint=True
    ):
        compress_type = zipfile.ZIP_DEFLATED if compress_hint else zipfile.ZIP_STORED
        # TODO: Optimize for on-disk files and BytesIO (.getvalue())
        if isinstance(fileobj_or_bytes, bytes):
            return self._zip.writestr(
                member_fn, fileobj_or_bytes, compress_type=compress_type
            )

        self._zip.writestr(
            member_fn, fileobj_or_bytes.read(), compress_type=compress_type
        )

    def close(self):
        self._zip.close()


class DirectoryArchive(Archive):
    extensions = [""]

    @staticmethod
    def is_readable(archive_fn):
        return os.path.isdir(archive_fn)

    def __init__(self, archive_fn: Union[str, pathlib.Path], mode: str = "r"):
        self.archive_fn = archive_fn
        self.mode = mode

    def members(self):
        def findall():
            for root, dirs, files in os.walk(self.archive_fn):
                relroot = os.path.relpath(root, self.archive_fn)
                for fn in files:
                    yield os.path.join(relroot, fn)

        return list(findall())

    def open(self, member_fn: str, mode="r", compress_hint=True) -> IO:
        return open(os.path.join(self.archive_fn, member_fn), mode=mode)

    def write_member(
        self, member_fn: str, fileobj_or_bytes: Union[IO, bytes], compress_hint=True
    ):
        with self.open(member_fn, "w") as f:
            if isinstance(fileobj_or_bytes, bytes):
                f.write(fileobj_or_bytes)
            else:
                shutil.copyfileobj(fileobj_or_bytes, f)

    def close(self):
        pass
