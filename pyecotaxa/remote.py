import concurrent.futures
import enum
import getpass
import glob
import hashlib
import logging
import os
import posixpath
import shutil
import time
import traceback
import urllib.parse
import warnings
import zipfile
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import requests
import requests_toolbelt
import semantic_version
import tqdm
import werkzeug
from atomicwrites import atomic_write as _atomic_write

from pyecotaxa._config import (
    JsonConfig,
    check_config,
    default_config,
    find_file_recursive,
    load_env,
)
from pyecotaxa.status import progress_meter

logger = logging.Logger(__name__)


class DummyExecutor(concurrent.futures.Executor):
    def submit(self, fn, *args, **kwargs):
        f = concurrent.futures.Future()

        try:
            f.set_result(fn(*args, **kwargs))
        except Exception as exc:
            f.set_exception(exc)
            pass

        return f


def atomic_write(path, **kwargs):
    prefix = f".{os.path.basename(path)}-"
    suffix = ".part"
    return _atomic_write(path, mode="wb", prefix=prefix, suffix=suffix, **kwargs)


class JobError(Exception):
    """Raised if a server-side job fails."""

    pass


class ApiVersionWarning(Warning):
    pass


def show_trace(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            traceback.print_exc()
            raise

    return wrapper


def removeprefix(s: str, prefix: str) -> str:
    """Polyfill for str.removeprefix introduced in Python 3.9."""

    if s.startswith(prefix):
        return s[len(prefix) :]
    else:
        return s[:]


def removesuffix(s: str, suffix: str) -> str:
    """Polyfill for str.removesuffix introduced in Python 3.9."""

    if suffix and s.endswith(suffix):
        return s[: -len(suffix)]
    else:
        return s[:]


def copyfile_progress(src, dst, chunksize=1024**2):
    """Copy data from src to dst with progress"""

    with open(src, "rb") as fsrc:
        total = os.fstat(fsrc.fileno()).st_size

        with atomic_write(dst) as fdst, progress_meter(
            f"file:{dst}", unit="B", unit_scale=True, unit_binary=True, total=total
        ) as pm:
            buf = memoryview(bytearray(chunksize))
            while 1:
                nbytes = fsrc.readinto(buf)
                if not nbytes:
                    break
                fdst.write(buf[:nbytes])

                pm.update(nbytes)


class State(enum.Enum):
    FINISHED = 0
    RUNNING = 1
    FAILED = 2
    WAITING = 3


class ProgressListener:
    def __init__(self) -> None:
        self.progress_bars = {}

    def update(
        self,
        target,
        state: Optional[State] = None,
        message: Optional[str] = None,
        description: Optional[str] = None,
        progress: Optional[int] = None,
        total: Optional[int] = None,
        unit: Optional[str] = None,
    ):
        try:
            progress_bar = self.progress_bars[target]
        except KeyError:
            progress_bar = self.progress_bars[target] = tqdm.tqdm(
                position=0 if target is None else None, unit_scale=True
            )

        if progress_bar.disable:
            # Progress bar is already closed
            return

        if message is not None:
            if target is not None:
                message = f"{target}: {message}"
            progress_bar.write(message)

        if description is not None:
            if target is not None:
                description = f"{target}: {description}"
            progress_bar.set_description(description, refresh=False)

        if progress is not None:
            progress_bar.n = progress

        if total is not None:
            progress_bar.total = total

        if unit is not None:
            progress_bar.unit = unit
        else:
            progress_bar.unit = "it"

        if state == State.FINISHED:
            progress_bar.close()
        else:
            progress_bar.refresh()


class Obervable:
    def __init__(self) -> None:
        self.__observers = []

    def register_observer(self, fn):
        self.__observers.append(fn)
        return fn

    def _notify_observers(self, *args, **kwargs):
        for fn in self.__observers:
            fn(*args, **kwargs)


def _value_matches_query(value, query) -> bool:
    if isinstance(value, Mapping) and isinstance(query, Mapping):
        _none = object()
        return all(
            _value_matches_query(value.get(k, _none), qv) for k, qv in query.items()
        )

    if isinstance(value, Sequence) and isinstance(query, Sequence):
        if len(value) != len(query):
            return False
        return all(_value_matches_query(v, qv) for v, qv in zip(value, query))

    if value == query:
        return True

    return False


def _file_hash(f) -> str:
    fhash = hashlib.sha256()
    for chunk in iter(lambda: f.read(4096), b""):
        fhash.update(chunk)

    return fhash.hexdigest()


class Remote(Obervable):
    """
    Interact with a remote EcoTaxa server.

    Args:
        api_endpoint (str, optional): EcoTaxa API endpoint.
        api_token (str, optional): API token.
            If not given, it is read from the environment variable ECOTAXA_API_TOKEN.
            ECOTAXA_API_TOKEN can be given on the command line or defined in a .env file in the project root.
            If that does not succeed, the user is prompted for username and password.
        exported_data_share (str or bool, optional): Locally accessible location for the results of an export job.
            If given, the archive will be copied, otherwise, it will be transferred over HTTP.
            If not given, the default location (/remote/plankton_rw/ftp_plankton/Ecotaxa_Exported_data/) will be used if available.
            To disable the detection of the default location, pass False.
    """

    REQUIRED_OPENAPI_VERSION = "~=0.0.25"

    def __init__(
        self,
        *,
        api_endpoint: Optional[str] = None,
        api_token: Optional[str] = None,
        exported_data_share: Union[None, str, bool] = None,
        import_data_share: Union[None, str, bool] = None,
        verbose=False,
    ):
        super().__init__()

        user_config_fn = os.path.expanduser("~/.pyecotaxa.json")
        local_config_fn = find_file_recursive(".pyecotaxa.json")

        # Load config from files and environment
        config = {
            **default_config,
            **JsonConfig(user_config_fn, verbose=verbose),
            **JsonConfig(local_config_fn, verbose=verbose),
            **load_env(verbose=verbose),
        }

        if api_endpoint is not None:
            config["api_endpoint"] = api_endpoint

        if api_token is not None:
            config["api_token"] = api_token

        if exported_data_share is not None:
            config["exported_data_share"] = exported_data_share

        if import_data_share is not None:
            config["import_data_share"] = import_data_share

        self.config = check_config(config)

        self._check_version()

    def _check_version(self):
        response = requests.get(
            urllib.parse.urljoin(self.config["api_endpoint"], "openapi.json"),
        )

        self._check_response(response)

        openapi_schema = response.json()

        version = openapi_schema.get("info", {}).get("version", "0.0.0")

        logger.info(f"Server OpenAPI version is {version}")

        if semantic_version.Version(version) not in semantic_version.SimpleSpec(
            self.REQUIRED_OPENAPI_VERSION
        ):
            warnings.warn(
                f"Required OpenAPI version is {self.REQUIRED_OPENAPI_VERSION}, Server has {version}.",
                category=ApiVersionWarning,
                stacklevel=2,
            )

    def login(self, username: str, password: str):
        """Login and store api_token."""
        response = requests.post(
            urllib.parse.urljoin(self.config["api_endpoint"], "login"),
            json={"password": password, "username": username},
        )

        self._check_response(response)

        self.config["api_token"] = api_token = response.json()

        return api_token

    @property
    def auth_headers(self):
        if not self.config["api_token"]:
            raise ValueError("API token not set")

        return {"Authorization": f"Bearer {self.config['api_token']}"}

    def _get_job(self, job_id) -> Dict:
        """Retrieve details about a job."""
        response = requests.get(
            urllib.parse.urljoin(self.config["api_endpoint"], f"jobs/{job_id}/"),
            headers=self.auth_headers,
        )

        self._check_response(response)

        return response.json()

    def _get_job_file_remote(self, project_id, job_id, *, target_directory: str) -> str:
        """Download an exported archive and return the local file name."""

        response = requests.get(
            urllib.parse.urljoin(self.config["api_endpoint"], f"jobs/{job_id}/file"),
            params={},
            headers=self.auth_headers,
            stream=True,
        )

        self._check_response(response)

        content_length = int(response.headers.get("Content-Length", 0)) or None

        chunksize = 1024

        _, options = werkzeug.http.parse_options_header(
            response.headers["content-disposition"]
        )
        filename = posixpath.basename(options["filename"])

        dest = os.path.join(target_directory, filename)

        print(f"Downloading {response.url} to {dest}...")

        try:
            self._notify_observers(
                project_id, description=f"Downloading...", progress=0, total=1
            )
            with atomic_write(dest) as f:
                progress = 0
                for chunk in response.iter_content(chunksize):
                    f.write(chunk)
                    progress += len(chunk)
                    self._notify_observers(
                        project_id,
                        description=f"Downloading...",
                        progress=progress,
                        total=content_length,
                        unit="iB",
                    )
                self._notify_observers(
                    project_id,
                    description=f"Downloading...",
                    progress=progress,
                    total=progress,
                    unit="iB",
                )
        except:
            # Cleaup destination file
            try:
                os.remove(dest)
            except FileNotFoundError:
                pass

            raise

        return dest

    def _get_job_file_local(self, project_id, job_id, *, target_directory: str) -> str:
        """Download an exported archive and return the local file name."""

        pattern = os.path.join(
            self.config["exported_data_share"], f"task_{job_id}_*.zip"
        )
        matches = glob.glob(pattern)

        if not matches:
            raise ValueError(
                f"No locally accessible export for job {job_id}.\nPattern: {pattern}"
            )

        remote_fn = matches[0]
        filename = os.path.basename(remote_fn)

        # Local filename should not have the task_<id>_ prefix to match get_job_file_remote
        dest = os.path.join(target_directory, removeprefix(filename, f"task_{job_id}_"))

        print(f"Copying {remote_fn} to {dest}...")

        try:
            copyfile_progress(remote_fn, dest)
            shutil.copymode(remote_fn, dest)
        except:
            # Cleaup destination file
            try:
                os.remove(dest)
            except FileNotFoundError:
                pass
            raise

        return dest

    def _get_job_file(self, project_id, job, *, target_directory: str) -> str:
        job_id = job["id"]

        out_to_ftp = job.get("params", {}).get("req", {}).get("out_to_ftp", False)

        if self.config["exported_data_share"] and out_to_ftp:
            return self._get_job_file_local(
                project_id, job_id, target_directory=target_directory
            )
        return self._get_job_file_remote(
            project_id, job_id, target_directory=target_directory
        )

    def _start_project_export(self, project_id, *, with_images):
        response = requests.post(
            urllib.parse.urljoin(self.config["api_endpoint"], "object_set/export"),
            json={
                "filters": {},
                "request": {
                    "project_id": project_id,
                    "exp_type": "BAK",
                    "use_latin1": False,
                    "tsv_entities": "OPAS",
                    "split_by": "S",
                    "coma_as_separator": False,
                    "format_dates_times": False,
                    "with_images": with_images,
                    "with_internal_ids": False,
                    "only_first_image": False,
                    "sum_subtotal": "A",
                    "out_to_ftp": self.config["exported_data_share"] is not None,
                },
            },
            headers=self.auth_headers,
        )

        self._check_response(response)

        data = response.json()

        job_id = data["job_id"]

        self._notify_observers(
            project_id,
            description=f"Enqueued export job.",
            progress=0,
            total=100,
            state=State.WAITING,
        )

        # Get job data
        return self._get_job(job_id)

    def _get_jobs(self, type=None, params=None):
        response = requests.get(
            urllib.parse.urljoin(self.config["api_endpoint"], "jobs"),
            params={"for_admin": False},
            headers=self.auth_headers,
        )

        self._check_response(response)

        jobs = response.json()

        if type is not None:
            jobs = [job for job in jobs if job.get("type") == type]

        if params is not None:
            jobs = [
                job
                for job in jobs
                if _value_matches_query(job.get("params", {}), params)
            ]

        return jobs

    def _wait_job_progress(self, job, task_descr: str):
        """
        Args:
            job:
            task_descr: Task description for progress meter.
        """
        with progress_meter(
            "remote_export", desc=task_descr, total=100, unit="%"
        ) as pm:
            # 'P' for Pending (Waiting for an execution thread)
            # 'R' for Running (Being executed inside a thread)
            # 'A' for Asking (Needing user information before resuming)
            # 'E' for Error (Stopped with error)
            # 'F' for Finished (Done)."
            while job["state"] not in "FE":
                pm.set(
                    job["progress_pct"] or 0,
                )
                pm.set_description(f"{task_descr} ({job['progress_msg']})")

                time.sleep(5)

                # Update job data
                job = self._get_job(job["id"])

        if job["state"] == "E":
            raise JobError(job["progress_msg"])

        return job

    def _download_archive(
        self, project_id, *, target_directory: str, with_images: bool
    ) -> str:
        """Export and download project."""

        # Find running or finished export task for project_id
        matches = [
            job
            for job in self._get_jobs()
            if job.get("type") == "GenExport"
            and job.get("params", {}).get("req", {}).get("project_id") == project_id
        ]

        if not matches:
            job = self._start_project_export(project_id, with_images=with_images)
        else:
            print(f"Existing export job for {project_id}.")
            job = matches[0]

        # Wait for job to be finished
        job = self._wait_job_progress(job, f"Exporting {project_id}...")

        print(f"Export job for {project_id} done.")

        # Download job file
        archive_fn = self._get_job_file(
            project_id, job, target_directory=target_directory
        )

        # Store metadata
        job_params_req = job.get("params", {}).get("req", {})
        FileMeta(archive_fn).update(job_params_req).save()

        return archive_fn

    def _check_archive(self, project_id, archive_fn) -> str:
        self._notify_observers(
            project_id, description=f"Checking archive...", progress=0, total=1
        )

        try:
            with zipfile.ZipFile(archive_fn) as zf:
                zf.testzip()
        except Exception:
            self._notify_observers(
                project_id, description=f"Checking archive...", state=State.FAILED
            )
            raise

        self._notify_observers(
            project_id, description=f"Checking archive...", progress=1, total=1
        )

    def _cleanup_task_data(self, project_id):
        # Find finished export task for project_id

        self._notify_observers(
            project_id,
            description=f"Cleaning up...",
            progress=0,
            total=1,
            state=State.RUNNING,
        )

        jobs = self._get_jobs()

        matches = [
            job
            for job in jobs
            if job.get("params", {}).get("req", {}).get("project_id") == project_id
        ]

        for job in matches:
            job_id = job["id"]

            response = requests.delete(
                urllib.parse.urljoin(self.config["api_endpoint"], f"jobs/{job_id}"),
                headers=self.auth_headers,
            )

            self._check_response(response)

        self._notify_observers(
            project_id,
            description=f"Cleaning up...",
            progress=1,
            total=1,
            state=State.RUNNING,
        )

    def _get_archive_for_project(
        self,
        project_id: int,
        *,
        target_directory: str,
        check_integrity: bool,
        cleanup_task_data: bool,
        with_images: bool,
    ) -> str:
        """Find and return the name of the local copy of the requested project."""

        try:
            pattern = os.path.join(target_directory, f"export_{project_id}_*.zip")
            matches = glob.glob(pattern)

            if matches:
                print(f"Export for {project_id} is available locally.")
                archive_fn = matches[0]
            else:
                archive_fn = self._download_archive(
                    project_id,
                    target_directory=target_directory,
                    with_images=with_images,
                )

            print(f"Got {archive_fn}.")

            if check_integrity:
                self._check_archive(project_id, archive_fn)

            if cleanup_task_data:
                self._cleanup_task_data(project_id)
        except Exception as exc:
            self._notify_observers(
                project_id,
                description=f"FAILED ({exc})",
                progress=1,
                total=1,
                state=State.FINISHED,
            )
            raise exc

        self._notify_observers(
            project_id, description="OK", progress=1, total=1, state=State.FINISHED
        )

        return archive_fn

    def _check_response(self, response: requests.Response):
        try:
            response.raise_for_status()
        except:
            logger.error(
                "\n".join(
                    [
                        "Request failed!",
                        f"Request url: {response.request.method} {response.request.url}",
                        f"Request headers: {response.request.headers}",
                        f"Response headers: {response.headers}",
                        f"Response text: {response.text}",
                    ]
                )
            )
            raise

    def pull(
        self,
        project_ids: Union[List[int], int],
        *,
        target_directory=".",
        n_parallel=1,
        check_integrity=True,
        cleanup_task_data=True,
        with_images=True,
    ) -> List[str]:
        """
        Export a project archive and transfer to a local directory.

        Args:
            project_ids (int or list): Project IDs to pull.
            target_directory (str, optional): Directory for exported archives.
            n_parallel (int, optional): Number of projects to be processed in parallel (export jobs, file transfer, ...).
                More parallel tasks might speed up the pull but also put more load on the server.
            check_integrity (bool, optional): Ensure the integrity of the exported archive.
            cleanup_task_data (bool, optional): Clean up task data on the server after successful download.
            with_images (bool, optional): Include images in the exported archive.
        """

        if isinstance(project_ids, int):
            project_ids = [project_ids]

        os.makedirs(target_directory, exist_ok=True)

        if n_parallel > 1:
            executor = concurrent.futures.ThreadPoolExecutor(n_parallel)
        else:
            executor = DummyExecutor()

        self._notify_observers(
            None, description="Pulling projects...", total=len(project_ids), unit="proj"
        )

        futures = [
            executor.submit(
                show_trace(self._get_archive_for_project),
                project_id,
                target_directory=target_directory,
                check_integrity=check_integrity,
                cleanup_task_data=cleanup_task_data,
                with_images=with_images,
            )
            for project_id in project_ids
        ]

        try:
            archive_fns = []
            for i, archive_fn_future in enumerate(
                concurrent.futures.as_completed(futures)
            ):
                archive_fn = archive_fn_future.result()

                self._notify_observers(None, progress=i + 1, unit="proj")

                archive_fns.append(archive_fn)

            return archive_fns
        except:
            executor.shutdown(False)
            raise

    def current_user(self):
        response = requests.get(
            urllib.parse.urljoin(self.config["api_endpoint"], "users/me"),
            headers=self.auth_headers,
        )

        self._check_response(response)

        return response.json()

    def _start_project_import(self, project_id, source_path):
        print("Starting project import...")
        response = requests.post(
            urllib.parse.urljoin(
                self.config["api_endpoint"], f"file_import/{project_id}"
            ),
            json={
                "source_path": source_path,
                "taxo_mappings": {},
                "skip_loaded_files": False,
                "skip_existing_objects": True,  # Has to be True for an update.
                "update_mode": "Cla",  # Yes = Update metadata only. Cla = Also update classifications.
            },
            headers=self.auth_headers,
        )

        self._check_response(response)

        data = response.json()

        job_id = data["job_id"]

        # Get job data
        return self._get_job(job_id)

    def _get_remote_fn_by_hashh(self, fhash):
        # Currently, the /my_files endpoint is broken
        # See https://github.com/ecotaxa/ecotaxa_back/issues/56
        return None

        response = requests.get(
            urllib.parse.urljoin(self.config["api_endpoint"], f"my_files/{fhash}"),
            headers=self.auth_headers,
        )

        if response.status_code == 404:
            return None

        self._check_response(response)

        # {'path': 'deadbeef...', 'entries': [{'name': 'archive.zip', 'type': 'F', 'size': 0, 'mtime': '2023-02-16 19:06:50.901954'}]}
        data = response.json()

        if "entries" not in data or "path" not in data:
            return None

        try:
            (entry,) = data["entries"]
        except ValueError:
            return None

        return posixpath.join(data["path"], entry["name"])

    def _upload_file_http(self, src_fn) -> str:
        name = os.path.basename(src_fn)

        with open(src_fn, "rb") as f:
            fhash = _file_hash(f)
            f.seek(0)

            remote_fn = self._get_remote_fn_by_hashh(fhash)

            if remote_fn is not None:
                logger.info(f"{src_fn} is already available remotely: {remote_fn}")
                return remote_fn

            logger.info(f"Uploading {src_fn} via HTTP...")
            total = os.fstat(f.fileno()).st_size
            with progress_meter(
                f"file:{src_fn}",
                unit="B",
                unit_scale=True,
                unit_binary=True,
                total=total,
            ) as pm:
                me = requests_toolbelt.MultipartEncoder(
                    {"file": (name, f), "path": src_fn, "tag": fhash}
                )
                mm = requests_toolbelt.MultipartEncoderMonitor(
                    me, lambda monitor: pm.set(monitor.bytes_read)
                )
                response = requests.post(
                    urllib.parse.urljoin(self.config["api_endpoint"], f"my_files/"),
                    data=mm,
                    headers={**self.auth_headers, "Content-Type": mm.content_type},
                )

        self._check_response(response)

        return response.text

    def _upload_file_share(self, src_fn):
        dst_dir = os.path.join(self.config["import_data_share"], "pyecotaxa")
        os.makedirs(dst_dir, exist_ok=True)

        # Copy file into the import directory
        dst_fn = os.path.join(dst_dir, os.path.basename(src_fn))
        if not os.path.isfile(dst_fn):
            copyfile_progress(src_fn, dst_fn)
            shutil.copymode(src_fn, dst_fn)

        return posixpath.join("FTP/Ecotaxa_Data_to_import/pyecotaxa", dst_fn)

    def _push_file(self, src_fn: str, project_id: int):
        print(f"Pushing {src_fn} to {project_id}...")

        if self.config["import_data_share"] is not None:
            remote_fn = self._upload_file_share(src_fn)
        else:
            remote_fn = self._upload_file_http(src_fn)

        print(f"Remote filename is {remote_fn}.")

        # Find running or finished import task for project_id
        jobs = self._get_jobs(
            type="FileImport",
            params={"req": {"project_id": project_id, "source_path": remote_fn}},
        )

        if not jobs:
            job = self._start_project_import(project_id, remote_fn)
        else:
            job = jobs[0]

        # Wait for job to be finished
        job = self._wait_job_progress(job, f"Importing to {project_id}...")

        # TODO: Cleanup

    def _validate_meta(self, root: str, meta: Mapping[str, Mapping[str, Any]]):
        def validate():
            for file_fn, file_meta in meta.items():
                print(file_fn, meta)
                if not os.path.isfile(os.path.join(root, file_fn)):
                    print(f"WARNING: {file_fn} is missing.")
                    continue

                if "project_id" not in file_meta:
                    print(f"WARNING: No project_id set for {file_fn}.")
                    continue

                yield (file_fn, file_meta)

        return dict(validate())

    def push(
        self,
        file_fn_project_id: Sequence[Tuple[str, int]],
        n_parallel=1,
    ):
        """
        Push a local checkout to EcoTaxa.

        The respective projects need to already exist.

        Args:
            mode: create / update / update_with_classification
        """

        print(f"Pushing {len(file_fn_project_id)} files...")

        if n_parallel:
            executor = concurrent.futures.ThreadPoolExecutor(n_parallel)
        else:
            executor = DummyExecutor()

        [
            self._push_file(
                file_fn,
                project_id,
            )
            for file_fn, project_id in file_fn_project_id
        ]

        # futures = [
        #     executor.submit(
        #         self._push_file,
        #         file_fn,
        #         project_id,
        #     )
        #     for file_fn, project_id in file_fn_project_id
        # ]

        # try:
        #     with progress_meter(
        #         "total", unit="B", unit_scale=True, unit_binary=True, total=len(futures)
        #     ) as pm:
        #         for fut in concurrent.futures.as_completed(futures):
        #             fut.result()
        #             pm.update()

        # finally:
        #     executor.shutdown()
