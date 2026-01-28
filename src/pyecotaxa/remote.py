import concurrent.futures
import enum
import fnmatch
import ftplib
import functools
import glob
import hashlib
import logging
import os
import posixpath
import shutil
import time
import traceback
import urllib.parse
import uuid
import warnings
import zipfile
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import requests
import requests.adapters
import requests_toolbelt
import semantic_version
import urllib3.util.retry
import werkzeug
from atomicwrites import atomic_write as _atomic_write

from pyecotaxa._config import (
    JsonConfig,
    MultiConfig,
    check_config,
    default_config,
    find_file_recursive,
    load_env,
)
from pyecotaxa.meta import FileMeta
from tqdm.auto import tqdm

from .archive import Archive

logger = logging.getLogger(__name__)


class ImportMode(enum.Enum):
    # Yes = Update metadata only. Cla = Also update classifications. Else create.
    CREATE = "No"
    UPDATE_META = "Yes"
    UPDATE_ANNO = "Cla"


class Transport(enum.Enum):
    SHARE = "share"
    HTTP = "http"
    FTP = "ftp"


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

        with (
            atomic_write(dst) as fdst,
            tqdm(unit="B", unit_scale=True, unit_divisor=1024, total=total) as pm,
        ):
            buf = memoryview(bytearray(chunksize))
            while 1:
                nbytes = fsrc.readinto(buf)
                if not nbytes:
                    break
                fdst.write(buf)

                n_written += len(buf)

                yield n_written, total


def match_request(request: Mapping, pattern: Mapping):
    """
    Return True if request matches the pattern.

    I.e., request contains all the keys in pattern and the respective values match.
    """

    _missing = object()

    differences = {
        k: f"{v} vs. {request.get(k, '<missing>')}"
        for k, v in pattern.items()
        if request.get(k, _missing) != v
    }

    if differences:
        logger.debug(
            "Differences: " + (", ".join(f"{k}: {v}" for k, v in differences.items()))
        )

    return not differences


def _value_matches_query(value, query) -> bool:
    if isinstance(value, Mapping) and isinstance(query, Mapping):
        _none = object()
        return all(
            _value_matches_query(value.get(k, _none), qv) for k, qv in query.items()
        )

    if isinstance(value, str) and isinstance(query, str):
        return value == query

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


class Remote:
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
        retry: Union[int, Literal[False]] = 3,
    ):
        super().__init__()

        config = MultiConfig()

        config.update_from(default_config, "<default>")

        user_config_fn = os.path.expanduser("~/.pyecotaxa.json")
        config.update_from(JsonConfig(user_config_fn, verbose=verbose), user_config_fn)

        local_config_fn = find_file_recursive(".pyecotaxa.json")
        config.update_from(
            JsonConfig(local_config_fn, verbose=verbose), local_config_fn
        )

        config.update_from(load_env(verbose=verbose), "<environment>")

        # Update config from parameters
        # TODO: Use `set`
        if api_endpoint is not None:
            config["api_endpoint"] = api_endpoint

        if api_token is not None:
            config["api_token"] = api_token

        if exported_data_share is not None:
            config["exported_data_share"] = exported_data_share

        if import_data_share is not None:
            config["import_data_share"] = import_data_share

        self.config = check_config(config)

        # TODO: Use session everywhere
        self._session = requests.Session()
        if retry:
            retry_cfg = urllib3.util.retry.Retry(retry, backoff_factor=0.5)
            adapter = requests.adapters.HTTPAdapter(max_retries=retry_cfg)
            self._session.mount("http://", adapter)
            self._session.mount("https://", adapter)

        self._check_version()

    def _endpoint_url(self, path: str) -> str:
        path = path.lstrip("/\\")
        return urllib.parse.urljoin(self.config["api_endpoint"], path)

    def get(self, endpoint, headers: Optional[Mapping] = None, **kwargs):
        """Get data from the specified path (retrieve)."""
        # Build url from API endpoint and supplied path
        url = self._endpoint_url(endpoint)

        # Build headers
        if headers is None:
            headers = self.auth_headers
        else:
            headers = {**self.auth_headers, **headers}

        response = self._session.get(url, headers=headers, **kwargs)

        self._check_response(response)

        return response.json()

    def post(self, endpoint, headers: Optional[Mapping] = None, **kwargs):
        """Post data to the specified path (create)."""
        # Build url from API endpoint and supplied path
        url = self._endpoint_url(endpoint)

        # Build headers
        if headers is None:
            headers = self.auth_headers
        else:
            headers = {**self.auth_headers, **headers}

        response = self._session.post(url, headers=headers, **kwargs)

        self._check_response(response)

        return response.json()

    def put(self, endpoint, headers: Optional[Mapping] = None, **kwargs):
        """Put data to the specified path (update)."""
        # Build url from API endpoint and supplied path
        url = self._endpoint_url(endpoint)

        # Build headers
        if headers is None:
            headers = self.auth_headers
        else:
            headers = {**self.auth_headers, **headers}

        response = self._session.put(url, headers=headers, **kwargs)

        self._check_response(response)

        return response.json()

    def delete(self, endpoint, headers: Optional[Mapping] = None, **kwargs):
        """Delete data at the specified path (delete)."""
        # Build url from API endpoint and supplied path
        url = self._endpoint_url(endpoint)

        # Build headers
        if headers is None:
            headers = self.auth_headers
        else:
            headers = {**self.auth_headers, **headers}

        response = self._session.delete(url, headers=headers, **kwargs)

        self._check_response(response)

        return response.json()

    def _check_version(self):
        openapi_schema = self.get("openapi.json")

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

        api_token = self.post(
            "login",
            json={"password": password, "username": username},
        )

        self.config["api_token"] = api_token

        return api_token

    def ensure_login_interactive(self):
        """
        Ensure that the user is logged in.
        If not, prompt interactively for username and password.
        """

        if self.is_logged_in():
            return

        import getpass

        print(
            "Please enter your EcoTaxa credentials for {self.config['api_endpoint']}:"
        )
        username = input("Username: ")
        password = getpass.getpass("Password: ")

        self.login(username, password)

    @property
    def auth_headers(self):
        if not self.config["api_token"]:
            return {}

        return {"Authorization": f"Bearer {self.config['api_token']}"}

    def _get_job(self, job_id) -> Dict:
        """Retrieve details about a job."""
        return self.get(f"jobs/{job_id}/")

    def _get_job_file_http(self, project_id, job_id, *, target_directory: str) -> str:
        """Download an exported archive over HTTP and return the local file name."""

        response = self._session.get(
            self._endpoint_url(f"jobs/{job_id}/file"),
            params={},
            headers=self.auth_headers,
            stream=True,
        )

        self._check_response(response)

        content_length = int(response.headers.get("Content-Length", 0)) or None

        chunksize = 8 * 1024

        _, options = werkzeug.http.parse_options_header(
            response.headers["content-disposition"]
        )
        name = posixpath.basename(options["filename"])

        local_filename = os.path.join(target_directory, name)

        logger.info(f"Downloading {response.url} to {local_filename}...")

        try:
            with (
                tqdm(
                    unit="iB",
                    unit_scale=True,
                    unit_divisor=1024,
                    total=content_length,
                    desc=f"Downloading {name}...",
                ) as pm,
                atomic_write(local_filename) as fout,
            ):
                for chunk in response.iter_content(chunksize):
                    fout.write(chunk)
                    pm.update(len(chunk))
        except:
            # Cleaup destination file
            try:
                os.remove(local_filename)
            except FileNotFoundError:
                pass

            raise

        return local_filename

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

        (remote_fn,) = matches
        filename = os.path.basename(remote_fn)

        dest_fn = os.path.join(
            target_directory,
            # Destination filename should not have the task_<id>_ prefix to match get_job_file_remote
            removeprefix(filename, f"task_{job_id}_"),
        )

        logger.info(f"Copying {remote_fn} to {dest_fn}...")

        try:
            copyfile_progress(remote_fn, dest_fn)
            shutil.copymode(remote_fn, dest_fn)
        except:
            # Cleaup destination file
            try:
                os.remove(dest_fn)
            except FileNotFoundError:
                pass
            raise

        return dest_fn

    def validate_ftp_config(
        self, ftp_host=None, ftp_user=None, ftp_passwd=None, ftp_export_dir=None
    ):
        """Check if the FTP configuration is valid."""

        ftp_host = ftp_host or self.config["ftp_host"]
        ftp_user = ftp_user or self.config["ftp_user"]
        ftp_passwd = ftp_passwd or self.config["ftp_passwd"]
        ftp_export_dir = ftp_export_dir or self.config["ftp_export_dir"]

        with ftplib.FTP(
            ftp_host,
            ftp_user,
            ftp_passwd,
        ) as ftp:
            ftp.cwd(ftp_export_dir)

    def _get_job_file_ftp(self, project_id, job_id, *, target_directory: str) -> str:
        """Download an exported archive over FTP and return the local file name."""

        with ftplib.FTP(
            self.config["ftp_host"],
            self.config["ftp_user"],
            self.config["ftp_passwd"],
        ) as ftp:
            ftp.cwd(self.config["ftp_export_dir"])

            # Find file for the job
            pattern = f"task_{job_id}_*.zip"
            matches = [
                remote_fn
                for remote_fn in ftp.nlst()
                if fnmatch.fnmatchcase(remote_fn, pattern)
            ]

            if not matches:
                raise ValueError(
                    f"No FTP-accessible export for job {job_id}.\nPattern: {pattern}"
                )

            (remote_fn,) = matches

            size = ftp.size(remote_fn)

            filename = posixpath.basename(remote_fn)

            dest_fn = os.path.join(
                target_directory,
                # Destination filename should not have the task_<id>_ prefix to match get_job_file_remote
                removeprefix(filename, f"task_{job_id}_"),
            )

            with (
                tqdm(
                    unit="iB",
                    unit_scale=True,
                    unit_divisor=1024,
                    total=size,
                    desc=f"Downloading {filename}...",
                ) as pm,
                atomic_write(dest_fn) as fout,
            ):

                def writeblock(block: bytes):
                    fout.write(block)
                    pm.update(len(block))

                ftp.retrbinary(f"RETR {remote_fn}", callback=writeblock)

            return dest_fn

    def _get_job_file(
        self, project_id, job, *, target_directory: str, transport: Transport
    ) -> str:
        job_id = job["id"]

        out_to_ftp = job.get("params", {}).get("req", {}).get("out_to_ftp", False)

        if transport == Transport.SHARE:
            if self.config["import_data_share"] is None:
                raise ValueError(f"import_data_share is not available")

            if not out_to_ftp:
                raise ValueError(f"out_to_ftp is False")

            return self._get_job_file_local(
                project_id, job_id, target_directory=target_directory
            )

        if transport == Transport.HTTP:
            return self._get_job_file_http(
                project_id, job_id, target_directory=target_directory
            )

        if transport == Transport.FTP:
            return self._get_job_file_ftp(
                project_id, job_id, target_directory=target_directory
            )

        raise ValueError(f"Unknown transport: {transport!r}")

    def _start_project_export(self, project_id, *, request: Mapping, filters: Mapping):
        data = self.post(
            "object_set/export",
            json={
                "filters": filters,
                "request": request,
            },
        )

        job_id = data["job_id"]

        logger.info(f"Enqueued export job for project {project_id}.")

        # Get job data
        return self._get_job(job_id)

    def _get_jobs(self, type=None, params=None):
        jobs = self.get(
            "jobs/",
            params={"for_admin": False},
        )

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
        with tqdm(desc=task_descr, total=100, unit="%") as pbar:
            # 'P' for Pending (Waiting for an execution thread)
            # 'R' for Running (Being executed inside a thread)
            # 'A' for Asking (Needing user information before resuming)
            # 'E' for Error (Stopped with error)
            # 'F' for Finished (Done)."
            while job["state"] not in "FE":
                pct = job["progress_pct"] or 0
                pbar.update(pct - pbar.n)
                pbar.set_description(f"{task_descr} ({job['progress_msg']})")

                time.sleep(5)

                # Update job data
                job = self._get_job(job["id"])

        if job["state"] == "E":
            # Link to the job page
            # NB: We actually want urljoin here and start the path with "/" so it is relative to the domain name.
            job_url = urllib.parse.urljoin(
                self.config["api_endpoint"], f"/Job/Monitor/{job['id']}"
            )

            raise JobError(job["progress_msg"] + f"\nVisit <{job_url}> for more info.")

        return job

    def _export_and_download_archive(
        self,
        project_id,
        *,
        target_directory: str,
        with_images: bool,
        filters: Mapping,
        transport: Transport,
    ) -> str:
        """Export and download a project archive."""

        # Configure project export
        request = {
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
            "out_to_ftp": transport in (Transport.FTP, Transport.SHARE),
        }

        # Find running or finished export task for request
        matches = [
            job
            for job in self._get_jobs()
            if job.get("type") == "GenExport"
            and match_request(job.get("params", {}).get("req", {}), request)
        ]

        if not matches:
            job = self._start_project_export(
                project_id, request=request, filters=filters
            )
        else:
            job = matches[0]

        # Wait for job to be finished
        job = self._wait_job_progress(job, f"Exporting {project_id}...")

        logger.info(f"Export job for {project_id} done.")

        # Download job file
        archive_fn = self._get_job_file(
            project_id,
            job,
            target_directory=target_directory,
            transport=transport,
        )

        # Store metadata
        job_params_req = job.get("params", {}).get("req", {})
        FileMeta(archive_fn).update(job_params_req).save()

        return archive_fn

    def _check_archive(self, project_id, archive_fn):
        logger.info(f"Checking {archive_fn} ({project_id})...")

        try:
            with zipfile.ZipFile(archive_fn) as zf:
                zf.testzip()
        except Exception:
            raise

    def _cleanup_task_data(self, project_id):
        # Find finished export task for project_id

        logger.info(f"Cleaning up export task data for {project_id}...")

        jobs = self._get_jobs()

        matches = [
            job
            for job in jobs
            if job.get("params", {}).get("req", {}).get("project_id") == project_id
        ]

        for job in matches:
            job_id = job["id"]
            self.delete(f"jobs/{job_id}")

    def _pull_individual_project(
        self,
        project_id: int,
        *,
        target_directory: str,
        check_integrity: bool,
        cleanup_task_data: bool,
        with_images: bool,
        filters: Mapping,
        force_download: bool,
        transport: Transport,
    ) -> str:
        """Find and return the name of the local copy of the requested project."""

        try:
            pattern = os.path.join(target_directory, f"export_{project_id}_*.zip")
            matches = glob.glob(pattern)

            if matches and not force_download:
                logger.info(f"Export for {project_id} is available locally.")
                archive_fn = matches[0]
            else:
                archive_fn = self._export_and_download_archive(
                    project_id,
                    target_directory=target_directory,
                    with_images=with_images,
                    filters=filters,
                    transport=transport,
                )

            logger.info(f"Got {archive_fn}.")

            if check_integrity:
                self._check_archive(project_id, archive_fn)

            if cleanup_task_data:
                self._cleanup_task_data(project_id)
        except Exception as exc:
            raise exc

        logger.info(f"Project {project_id} OK.")

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
        filters: Optional[Mapping] = None,
        force_download: bool = False,
        transport: Transport = Transport.HTTP,
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

        if filters is None:
            filters = {}

        os.makedirs(target_directory, exist_ok=True)

        if n_parallel > 1:
            executor = concurrent.futures.ThreadPoolExecutor(n_parallel)
        else:
            executor = DummyExecutor()

        logger.info("Pulling projects...")

        futures = [
            executor.submit(
                show_trace(self._pull_individual_project),
                project_id,
                target_directory=target_directory,
                check_integrity=check_integrity,
                cleanup_task_data=cleanup_task_data,
                with_images=with_images,
                filters=filters,
                force_download=force_download,
                transport=transport,
            )
            for project_id in project_ids
        ]

        try:
            archive_fns = []
            for i, archive_fn_future in enumerate(
                concurrent.futures.as_completed(futures)
            ):
                archive_fn = archive_fn_future.result()

                archive_fns.append(archive_fn)

            return archive_fns
        except:
            executor.shutdown(False)
            raise

    def current_user(self):
        return self.get("users/me")

    def is_logged_in(self):
        try:
            self.current_user()
            return True
        except requests.exceptions.HTTPError as exc:
            response = exc.response
            if response is not None and response.status_code == 401:
                return False
            raise

    def _start_project_import(self, project_id, source_path, mode: ImportMode):
        logger.info("Starting project import...")
        data = self.post(
            f"file_import/{project_id}",
            json={
                "source_path": source_path,
                "taxo_mappings": {},
                "skip_loaded_files": False,
                "skip_existing_objects": True,  # Has to be True for an update (so that updateable objects are calculated)
                "update_mode": mode.value,
            },
        )

        job_id = data["job_id"]

        # Get job data
        return self._get_job(job_id)

    def _get_remote_fn_by_hashh(self, fhash):
        # Currently, the /my_files endpoint is broken
        # See https://github.com/ecotaxa/ecotaxa_back/issues/56
        # TODO: source_path = "/tmp/ecotaxa_user.{CREATOR_USER_ID}/{TAG}/{DEST_FILE_NAME}"
        return None

        response = self._session.get(self._endpoint_url(f"my_files/{fhash}"))

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

    def _upload_file_http(self, src_fn, force=False) -> str:
        name = os.path.basename(src_fn)

        with open(src_fn, "rb") as f:
            if force:
                # Generate random hash
                tag = uuid.uuid4().hex
                logger.info(f"Pushing to random tag: {tag}")
            else:
                # Compute hash
                tag = _file_hash(f)
                f.seek(0)

                remote_fn = self._get_remote_fn_by_hashh(tag)
                if remote_fn is not None:
                    logger.info(f"{src_fn} is already available remotely: {remote_fn}")
                    return remote_fn

            logger.info(f"Uploading {src_fn} via HTTP...")
            total = os.fstat(f.fileno()).st_size

            if total > 500 * 1024**2:
                logger.warning(
                    "File is larger than 500MiB, HTTP upload will most likely fail."
                )

            with tqdm(
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                total=total,
                desc=f"Uploading {name}...",
            ) as pm:
                me = requests_toolbelt.MultipartEncoder(
                    {"file": (name, f), "path": src_fn, "tag": tag}
                )
                mm = requests_toolbelt.MultipartEncoderMonitor(
                    me, lambda monitor: setattr(pm, "n", monitor.bytes_read)
                )
                response = self._session.post(
                    self._endpoint_url(f"my_files/"),
                    data=mm,
                    headers={**self.auth_headers, "Content-Type": mm.content_type},
                )

        self._check_response(response)

        filename = response.json()

        assert isinstance(filename, str)

        return filename

    def _upload_file_ftp(self, src_fn, force=False) -> str:
        logger.info(f"Uploading {src_fn} via FTP...")
        name: str = os.path.basename(src_fn)

        with open(src_fn, "rb") as f:
            if force:
                # Generate random hash
                tag = uuid.uuid4().hex
                logger.info(f"Pushing to random tag: {tag}")
            else:
                # Compute hash
                logger.info(f"Computing hash for {name}...")
                tag = _file_hash(f)
                f.seek(0)

                # remote_fn = self._get_remote_fn_by_hashh(tag)
                # if remote_fn is not None:
                #     logger.info(f"{src_fn} is already available remotely: {remote_fn}")
                #     return remote_fn

            total = os.fstat(f.fileno()).st_size

            with ftplib.FTP(
                self.config["ftp_host"],
                self.config["ftp_user"],
                self.config["ftp_passwd"],
            ) as ftp:
                ftp.cwd(self.config["ftp_datadir"])
                try:
                    ftp.cwd(tag)
                except ftplib.Error as exc:
                    if not exc.args[0].strip().startswith("550"):
                        raise
                    ftp.mkd(tag)
                    ftp.cwd(tag)

                try:
                    # Check for existence
                    ftp.size(name)
                except ftplib.Error:
                    should_upload = True
                else:
                    should_upload = False

                if should_upload:
                    with tqdm(
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        total=total,
                        desc=f"Uploading {name}...",
                    ) as pm:
                        ftp.storbinary(
                            f"STOR {name}",
                            f,
                            callback=lambda block: pm.update(len(block)),
                        )

        filename = posixpath.join(
            self.config["ftp_server_root"],
            self.config["ftp_datadir"].lstrip("/"),
            tag,
            name,
        )

        return filename

    def _upload_file_share(self, src_fn):
        dst_dir = os.path.join(self.config["import_data_share"], "pyecotaxa")
        os.makedirs(dst_dir, exist_ok=True)

        # Copy file into the import directory
        dst_fn = os.path.join(dst_dir, os.path.basename(src_fn))
        if not os.path.isfile(dst_fn):
            copyfile_progress(src_fn, dst_fn)
            shutil.copymode(src_fn, dst_fn)

        return posixpath.join("FTP/Ecotaxa_Data_to_import/pyecotaxa", dst_fn)

    def _push_individual_archive(
        self,
        src_fn: str,
        project_id: int,
        force=False,
        mode: ImportMode = ImportMode.CREATE,
        transport: Transport = Transport.HTTP,
        validate: bool = False,
    ):
        logger.info(f"Pushing {src_fn} to {project_id}...")

        if validate:
            Archive(src_fn).validate()
            logger.info(f"Archive {src_fn} seems to be valid.")

        if transport == Transport.SHARE:
            if self.config["import_data_share"] is None:
                raise ValueError(f"import_data_share is not available")
            remote_fn = self._upload_file_share(src_fn)
        elif transport == Transport.HTTP:
            remote_fn = self._upload_file_http(src_fn, force=force)
        elif transport == Transport.FTP:
            remote_fn = self._upload_file_ftp(src_fn, force=force)
        else:
            raise ValueError(f"Unknown transport: {transport!r}")

        logger.info(f"Remote filename is {remote_fn}.")

        # Find running or finished import task for project_id
        jobs = self._get_jobs(
            type="FileImport",
            params={
                "prj_id": project_id,
                "req": {"source_path": remote_fn, "update_mode": mode.value},
            },
        )

        # Only look for non-failed jobs
        jobs = [job for job in jobs if job["state"] != "E"]

        if not jobs:
            job = self._start_project_import(project_id, remote_fn, mode)
        else:
            job = jobs[0]
            logger.info(f"Job to import {remote_fn} already exists...")

        # Wait for job to be finished
        job = self._wait_job_progress(job, f"Importing to {project_id}...")

        # TODO: Cleanup of job (and files?)

        # TODO: Update metadata of local file: {"push:{project_id}": {"hash": ...}}

    def _validate_meta(self, root: str, meta: Mapping[str, Mapping[str, Any]]):
        def validate():
            for file_fn, file_meta in meta.items():
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
        force=False,
        mode: ImportMode = ImportMode.CREATE,
        transport: Transport = Transport.HTTP,
        validate: bool = False,
    ):
        """
        Push a local checkout to EcoTaxa.

        The respective projects need to already exist.

        Args:
            mode: create / update / update_with_classification
        """

        logger.info(f"Pushing {len(file_fn_project_id)} files...")

        if n_parallel:
            executor = concurrent.futures.ThreadPoolExecutor(n_parallel)
        else:
            executor = DummyExecutor()

        futures = [
            executor.submit(
                self._push_individual_archive,
                file_fn,
                project_id,
                force=force,
                mode=mode,
                transport=transport,
                validate=validate,
            )
            for file_fn, project_id in file_fn_project_id
        ]

        try:
            for fut in tqdm(
                concurrent.futures.as_completed(futures), total=len(futures)
            ):
                fut.result()

        finally:
            executor.shutdown()
