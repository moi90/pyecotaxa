import os
from typing import Any, Dict, Mapping, Union
from pathlib import Path
import json
import collections.abc


class JsonConfig(collections.abc.MutableMapping):
    def __init__(self, filename) -> None:
        self._data, self._path = (
            self._try_load_absolute(filename)
            if os.path.isabs(filename)
            else self._try_load_relative(filename)
        )

    @staticmethod
    def _try_load_absolute(filename):
        try:
            with open(filename, "r") as f:
                contents = f.read()

            return json.loads(contents), filename
        except FileNotFoundError:
            return {}, filename

    @staticmethod
    def _try_load_relative(filename):
        wdir = os.getcwd()

        # TODO: Refactor search into own class
        while True:
            try:
                path = os.path.join(wdir, filename)
                with open(path, "r") as f:
                    contents = f.read()

                return json.loads(contents), path
            except FileNotFoundError:
                pass

            # Do not cross fs boundaries
            if os.path.ismount(wdir):
                break

            wdir = os.path.dirname(wdir)

        return {}, os.path.join(os.getcwd(), filename)

    def update(self, *args, **kwargs):
        self._data.update(*args, **kwargs)
        return self

    def setdefaults(self, *args, **kwargs):
        data = dict(*args, **kwargs)
        data.update(self._data)
        self._data = data
        return self

    def update_from(self, path):
        other = FileMeta(path)
        return self.update(other)

    def save(self):
        with open(self._path, "w") as f:
            json.dump(self._data, f)

        return self

    # Mutable mapping interface
    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __delitem__(self, key):
        del self._data[key]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return self._data.__iter__()

    def __repr__(self) -> str:
        return f"<FileMeta {self._data}>"


class FileMeta(JsonConfig):
    SUFFIX = ".meta.json"

    def __init__(self, path: Union[Path, str]) -> None:
        path = str(path)

        if not path.endswith(self.SUFFIX):
            path = path + ".meta.json"

        super().__init__(path)
