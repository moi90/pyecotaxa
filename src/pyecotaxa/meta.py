import os
from typing import Any, Dict, Mapping, Union
from pathlib import Path
import pyecotaxa._config

class FileMeta(pyecotaxa._config.JsonConfig):
    SUFFIX = ".meta.json"

    def __init__(self, path: Union[Path, str]) -> None:
        path = str(path)

        if not path.endswith(self.SUFFIX):
            path = path + self.SUFFIX

        super().__init__(path)
