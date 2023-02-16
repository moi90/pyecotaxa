from abc import abstractmethod
import abc
from typing import Optional


_status_manager = None


def _get_default_status_manager() -> "StatusManager":
    try:
        import tqdm

        return TqdmStatusManager()
    except:
        pass

    raise RuntimeError("No default Status manager")


def _set_status_manager(status_manager: "StatusManager") -> Optional["StatusManager"]:
    global _status_manager

    old = _status_manager
    _status_manager = status_manager
    return old


def _get_status_manager() -> "StatusManager":
    global _status_manager

    if _status_manager is None:
        _status_manager = _get_default_status_manager()

    return _status_manager


def progress_meter(
    id,
    desc: Optional[str] = None,
    leave=False,
    unit="it",
    unit_scale=False,
    unit_binary=False,
    initial=0,
    total=None,
    position=None,
):
    """Create a new progress meter widget."""
    status_manager = _get_status_manager()
    return status_manager.progress_meter(
        id,
        desc=desc,
        leave=leave,
        unit=unit,
        unit_scale=unit_scale,
        unit_binary=unit_binary,
        initial=initial,
        total=total,
        position=position,
    )


class StatusManager(abc.ABC):
    def __init__(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        pass

    @abstractmethod
    def progress_meter(self, id, *args, **kwargs) -> "ProgressMeter":
        ...


class ProgressMeter(abc.ABC):
    def __init__(
        self,
        desc=None,
        leave=False,
        unit=False,
        unit_scale=False,
        unit_binary=False,
        initial=0,
        total=0,
        position=None,
    ) -> None:
        pass

    @abstractmethod
    def set_description(self, desc):
        pass

    @abstractmethod
    def set(self, n):
        pass

    @abstractmethod
    def update(self, n=1):
        pass

    def reset(self, total=None):
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class TqdmStatusManager(StatusManager):
    def progress_meter(self, id, **kwargs) -> ProgressMeter:
        return TqdmProgressMeter(**kwargs)


class TqdmProgressMeter(ProgressMeter):
    def __init__(
        self,
        *,
        desc,
        leave,
        unit,
        unit_scale,
        unit_binary,
        initial,
        total,
        position,
    ) -> None:
        import tqdm

        unit_divisor = 1024 if unit_binary else 1000

        self.pm = tqdm.tqdm(
            desc=desc,
            leave=leave,
            unit=unit,
            unit_scale=unit_scale,
            unit_divisor=unit_divisor,
            initial=initial,
            total=total,
            position=position,
        )

    def set_description(self, desc):
        self.pm.set_description(desc)

    def update(self, n=1):
        self.pm.update(n)

    def set(self, n):
        self.pm.n = n
        self.pm.update(0)

    def close(self):
        self.pm.close()


class JSONProgressMeter(ProgressMeter):
    def dump(self):
        return {
            id: self.id,
            unit: self.unit,
            value: self.value,
            total: self.total,
            eta: self.eta,
            elapsed: self.elapsed,
            descr: self.descr,
        }


class JSONStatusManager(StatusManager):
    """
    {
        progress: {
            id: "",
            unit: "",
            current: 0,
            total: 0,
            eta: 0,
            elapsed: 0,
            descr: "",
        }
    }
    """

    def __init__(self) -> None:
        super().__init__()

        self.widgets = []

    def progress_meter(self, id, position=None, **kwargs) -> ProgressMeter:
        widget = JSONProgressMeter(**kwargs)

        # TODO: Insert at specified position (even when None, ...)
        self.widgets.insert(position, widget)

        return widget

    def dump(self):
        return [w.dump() for w in self.widgets]


class ReportStatusManager(StatusManager):
    ...


def set_manager(pm: StatusManager):
    global _status_manager
    _status_manager = pm


def get_manager():
    if _status_manager is not None:
        return _status_manager
    raise RuntimeError("")


# ### Library:
# def do_something():
#     with get_manager() as m:
#         with m.progress_meter() as pm:
#             for _ in range(1000):
#                 pm.update(1)
