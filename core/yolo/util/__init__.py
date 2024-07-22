# Ultralytics YOLO 🚀, AGPL-3.0 license

import contextlib
import os
import platform
import re
import urllib
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Union

import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm as tqdm_original

# PyTorch Multi-GPU DDP Constants
RANK = int(os.getenv("RANK", -1))

# Other Constants
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLO
DEFAULT_CFG_PATH = ROOT / "cfg/default.yaml"
VERBOSE = str(os.getenv("YOLO_VERBOSE", True)).lower() == "true"  # global verbose mode
TQDM_BAR_FORMAT = "{l_bar}{bar:10}{r_bar}" if VERBOSE else None  # tqdm bar format
MACOS, LINUX, WINDOWS = (platform.system() == x for x in ["Darwin", "Linux", "Windows"])  # environment booleans


class TQDM(tqdm_original):
    """
    Custom Ultralytics tqdm class with different default arguments.

    Args:
        *args (list): Positional arguments passed to original tqdm.
        **kwargs (any): Keyword arguments, with custom defaults applied.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize custom Ultralytics tqdm class with different default arguments.

        Note these can still be overridden when calling TQDM.
        """
        kwargs["disable"] = not VERBOSE or kwargs.get("disable", False)  # logical 'and' with default value if passed
        kwargs.setdefault("bar_format", TQDM_BAR_FORMAT)  # override default value if passed
        super().__init__(*args, **kwargs)


class SimpleClass:
    """Ultralytics SimpleClass is a base class providing helpful string representation, error reporting, and attribute
    access methods for easier debugging and usage.
    """

    def __str__(self):
        """Return a human-readable string representation of the object."""
        attr = []
        for a in dir(self):
            v = getattr(self, a)
            if not callable(v) and not a.startswith("_"):
                if isinstance(v, SimpleClass):
                    # Display only the module and class name for subclasses
                    s = f"{a}: {v.__module__}.{v.__class__.__name__} object"
                else:
                    s = f"{a}: {repr(v)}"
                attr.append(s)
        return f"{self.__module__}.{self.__class__.__name__} object with attributes:\n\n" + "\n".join(attr)

    def __repr__(self):
        """Return a machine-readable string representation of the object."""
        return self.__str__()

    def __getattr__(self, attr):
        """Custom attribute access error message with helpful information."""
        name = self.__class__.__name__
        raise AttributeError(f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}")


class IterableSimpleNamespace(SimpleNamespace):
    """Ultralytics IterableSimpleNamespace is an extension class of SimpleNamespace that adds iterable functionality and
    enables usage with dict() and for loops.
    """

    def __iter__(self):
        """Return an iterator of key-value pairs from the namespace's attributes."""
        return iter(vars(self).items())

    def __str__(self):
        """Return a human-readable string representation of the object."""
        return "\n".join(f"{k}={v}" for k, v in vars(self).items())

    def __getattr__(self, attr):
        """Custom attribute access error message with helpful information."""
        name = self.__class__.__name__
        raise AttributeError(
            f"""
            '{name}' object has no attribute '{attr}'. This may be caused by a modified or out of date ultralytics
            'default.yaml' file.\nPlease update your code with 'pip install -U ultralytics' and if necessary replace
            {DEFAULT_CFG_PATH} with the latest version from
            https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml
            """
        )

    def get(self, key, default=None):
        """Return the value of the specified key if it exists; otherwise, return the default value."""
        return getattr(self, key, default)


def plt_settings(rcparams=None, backend="Agg"):
    """
    Decorator to temporarily set rc parameters and the backend for a plotting function.

    Example:
        decorator: @plt_settings({"font.size": 12})
        context manager: with plt_settings({"font.size": 12}):

    Args:
        rcparams (dict): Dictionary of rc parameters to set.
        backend (str, optional): Name of the backend to use. Defaults to 'Agg'.

    Returns:
        (Callable): Decorated function with temporarily set rc parameters and backend. This decorator can be
            applied to any function that needs to have specific matplotlib rc parameters and backend for its execution.
    """

    if rcparams is None:
        rcparams = {"font.size": 11}

    def decorator(func):
        """Decorator to apply temporary rc parameters and backend to a function."""

        def wrapper(*args, **kwargs):
            """Sets rc parameters and backend, calls the original function, and restores the settings."""
            original_backend = plt.get_backend()
            if backend.lower() != original_backend.lower():
                plt.close("all")  # auto-close()ing of figures upon backend switching is deprecated since 3.8
                plt.switch_backend(backend)

            with plt.rc_context(rcparams):
                result = func(*args, **kwargs)

            if backend != original_backend:
                plt.close("all")
                plt.switch_backend(original_backend)
            return result

        return wrapper

    return decorator


def emojis(string=""):
    """Return platform-dependent emoji-safe version of string."""
    return string.encode().decode("ascii", "ignore") if WINDOWS else string


def yaml_save(file="data.yaml", data=None, header=""):
    """
    Save YAML data to a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        data (dict): Data to save in YAML format.
        header (str, optional): YAML header to add.

    Returns:
        (None): Data is saved to the specified file.
    """
    if data is None:
        data = {}
    file = Path(file)
    if not file.parent.exists():
        # Create parent directories if they don't exist
        file.parent.mkdir(parents=True, exist_ok=True)

    # Convert Path objects to strings
    valid_types = int, float, str, bool, list, tuple, dict, type(None)
    for k, v in data.items():
        if not isinstance(v, valid_types):
            data[k] = str(v)

    # Dump data to file in YAML format
    with open(file, "w", errors="ignore", encoding="utf-8") as f:
        if header:
            f.write(header)
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def yaml_load(file="data.yaml", append_filename=False):
    """
    Load YAML data from a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        append_filename (bool): Add the YAML filename to the YAML dictionary. Default is False.

    Returns:
        (dict): YAML data and file name.
    """
    assert Path(file).suffix in {".yaml", ".yml"}, f"Attempting to load non-YAML file {file} with yaml_load()"
    with open(file, errors="ignore", encoding="utf-8") as f:
        s = f.read()  # string

        # Remove special characters
        if not s.isprintable():
            s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+", "", s)

        # Add YAML filename to dict and return
        data = yaml.safe_load(s) or {}  # always return a dict (yaml.safe_load() may return None for empty files)
        if append_filename:
            data["yaml_file"] = str(file)
        return data


# Default configuration
DEFAULT_CFG_DICT = yaml_load(DEFAULT_CFG_PATH)


def is_online() -> bool:
    """
    Check internet connectivity by attempting to connect to a known online host.

    Returns:
        (bool): True if connection is successful, False otherwise.
    """
    with contextlib.suppress(Exception):
        assert str(os.getenv("YOLO_OFFLINE", "")).lower() != "true"  # check if ENV var YOLO_OFFLINE="True"
        import socket

        for dns in ("1.1.1.1", "8.8.8.8"):  # check Cloudflare and Google DNS
            socket.create_connection(address=(dns, 80), timeout=2.0).close()
            return True
    return False


def is_dir_writeable(dir_path: Union[str, Path]) -> bool:
    """
    Check if a directory is writeable.

    Args:
        dir_path (str | Path): The path to the directory.

    Returns:
        (bool): True if the directory is writeable, False otherwise.
    """
    return os.access(str(dir_path), os.W_OK)


def get_git_dir():
    """
    Determines whether the current file is part of a git repository and if so, returns the repository root directory. If
    the current file is not part of a git repository, returns None.

    Returns:
        (Path | None): Git root directory if found or None if not found.
    """
    for d in Path(__file__).parents:
        if (d / ".git").is_dir():
            return d


def get_user_config_dir(sub_dir="yolo"):
    """
    Return the appropriate config directory based on the environment operating system.

    Args:
        sub_dir (str): The name of the subdirectory to create.

    Returns:
        (Path): The path to the user config directory.
    """
    if WINDOWS:
        path = Path.home() / "AppData" / "Roaming" / sub_dir
    elif MACOS:  # macOS
        path = Path.home() / "Library" / "Application Support" / sub_dir
    elif LINUX:
        path = Path.home() / ".config" / sub_dir
    else:
        raise ValueError(f"Unsupported operating system: {platform.system()}")

    # GCP and AWS lambda fix, only /tmp is writeable
    if not is_dir_writeable(path.parent):
        print(
            f"WARNING ⚠️ user config directory '{path}' is not writeable, defaulting to '/tmp' or CWD."
            "Alternatively you can define a YOLO_CONFIG_DIR environment variable for this path."
        )
        path = Path("/tmp") / sub_dir if is_dir_writeable("/tmp") else Path().cwd() / sub_dir

    # Create the subdirectory if it does not exist
    path.mkdir(parents=True, exist_ok=True)

    return path


GIT_DIR = get_git_dir()
USER_CONFIG_DIR = Path(os.getenv("YOLO_CONFIG_DIR") or get_user_config_dir())  # core/yolo settings dir
SETTINGS_YAML = USER_CONFIG_DIR / "settings.yaml"


class TryExcept(contextlib.ContextDecorator):
    """
    Ultralytics TryExcept class. Use as @TryExcept() decorator or 'with TryExcept():' context manager.

    Examples:
        As a decorator:
        >>> @TryExcept(msg="Error occurred in func", verbose=True)
        >>> def func():
        >>>    # Function logic here
        >>>     pass

        As a context manager:
        >>> with TryExcept(msg="Error occurred in block", verbose=True):
        >>>     # Code block here
        >>>     pass
    """

    def __init__(self, msg="", verbose=True):
        """Initialize TryExcept class with optional message and verbosity settings."""
        self.msg = msg
        self.verbose = verbose

    def __enter__(self):
        """Executes when entering TryExcept context, initializes instance."""
        pass

    def __exit__(self, exc_type, value, traceback):
        """Defines behavior when exiting a 'with' block, prints error message if necessary."""
        if self.verbose and value:
            print(emojis(f"{self.msg}{': ' if self.msg else ''}{value}"))
        return True


class SettingsManager(dict):
    """
    Manages Ultralytics settings stored in a YAML file.

    Args:
        file (str | Path): Path to the Ultralytics settings YAML file. Default is USER_CONFIG_DIR / 'settings.yaml'.
        version (str): Settings version. In case of local version mismatch, new default settings will be saved.
    """

    def __init__(self, file=SETTINGS_YAML, version="0.0.4"):
        """Initialize the SettingsManager with default settings, load and validate current settings from the YAML
        file.
        """
        import copy
        import hashlib

        from .checks import check_version
        from .torch_utils import torch_distributed_zero_first

        root = GIT_DIR or Path()
        datasets_root = (root.parent if GIT_DIR and is_dir_writeable(root.parent) else root).resolve()

        self.file = Path(file)
        self.version = version
        self.defaults = {
            "settings_version": version,
            "datasets_dir": str(datasets_root / "datasets"),
            "weights_dir": str(root / "weights"),
            "runs_dir": str(root / "runs"),
            "uuid": hashlib.sha256(str(uuid.getnode()).encode()).hexdigest(),
            "sync": True,
            "api_key": "",
            "openai_api_key": "",
            "clearml": True,  # integrations
            "comet": True,
            "dvc": True,
            "hub": True,
            "mlflow": True,
            "neptune": True,
            "raytune": True,
            "tensorboard": True,
            "wandb": True,
        }

        super().__init__(copy.deepcopy(self.defaults))

        with torch_distributed_zero_first(RANK):
            if not self.file.exists():
                self.save()

            self.load()
            correct_keys = self.keys() == self.defaults.keys()
            correct_types = all(type(a) is type(b) for a, b in zip(self.values(), self.defaults.values()))
            correct_version = check_version(self["settings_version"], self.version)
            help_msg = (
                f"\nView settings with 'yolo settings' or at '{self.file}'"
                "\nUpdate settings with 'yolo settings key=value', i.e. 'yolo settings runs_dir=path/to/dir'. "
                "For help see https://docs.ultralytics.com/quickstart/#ultralytics-settings."
            )
            if not (correct_keys and correct_types and correct_version):
                print(
                    "WARNING ⚠️ Ultralytics settings reset to default values. This may be due to a possible problem "
                    f"with your settings or a recent ultralytics package update. {help_msg}"
                )
                self.reset()

            if self.get("datasets_dir") == self.get("runs_dir"):
                print(
                    f"WARNING ⚠️ Ultralytics setting 'datasets_dir: {self.get('datasets_dir')}' "
                    f"must be different than 'runs_dir: {self.get('runs_dir')}'. "
                    f"Please change one to avoid possible issues during training. {help_msg}"
                )

    def load(self):
        """Loads settings from the YAML file."""
        super().update(yaml_load(self.file))

    def save(self):
        """Saves the current settings to the YAML file."""
        yaml_save(self.file, dict(self))

    def update(self, *args, **kwargs):
        """Updates a setting value in the current settings."""
        super().update(*args, **kwargs)
        self.save()

    def reset(self):
        """Resets the settings to default and saves them."""
        self.clear()
        self.update(self.defaults)
        self.save()


def clean_url(url):
    """Strip auth from URL, i.e. https://url.com/file.txt?auth -> https://url.com/file.txt."""
    url = Path(url).as_posix().replace(":/", "://")  # Pathlib turns :// -> :/, as_posix() for Windows
    return urllib.parse.unquote(url).split("?")[0]  # '%2F' to '/', split https://url.com/file.txt?auth


def url2file(url):
    """Convert URL to filename, i.e. https://url.com/file.txt?auth -> file.txt."""
    return Path(clean_url(url)).name


# Run below code on util init ------------------------------------------------------------------------------------

# Check first-install steps
SETTINGS = SettingsManager()  # initialize settings
