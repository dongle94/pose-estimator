import contextlib
import json
import os
import platform
import re
import time
import warnings
from pathlib import Path
from threading import Lock
from typing import Union
from urllib.parse import unquote

import tqdm


# PyTorch Multi-GPU DDP Constants
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html

# Other Constants
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLO
DEFAULT_CFG_PATH = ROOT / "cfg/default.yaml"
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # number of YOLO multiprocessing threads
AUTOINSTALL = str(os.getenv("YOLO_AUTOINSTALL", True)).lower() == "true"  # global auto-install mode
VERBOSE = str(os.getenv("YOLO_VERBOSE", True)).lower() == "true"  # global verbose mode
TQDM_BAR_FORMAT = "{l_bar}{bar:10}{r_bar}" if VERBOSE else None  # tqdm bar format
MACOS, LINUX, WINDOWS = (platform.system() == x for x in ["Darwin", "Linux", "Windows"])  # environment booleans
MACOS_VERSION = platform.mac_ver()[0] if MACOS else None
ARM64 = platform.machine() in {"arm64", "aarch64"}  # ARM64 booleans
PYTHON_VERSION = platform.python_version()


if TQDM_RICH := str(os.getenv("YOLO_TQDM_RICH", False)).lower() == "true":
    from tqdm import rich


class TQDM(rich.tqdm if TQDM_RICH else tqdm.tqdm):
    """
    A custom TQDM progress bar class that extends the original tqdm functionality.

    This class modifies the behavior of the original tqdm progress bar based on global settings and provides
    additional customization options for Ultralytics projects. The progress bar is automatically disabled when
    VERBOSE is False or when explicitly disabled.

    Attributes:
        disable (bool): Whether to disable the progress bar. Determined by the global VERBOSE setting and
            any passed 'disable' argument.
        bar_format (str): The format string for the progress bar. Uses the global TQDM_BAR_FORMAT if not
            explicitly set.

    Methods:
        __init__: Initialize the TQDM object with custom settings.
        __iter__: Return self as iterator to satisfy Iterable interface.

    Examples:
        >>> from ultralytics.utils import TQDM
        >>> for i in TQDM(range(100)):
        ...     # Your processing code here
        ...     pass
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize a custom TQDM progress bar with Ultralytics-specific settings.

        Args:
            *args (Any): Variable length argument list to be passed to the original tqdm constructor.
            **kwargs (Any): Arbitrary keyword arguments to be passed to the original tqdm constructor.

        Notes:
            - The progress bar is disabled if VERBOSE is False or if 'disable' is explicitly set to True in kwargs.
            - The default bar format is set to TQDM_BAR_FORMAT unless overridden in kwargs.

        Examples:
            >>> from ultralytics.utils import TQDM
            >>> for i in TQDM(range(100)):
            ...     # Your code here
            ...     pass
        """
        warnings.filterwarnings("ignore", category=tqdm.TqdmExperimentalWarning)  # suppress tqdm.rich warning
        kwargs["disable"] = not VERBOSE or kwargs.get("disable", False)
        kwargs.setdefault("bar_format", TQDM_BAR_FORMAT)  # override default value if passed
        super().__init__(*args, **kwargs)

    def __iter__(self):
        """Return self as iterator to satisfy Iterable interface."""
        return super().__iter__()


class DataExportMixin:
    """
    Mixin class for exporting validation metrics or prediction results in various formats.

    This class provides utilities to export performance metrics (e.g., mAP, precision, recall) or prediction results
    from classification, object detection, segmentation, or pose estimation tasks into various formats: Pandas
    DataFrame, CSV, XML, HTML, JSON and SQLite (SQL).

    Methods:
        to_df: Convert summary to a Pandas DataFrame.
        to_csv: Export results as a CSV string.
        to_xml: Export results as an XML string (requires `lxml`).
        to_html: Export results as an HTML table.
        to_json: Export results as a JSON string.
        tojson: Deprecated alias for `to_json()`.
        to_sql: Export results to an SQLite database.

    Examples:
        >>> model = YOLO("yolo11n.pt")
        >>> results = model("image.jpg")
        >>> df = results.to_df()
        >>> print(df)
        >>> csv_data = results.to_csv()
        >>> results.to_sql(table_name="yolo_results")
    """

    def to_df(self, normalize=False, decimals=5):
        """
        Create a pandas DataFrame from the prediction results summary or validation metrics.

        Args:
            normalize (bool, optional): Normalize numerical values for easier comparison.
            decimals (int, optional): Decimal places to round floats.

        Returns:
            (DataFrame): DataFrame containing the summary data.
        """
        import pandas as pd  # scope for faster 'import ultralytics'

        return pd.DataFrame(self.summary(normalize=normalize, decimals=decimals))

    def to_csv(self, normalize=False, decimals=5):
        """
        Export results to CSV string format.

        Args:
           normalize (bool, optional): Normalize numeric values.
           decimals (int, optional): Decimal precision.

        Returns:
           (str): CSV content as string.
        """
        return self.to_df(normalize=normalize, decimals=decimals).to_csv()

    def to_xml(self, normalize=False, decimals=5):
        """
        Export results to XML format.

        Args:
            normalize (bool, optional): Normalize numeric values.
            decimals (int, optional): Decimal precision.

        Returns:
            (str): XML string.

        Notes:
            Requires `lxml` package to be installed.
        """
        df = self.to_df(normalize=normalize, decimals=decimals)
        return '<?xml version="1.0" encoding="utf-8"?>\n<root></root>' if df.empty else df.to_xml(parser="etree")

    def to_html(self, normalize=False, decimals=5, index=False):
        """
        Export results to HTML table format.

        Args:
            normalize (bool, optional): Normalize numeric values.
            decimals (int, optional): Decimal precision.
            index (bool, optional): Whether to include index column in the HTML table.

        Returns:
            (str): HTML representation of the results.
        """
        df = self.to_df(normalize=normalize, decimals=decimals)
        return "<table></table>" if df.empty else df.to_html(index=index)

    def tojson(self, normalize=False, decimals=5):
        """Deprecated version of to_json()."""
        print("Warning: 'result.tojson()' is deprecated, replace with 'result.to_json()'.")
        return self.to_json(normalize, decimals)

    def to_json(self, normalize=False, decimals=5):
        """
        Export results to JSON format.

        Args:
            normalize (bool, optional): Normalize numeric values.
            decimals (int, optional): Decimal precision.

        Returns:
            (str): JSON-formatted string of the results.
        """
        return self.to_df(normalize=normalize, decimals=decimals).to_json(orient="records", indent=2)

    def to_sql(self, normalize=False, decimals=5, table_name="results", db_path="results.db"):
        """
        Save results to an SQLite database.

        Args:
            normalize (bool, optional): Normalize numeric values.
            decimals (int, optional): Decimal precision.
            table_name (str, optional): Name of the SQL table.
            db_path (str, optional): SQLite database file path.
        """
        df = self.to_df(normalize, decimals)
        if df.empty or df.columns.empty:  # Exit if df is None or has no columns (i.e., no schema)
            return

        import sqlite3

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Dynamically create table schema based on summary to support prediction and validation results export
        columns = []
        for col in df.columns:
            sample_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else ""
            if isinstance(sample_val, dict):
                col_type = "TEXT"
            elif isinstance(sample_val, (float, int)):
                col_type = "REAL"
            else:
                col_type = "TEXT"
            columns.append(f'"{col}" {col_type}')  # Quote column names to handle special characters like hyphens

        # Create table (Drop table from db if it's already exist)
        cursor.execute(f'DROP TABLE IF EXISTS "{table_name}"')
        cursor.execute(f'CREATE TABLE "{table_name}" (id INTEGER PRIMARY KEY AUTOINCREMENT, {", ".join(columns)})')

        for _, row in df.iterrows():
            values = [json.dumps(v) if isinstance(v, dict) else v for v in row]
            column_names = ", ".join(f'"{col}"' for col in df.columns)
            placeholders = ", ".join("?" for _ in df.columns)
            cursor.execute(f'INSERT INTO "{table_name}" ({column_names}) VALUES ({placeholders})', values)

        conn.commit()
        conn.close()
        print(f"Info: Results saved to SQL table '{table_name}' in '{db_path}'.")


class SimpleClass:
    """
    A simple base class for creating objects with string representations of their attributes.

    This class provides a foundation for creating objects that can be easily printed or represented as strings,
    showing all their non-callable attributes. It's useful for debugging and introspection of object states.

    Methods:
        __str__: Return a human-readable string representation of the object.
        __repr__: Return a machine-readable string representation of the object.
        __getattr__: Provide a custom attribute access error message with helpful information.

    Examples:
        >>> class MyClass(SimpleClass):
        ...     def __init__(self):
        ...         self.x = 10
        ...         self.y = "hello"
        >>> obj = MyClass()
        >>> print(obj)
        __main__.MyClass object with attributes:

        x: 10
        y: 'hello'

    Notes:
        - This class is designed to be subclassed. It provides a convenient way to inspect object attributes.
        - The string representation includes the module and class name of the object.
        - Callable attributes and attributes starting with an underscore are excluded from the string representation.
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
        """Provide a custom attribute access error message with helpful information."""
        name = self.__class__.__name__
        raise AttributeError(f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}")


def plt_settings(rcparams=None, backend="Agg"):
    """
    Decorator to temporarily set rc parameters and the backend for a plotting function.

    Args:
        rcparams (dict, optional): Dictionary of rc parameters to set.
        backend (str, optional): Name of the backend to use.

    Returns:
        (Callable): Decorated function with temporarily set rc parameters and backend.

    Examples:
        >>> @plt_settings({"font.size": 12})
        >>> def plot_function():
        ...     plt.figure()
        ...     plt.plot([1, 2, 3])
        ...     plt.show()

        >>> with plt_settings({"font.size": 12}):
        ...     plt.figure()
        ...     plt.plot([1, 2, 3])
        ...     plt.show()
    """
    if rcparams is None:
        rcparams = {"font.size": 11}

    def decorator(func):
        """Decorator to apply temporary rc parameters and backend to a function."""

        def wrapper(*args, **kwargs):
            """Set rc parameters and backend, call the original function, and restore the settings."""
            import matplotlib.pyplot as plt  # scope for faster 'import ultralytics'

            original_backend = plt.get_backend()
            switch = backend.lower() != original_backend.lower()
            if switch:
                plt.close("all")  # auto-close()ing of figures upon backend switching is deprecated since 3.8
                plt.switch_backend(backend)

            # Plot with backend and always revert to original backend
            try:
                with plt.rc_context(rcparams):
                    result = func(*args, **kwargs)
            finally:
                if switch:
                    plt.close("all")
                    plt.switch_backend(original_backend)
            return result

        return wrapper

    return decorator


def emojis(string=""):
    """Return platform-dependent emoji-safe version of string."""
    return string.encode().decode("ascii", "ignore") if WINDOWS else string


class YAML:
    """
    YAML utility class for efficient file operations with automatic C-implementation detection.

    This class provides optimized YAML loading and saving operations using PyYAML's fastest available implementation
    (C-based when possible). It implements a singleton pattern with lazy initialization, allowing direct class method
    usage without explicit instantiation. The class handles file path creation, validation, and character encoding
    issues automatically.

    The implementation prioritizes performance through:
        - Automatic C-based loader/dumper selection when available
        - Singleton pattern to reuse the same instance
        - Lazy initialization to defer import costs until needed
        - Fallback mechanisms for handling problematic YAML content

    Attributes:
        _instance: Internal singleton instance storage.
        yaml: Reference to the PyYAML module.
        SafeLoader: Best available YAML loader (CSafeLoader if available).
        SafeDumper: Best available YAML dumper (CSafeDumper if available).

    Examples:
        >>> data = YAML.load("config.yaml")
        >>> data["new_value"] = 123
        >>> YAML.save("updated_config.yaml", data)
        >>> YAML.print(data)
    """

    _instance = None

    @classmethod
    def _get_instance(cls):
        """Initialize singleton instance on first use."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        """Initialize with optimal YAML implementation (C-based when available)."""
        import yaml

        self.yaml = yaml
        # Use C-based implementation if available for better performance
        try:
            self.SafeLoader = yaml.CSafeLoader
            self.SafeDumper = yaml.CSafeDumper
        except (AttributeError, ImportError):
            self.SafeLoader = yaml.SafeLoader
            self.SafeDumper = yaml.SafeDumper

    @classmethod
    def save(cls, file="data.yaml", data=None, header=""):
        """
        Save Python object as YAML file.

        Args:
            file (str | Path): Path to save YAML file.
            data (dict | None): Dict or compatible object to save.
            header (str): Optional string to add at file beginning.
        """
        instance = cls._get_instance()
        if data is None:
            data = {}

        # Create parent directories if needed
        file = Path(file)
        file.parent.mkdir(parents=True, exist_ok=True)

        # Convert non-serializable objects to strings
        valid_types = int, float, str, bool, list, tuple, dict, type(None)
        for k, v in data.items():
            if not isinstance(v, valid_types):
                data[k] = str(v)

        # Write YAML file
        with open(file, "w", errors="ignore", encoding="utf-8") as f:
            if header:
                f.write(header)
            instance.yaml.dump(data, f, sort_keys=False, allow_unicode=True, Dumper=instance.SafeDumper)

    @classmethod
    def load(cls, file="data.yaml", append_filename=False):
        """
        Load YAML file to Python object with robust error handling.

        Args:
            file (str | Path): Path to YAML file.
            append_filename (bool): Whether to add filename to returned dict.

        Returns:
            (dict): Loaded YAML content.
        """
        instance = cls._get_instance()
        assert str(file).endswith((".yaml", ".yml")), f"Not a YAML file: {file}"

        # Read file content
        with open(file, errors="ignore", encoding="utf-8") as f:
            s = f.read()

        # Try loading YAML with fallback for problematic characters
        try:
            data = instance.yaml.load(s, Loader=instance.SafeLoader) or {}
        except Exception:
            # Remove problematic characters and retry
            s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+", "", s)
            data = instance.yaml.load(s, Loader=instance.SafeLoader) or {}

        # Check for accidental user-error None strings (should be 'null' in YAML)
        if "None" in data.values():
            data = {k: None if v == "None" else v for k, v in data.items()}

        if append_filename:
            data["yaml_file"] = str(file)
        return data

    @classmethod
    def print(cls, yaml_file):
        """
        Pretty print YAML file or object to console.

        Args:
            yaml_file (str | Path | dict): Path to YAML file or dict to print.
        """
        instance = cls._get_instance()

        # Load file if path provided
        yaml_dict = cls.load(yaml_file) if isinstance(yaml_file, (str, Path)) else yaml_file

        # Use -1 for unlimited width in C implementation
        dump = instance.yaml.dump(yaml_dict, sort_keys=False, allow_unicode=True, width=-1, Dumper=instance.SafeDumper)

        print(f"Info: Printing '{colorstr('bold', 'black', yaml_file)}'\n\n{dump}")


# Default configuration
DEFAULT_CFG_DICT = YAML.load(DEFAULT_CFG_PATH)
DEFAULT_CFG_KEYS = DEFAULT_CFG_DICT.keys()


def is_online() -> bool:
    """
    Check internet connectivity by attempting to connect to a known online host.

    Returns:
        (bool): True if connection is successful, False otherwise.
    """
    try:
        assert str(os.getenv("YOLO_OFFLINE", "")).lower() != "true"  # check if ENV var YOLO_OFFLINE="True"
        import socket

        for dns in ("1.1.1.1", "8.8.8.8"):  # check Cloudflare and Google DNS
            socket.create_connection(address=(dns, 80), timeout=2.0).close()
            return True
    except Exception:
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
    Determine whether the current file is part of a git repository and if so, return the repository root directory.

    Returns:
        (Path | None): Git root directory if found or None if not found.
    """
    for d in Path(__file__).parents:
        if (d / ".git").is_dir():
            return d


def get_user_config_dir(sub_dir="Yolo"):
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
            f"Warning: user config directory '{path}' is not writeable, defaulting to '/tmp' or CWD."
            "Alternatively you can define a YOLO_CONFIG_DIR environment variable for this path."
        )
        path = Path("/tmp") / sub_dir if is_dir_writeable("/tmp") else Path().cwd() / sub_dir

    # Create the subdirectory if it does not exist
    path.mkdir(parents=True, exist_ok=True)

    return path


# Define constants (required below)
ONLINE = is_online()
GIT_DIR = get_git_dir()
USER_CONFIG_DIR = Path(os.getenv("YOLO_CONFIG_DIR") or get_user_config_dir())  # Ultralytics settings dir
SETTINGS_FILE = USER_CONFIG_DIR / "settings.json"


def colorstr(*input):
    r"""
    Color a string based on the provided color and style arguments using ANSI escape codes.

    This function can be called in two ways:
        - colorstr('color', 'style', 'your string')
        - colorstr('your string')

    In the second form, 'blue' and 'bold' will be applied by default.

    Args:
        *input (str | Path): A sequence of strings where the first n-1 strings are color and style arguments,
                      and the last string is the one to be colored.

    Returns:
        (str): The input string wrapped with ANSI escape codes for the specified color and style.

    Notes:
        Supported Colors and Styles:
        - Basic Colors: 'black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'
        - Bright Colors: 'bright_black', 'bright_red', 'bright_green', 'bright_yellow',
                       'bright_blue', 'bright_magenta', 'bright_cyan', 'bright_white'
        - Misc: 'end', 'bold', 'underline'

    Examples:
        >>> colorstr("blue", "bold", "hello world")
        >>> "\033[34m\033[1mhello world\033[0m"

    References:
        https://en.wikipedia.org/wiki/ANSI_escape_code
    """
    *args, string = input if len(input) > 1 else ("blue", "bold", input[0])  # color arguments, string
    colors = {
        "black": "\033[30m",  # basic colors
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",  # bright colors
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "end": "\033[0m",  # misc
        "bold": "\033[1m",
        "underline": "\033[4m",
    }
    return "".join(colors[x] for x in args) + f"{string}" + colors["end"]


class TryExcept(contextlib.ContextDecorator):
    """
    Ultralytics TryExcept class for handling exceptions gracefully.

    This class can be used as a decorator or context manager to catch exceptions and optionally print warning messages.
    It allows code to continue execution even when exceptions occur, which is useful for non-critical operations.

    Attributes:
        msg (str): Optional message to display when an exception occurs.
        verbose (bool): Whether to print the exception message.

    Examples:
        As a decorator:
        >>> @TryExcept(msg="Error occurred in func", verbose=True)
        >>> def func():
        >>> # Function logic here
        >>>     pass

        As a context manager:
        >>> with TryExcept(msg="Error occurred in block", verbose=True):
        >>> # Code block here
        >>>     pass
    """

    def __init__(self, msg="", verbose=True):
        """Initialize TryExcept class with optional message and verbosity settings."""
        self.msg = msg
        self.verbose = verbose

    def __enter__(self):
        """Execute when entering TryExcept context, initialize instance."""
        pass

    def __exit__(self, exc_type, value, traceback):
        """Define behavior when exiting a 'with' block, print error message if necessary."""
        if self.verbose and value:
            print(f"Warning: {self.msg}{': ' if self.msg else ''}{value}")
        return True


class Retry(contextlib.ContextDecorator):
    """
    Retry class for function execution with exponential backoff.

    This decorator can be used to retry a function on exceptions, up to a specified number of times with an
    exponentially increasing delay between retries. It's useful for handling transient failures in network
    operations or other unreliable processes.

    Attributes:
        times (int): Maximum number of retry attempts.
        delay (int): Initial delay between retries in seconds.

    Examples:
        Example usage as a decorator:
        >>> @Retry(times=3, delay=2)
        >>> def test_func():
        >>> # Replace with function logic that may raise exceptions
        >>>     return True
    """

    def __init__(self, times=3, delay=2):
        """Initialize Retry class with specified number of retries and delay."""
        self.times = times
        self.delay = delay
        self._attempts = 0

    def __call__(self, func):
        """Decorator implementation for Retry with exponential backoff."""

        def wrapped_func(*args, **kwargs):
            """Apply retries to the decorated function or method."""
            self._attempts = 0
            while self._attempts < self.times:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    self._attempts += 1
                    print(f"Warning: Retry {self._attempts}/{self.times} failed: {e}")
                    if self._attempts >= self.times:
                        raise e
                    time.sleep(self.delay * (2**self._attempts))  # exponential backoff delay

        return wrapped_func


class JSONDict(dict):
    """
    A dictionary-like class that provides JSON persistence for its contents.

    This class extends the built-in dictionary to automatically save its contents to a JSON file whenever they are
    modified. It ensures thread-safe operations using a lock and handles JSON serialization of Path objects.

    Attributes:
        file_path (Path): The path to the JSON file used for persistence.
        lock (threading.Lock): A lock object to ensure thread-safe operations.

    Methods:
        _load: Load the data from the JSON file into the dictionary.
        _save: Save the current state of the dictionary to the JSON file.
        __setitem__: Store a key-value pair and persist it to disk.
        __delitem__: Remove an item and update the persistent storage.
        update: Update the dictionary and persist changes.
        clear: Clear all entries and update the persistent storage.

    Examples:
        >>> json_dict = JSONDict("data.json")
        >>> json_dict["key"] = "value"
        >>> print(json_dict["key"])
        value
        >>> del json_dict["key"]
        >>> json_dict.update({"new_key": "new_value"})
        >>> json_dict.clear()
    """

    def __init__(self, file_path: Union[str, Path] = "data.json"):
        """Initialize a JSONDict object with a specified file path for JSON persistence."""
        super().__init__()
        self.file_path = Path(file_path)
        self.lock = Lock()
        self._load()

    def _load(self):
        """Load the data from the JSON file into the dictionary."""
        try:
            if self.file_path.exists():
                with open(self.file_path) as f:
                    self.update(json.load(f))
        except json.JSONDecodeError:
            print(f"Warning: Error decoding JSON from {self.file_path}. Starting with an empty dictionary.")
        except Exception as e:
            print(f"Error: Error reading from {self.file_path}: {e}")

    def _save(self):
        """Save the current state of the dictionary to the JSON file."""
        try:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(dict(self), f, indent=2, default=self._json_default)
        except Exception as e:
            print(f"Error: Error writing to {self.file_path}: {e}")

    @staticmethod
    def _json_default(obj):
        """Handle JSON serialization of Path objects."""
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    def __setitem__(self, key, value):
        """Store a key-value pair and persist to disk."""
        with self.lock:
            super().__setitem__(key, value)
            self._save()

    def __delitem__(self, key):
        """Remove an item and update the persistent storage."""
        with self.lock:
            super().__delitem__(key)
            self._save()

    def __str__(self):
        """Return a pretty-printed JSON string representation of the dictionary."""
        contents = json.dumps(dict(self), indent=2, ensure_ascii=False, default=self._json_default)
        return f'JSONDict("{self.file_path}"):\n{contents}'

    def update(self, *args, **kwargs):
        """Update the dictionary and persist changes."""
        with self.lock:
            super().update(*args, **kwargs)
            self._save()

    def clear(self):
        """Clear all entries and update the persistent storage."""
        with self.lock:
            super().clear()
            self._save()


class SettingsManager(JSONDict):
    """
    SettingsManager class for managing and persisting Ultralytics settings.

    This class extends JSONDict to provide JSON persistence for settings, ensuring thread-safe operations and default
    values. It validates settings on initialization and provides methods to update or reset settings. The settings
    include directories for datasets, weights, and runs, as well as various integration flags.

    Attributes:
        file (Path): The path to the JSON file used for persistence.
        version (str): The version of the settings schema.
        defaults (dict): A dictionary containing default settings.
        help_msg (str): A help message for users on how to view and update settings.

    Methods:
        _validate_settings: Validate the current settings and reset if necessary.
        update: Update settings, validating keys and types.
        reset: Reset the settings to default and save them.

    Examples:
        Initialize and update settings:
        >>> settings = SettingsManager()
        >>> settings.update(runs_dir="/new/runs/dir")
        >>> print(settings["runs_dir"])
        /new/runs/dir
    """

    def __init__(self, file=SETTINGS_FILE, version="0.0.6"):
        """Initialize the SettingsManager with default settings and load user settings."""
        import hashlib
        import uuid

        from core.yolo.util.torch_utils import torch_distributed_zero_first

        root = GIT_DIR or Path()
        datasets_root = (root.parent if GIT_DIR and is_dir_writeable(root.parent) else root).resolve()

        self.file = Path(file)
        self.version = version
        self.defaults = {
            "settings_version": version,  # Settings schema version
            "datasets_dir": str(datasets_root / "datasets"),  # Datasets directory
            "weights_dir": str(root / "weights"),  # Model weights directory
            "runs_dir": str(root / "runs"),  # Experiment runs directory
            "uuid": hashlib.sha256(str(uuid.getnode()).encode()).hexdigest(),  # SHA-256 anonymized UUID hash
            "sync": True,  # Enable synchronization
            "api_key": "",  # Ultralytics API Key
            "openai_api_key": "",  # OpenAI API Key
            "clearml": True,  # ClearML integration
            "comet": True,  # Comet integration
            "dvc": True,  # DVC integration
            "hub": True,  # Ultralytics HUB integration
            "mlflow": True,  # MLflow integration
            "neptune": True,  # Neptune integration
            "raytune": True,  # Ray Tune integration
            "tensorboard": False,  # TensorBoard logging
            "wandb": False,  # Weights & Biases logging
            "vscode_msg": True,  # VSCode message
            "openvino_msg": True,  # OpenVINO export on Intel CPU message
        }

        self.help_msg = (
            f"\nView Ultralytics Settings with 'yolo settings' or at '{self.file}'"
            "\nUpdate Settings with 'yolo settings key=value', i.e. 'yolo settings runs_dir=path/to/dir'. "
            "For help see https://docs.ultralytics.com/quickstart/#ultralytics-settings."
        )

        with torch_distributed_zero_first(LOCAL_RANK):
            super().__init__(self.file)

            if not self.file.exists() or not self:  # Check if file doesn't exist or is empty
                print(f"Warning: Creating new Ultralytics Settings v{version} file ✅ {self.help_msg}")
                self.reset()

            self._validate_settings()

    def _validate_settings(self):
        """Validate the current settings and reset if necessary."""
        correct_keys = frozenset(self.keys()) == frozenset(self.defaults.keys())
        correct_types = all(isinstance(self.get(k), type(v)) for k, v in self.defaults.items())
        correct_version = self.get("settings_version", "") == self.version

        if not (correct_keys and correct_types and correct_version):
            print(
                "Warning: Ultralytics settings reset to default values. This may be due to a possible problem "
                f"with your settings or a recent ultralytics package update. {self.help_msg}"
            )
            self.reset()

        if self.get("datasets_dir") == self.get("runs_dir"):
            print(
                f"Warnging: Ultralytics setting 'datasets_dir: {self.get('datasets_dir')}' "
                f"must be different than 'runs_dir: {self.get('runs_dir')}'. "
                f"Please change one to avoid possible issues during training. {self.help_msg}"
            )

    def __setitem__(self, key, value):
        """Update one key: value pair."""
        self.update({key: value})

    def update(self, *args, **kwargs):
        """Update settings, validating keys and types."""
        for arg in args:
            if isinstance(arg, dict):
                kwargs.update(arg)
        for k, v in kwargs.items():
            if k not in self.defaults:
                raise KeyError(f"No Ultralytics setting '{k}'. {self.help_msg}")
            t = type(self.defaults[k])
            if not isinstance(v, t):
                raise TypeError(
                    f"Ultralytics setting '{k}' must be '{t.__name__}' type, not '{type(v).__name__}'. {self.help_msg}"
                )
        super().update(*args, **kwargs)

    def reset(self):
        """Reset the settings to default and save them."""
        self.clear()
        self.update(self.defaults)


def clean_url(url):
    """Strip auth from URL, i.e. https://url.com/file.txt?auth -> https://url.com/file.txt."""
    url = Path(url).as_posix().replace(":/", "://")  # Pathlib turns :// -> :/, as_posix() for Windows
    return unquote(url).split("?", 1)[0]  # '%2F' to '/', split https://url.com/file.txt?auth


def url2file(url):
    """Convert URL to filename, i.e. https://url.com/file.txt?auth -> file.txt."""
    return Path(clean_url(url)).name


# Check first-install steps
SETTINGS = SettingsManager()  # initialize settings
PERSISTENT_CACHE = JSONDict(USER_CONFIG_DIR / "persistent_cache.json")  # initialize persistent cache
