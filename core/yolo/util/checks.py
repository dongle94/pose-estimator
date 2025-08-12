import functools
import glob
import math
import re
import subprocess
import time
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import torch

from core.yolo.util import (
    ARM64,
    AUTOINSTALL,
    LINUX,
    MACOS,
    ONLINE,
    ROOT,
    WINDOWS,
    Retry,
    TryExcept,
    clean_url,
    colorstr,
    downloads,
    url2file,
)

def parse_requirements(file_path=ROOT.parent / "requirements.txt", package=""):
    """
    Parse a requirements.txt file, ignoring lines that start with '#' and any text after '#'.

    Args:
        file_path (Path): Path to the requirements.txt file.
        package (str, optional): Python package to use instead of requirements.txt file.

    Returns:
        requirements (List[SimpleNamespace]): List of parsed requirements as SimpleNamespace objects with `name` and
            `specifier` attributes.

    Examples:
        >>> from ultralytics.utils.checks import parse_requirements
        >>> parse_requirements(package="ultralytics")
    """
    if package:
        requires = [x for x in metadata.distribution(package).requires if "extra == " not in x]
    else:
        requires = Path(file_path).read_text().splitlines()

    requirements = []
    for line in requires:
        line = line.strip()
        if line and not line.startswith("#"):
            line = line.partition("#")[0].strip()  # ignore inline comments
            if match := re.match(r"([a-zA-Z0-9-_]+)\s*([<>!=~]+.*)?", line):
                requirements.append(SimpleNamespace(name=match[1], specifier=match[2].strip() if match[2] else ""))

    return requirements


@functools.lru_cache
def parse_version(version="0.0.0") -> tuple:
    """
    Convert a version string to a tuple of integers, ignoring any extra non-numeric string attached to the version.

    Args:
        version (str): Version string, i.e. '2.0.1+cpu'

    Returns:
        (tuple): Tuple of integers representing the numeric part of the version, i.e. (2, 0, 1)
    """
    try:
        return tuple(map(int, re.findall(r"\d+", version)[:3]))  # '2.0.1+cpu' -> (2, 0, 1)
    except Exception as e:
        print(f"Warning: failure for parse_version({version}), returning (0, 0, 0): {e}")
        return 0, 0, 0


def check_imgsz(imgsz, stride=32, min_dim=1, max_dim=2, floor=0):
    """
    Verify image size is a multiple of the given stride in each dimension. If the image size is not a multiple of the
    stride, update it to the nearest multiple of the stride that is greater than or equal to the given floor value.

    Args:
        imgsz (int | List[int]): Image size.
        stride (int): Stride value.
        min_dim (int): Minimum number of dimensions.
        max_dim (int): Maximum number of dimensions.
        floor (int): Minimum allowed value for image size.

    Returns:
        (List[int] | int): Updated image size.
    """
    # Convert stride to integer if it is a tensor
    stride = int(stride.max() if isinstance(stride, torch.Tensor) else stride)

    # Convert image size to list if it is an integer
    if isinstance(imgsz, int):
        imgsz = [imgsz]
    elif isinstance(imgsz, (list, tuple)):
        imgsz = list(imgsz)
    elif isinstance(imgsz, str):  # i.e. '640' or '[640,640]'
        imgsz = [int(imgsz)] if imgsz.isnumeric() else eval(imgsz)
    else:
        raise TypeError(
            f"'imgsz={imgsz}' is of invalid type {type(imgsz).__name__}. "
            f"Valid imgsz types are int i.e. 'imgsz=640' or list i.e. 'imgsz=[640,640]'"
        )

    # Apply max_dim
    if len(imgsz) > max_dim:
        msg = (
            "'train' and 'val' imgsz must be an integer, while 'predict' and 'export' imgsz may be a [h, w] list "
            "or an integer, i.e. 'yolo export imgsz=640,480' or 'yolo export imgsz=640'"
        )
        if max_dim != 1:
            raise ValueError(f"imgsz={imgsz} is not a valid image size. {msg}")
        print(f"Warning: updating to 'imgsz={max(imgsz)}'. {msg}")
        imgsz = [max(imgsz)]
    # Make image size a multiple of the stride
    sz = [max(math.ceil(x / stride) * stride, floor) for x in imgsz]

    # Print warning message if image size was updated
    if sz != imgsz:
        print(f"Warning: imgsz={imgsz} must be multiple of max stride {stride}, updating to {sz}")

    # Add missing dimensions if necessary
    sz = [sz[0], sz[0]] if min_dim == 2 and len(sz) == 1 else sz[0] if min_dim == 1 and len(sz) == 1 else sz

    return sz


@functools.lru_cache
def check_uv():
    """Check if uv package manager is installed and can run successfully."""
    try:
        return subprocess.run(["uv", "-V"], capture_output=True).returncode == 0
    except FileNotFoundError:
        return False


@functools.lru_cache
def check_version(
    current: str = "0.0.0",
    required: str = "0.0.0",
    name: str = "version",
    hard: bool = False,
    verbose: bool = False,
    msg: str = "",
) -> bool:
    """
    Check current version against the required version or range.

    Args:
        current (str): Current version or package name to get version from.
        required (str): Required version or range (in pip-style format).
        name (str): Name to be used in warning message.
        hard (bool): If True, raise an AssertionError if the requirement is not met.
        verbose (bool): If True, print warning message if requirement is not met.
        msg (str): Extra message to display if verbose.

    Returns:
        (bool): True if requirement is met, False otherwise.

    Examples:
        Check if current version is exactly 22.04
        >>> check_version(current="22.04", required="==22.04")

        Check if current version is greater than or equal to 22.04
        >>> check_version(current="22.10", required="22.04")  # assumes '>=' inequality if none passed

        Check if current version is less than or equal to 22.04
        >>> check_version(current="22.04", required="<=22.04")

        Check if current version is between 20.04 (inclusive) and 22.04 (exclusive)
        >>> check_version(current="21.10", required=">20.04,<22.04")
    """
    if not current:  # if current is '' or None
        print(f"Warning: invalid check_version({current}, {required}) requested, please check values.")
        return True
    elif not current[0].isdigit():  # current is package name rather than version string, i.e. current='ultralytics'
        try:
            name = current  # assigned package name to 'name' arg
            current = metadata.version(current)  # get version string from package name
        except metadata.PackageNotFoundError as e:
            if hard:
                raise ModuleNotFoundError(f"{current} package is required but not installed") from e
            else:
                return False

    if not required:  # if required is '' or None
        return True

    if "sys_platform" in required and (  # i.e. required='<2.4.0,>=1.8.0; sys_platform == "win32"'
        (WINDOWS and "win32" not in required)
        or (LINUX and "linux" not in required)
        or (MACOS and "macos" not in required and "darwin" not in required)
    ):
        return True

    op = ""
    version = ""
    result = True
    c = parse_version(current)  # '1.2.3' -> (1, 2, 3)
    for r in required.strip(",").split(","):
        op, version = re.match(r"([^0-9]*)([\d.]+)", r).groups()  # split '>=22.04' -> ('>=', '22.04')
        if not op:
            op = ">="  # assume >= if no op passed
        v = parse_version(version)  # '1.2.3' -> (1, 2, 3)
        if op == "==" and c != v:
            result = False
        elif op == "!=" and c == v:
            result = False
        elif op == ">=" and not (c >= v):
            result = False
        elif op == "<=" and not (c <= v):
            result = False
        elif op == ">" and not (c > v):
            result = False
        elif op == "<" and not (c < v):
            result = False
    if not result:
        warning = f"{name}{required} is required, but {name}=={current} is currently installed {msg}"
        if hard:
            raise ModuleNotFoundError(warning)  # assert version requirements met
        if verbose:
            print("Warning:", warning)
    return result


@TryExcept()
def check_requirements(requirements=ROOT.parent / "requirements.txt", exclude=(), install=True, cmds=""):
    """
    Check if installed dependencies meet Ultralytics YOLO models requirements and attempt to auto-update if needed.

    Args:
        requirements (Path | str | List[str]): Path to a requirements.txt file, a single package requirement as a
            string, or a list of package requirements as strings.
        exclude (tuple): Tuple of package names to exclude from checking.
        install (bool): If True, attempt to auto-update packages that don't meet requirements.
        cmds (str): Additional commands to pass to the pip install command when auto-updating.

    Examples:
        >>> from ultralytics.utils.checks import check_requirements

        Check a requirements.txt file
        >>> check_requirements("path/to/requirements.txt")

        Check a single package
        >>> check_requirements("ultralytics>=8.0.0")

        Check multiple packages
        >>> check_requirements(["numpy", "ultralytics>=8.0.0"])
    """
    prefix = colorstr("red", "bold", "requirements:")
    if isinstance(requirements, Path):  # requirements.txt file
        file = requirements.resolve()
        assert file.exists(), f"{prefix} {file} not found, check failed."
        requirements = [f"{x.name}{x.specifier}" for x in parse_requirements(file) if x.name not in exclude]
    elif isinstance(requirements, str):
        requirements = [requirements]

    pkgs = []
    for r in requirements:
        r_stripped = r.rpartition("/")[-1].replace(".git", "")  # replace git+https://org/repo.git -> 'repo'
        match = re.match(r"([a-zA-Z0-9-_]+)([<>!=~]+.*)?", r_stripped)
        name, required = match[1], match[2].strip() if match[2] else ""
        try:
            assert check_version(metadata.version(name), required)  # exception if requirements not met
        except (AssertionError, metadata.PackageNotFoundError):
            pkgs.append(r)

    @Retry(times=2, delay=1)
    def attempt_install(packages, commands, use_uv):
        """Attempt package installation with uv if available, falling back to pip."""
        if use_uv:
            base = f"uv pip install --no-cache-dir {packages} {commands} --index-strategy=unsafe-best-match --break-system-packages --prerelease=allow"
            try:
                return subprocess.check_output(base, shell=True, stderr=subprocess.PIPE).decode()
            except subprocess.CalledProcessError as e:
                if e.stderr and "No virtual environment found" in e.stderr.decode():
                    return subprocess.check_output(
                        base.replace("uv pip install", "uv pip install --system"), shell=True
                    ).decode()
                raise
        return subprocess.check_output(f"pip install --no-cache-dir {packages} {commands}", shell=True).decode()

    s = " ".join(f'"{x}"' for x in pkgs)  # console string
    if s:
        if install and AUTOINSTALL:  # check environment variable
            # Note uv fails on arm64 macOS and Raspberry Pi runners
            n = len(pkgs)  # number of packages updates
            print(f"Info: {prefix} Ultralytics requirement{'s' * (n > 1)} {pkgs} not found, attempting AutoUpdate...")
            try:
                t = time.time()
                assert ONLINE, "AutoUpdate skipped (offline)"
                print("Info: ",attempt_install(s, cmds, use_uv=not ARM64 and check_uv()))
                dt = time.time() - t
                print(f"Info: {prefix} AutoUpdate success ✅ {dt:.1f}s")
                print(
                    f"Warning: {prefix} {colorstr('bold', 'Restart runtime or rerun command for updates to take effect')}\n"
                )
            except Exception as e:
                print(f"Warning: {prefix} ❌ {e}")
                return False
        else:
            return False

    return True


def check_suffix(file="yolo11n.pt", suffix=".pt", msg=""):
    """
    Check file(s) for acceptable suffix.

    Args:
        file (str | List[str]): File or list of files to check.
        suffix (str | tuple): Acceptable suffix or tuple of suffixes.
        msg (str): Additional message to display in case of error.
    """
    if file and suffix:
        if isinstance(suffix, str):
            suffix = {suffix}
        for f in file if isinstance(file, (list, tuple)) else [file]:
            if s := str(f).rpartition(".")[-1].lower().strip():  # file suffix
                assert f".{s}" in suffix, f"{msg}{f} acceptable suffix is {suffix}, not .{s}"


def check_yolov5u_filename(file: str, verbose: bool = True):
    """
    Replace legacy YOLOv5 filenames with updated YOLOv5u filenames.

    Args:
        file (str): Filename to check and potentially update.
        verbose (bool): Whether to print information about the replacement.

    Returns:
        (str): Updated filename.
    """
    if "yolov3" in file or "yolov5" in file:
        if "u.yaml" in file:
            file = file.replace("u.yaml", ".yaml")  # i.e. yolov5nu.yaml -> yolov5n.yaml
        elif ".pt" in file and "u" not in file:
            original_file = file
            file = re.sub(r"(.*yolov5([nsmlx]))\.pt", "\\1u.pt", file)  # i.e. yolov5n.pt -> yolov5nu.pt
            file = re.sub(r"(.*yolov5([nsmlx])6)\.pt", "\\1u.pt", file)  # i.e. yolov5n6.pt -> yolov5n6u.pt
            file = re.sub(r"(.*yolov3(|-tiny|-spp))\.pt", "\\1u.pt", file)  # i.e. yolov3-spp.pt -> yolov3-sppu.pt
            if file != original_file and verbose:
                print("Info: "
                    f"PRO TIP 💡 Replace 'model={original_file}' with new 'model={file}'.\nYOLOv5 'u' models are "
                    f"trained with https://github.com/ultralytics/ultralytics and feature improved performance vs "
                    f"standard YOLOv5 models trained with https://github.com/ultralytics/yolov5.\n"
                )
    return file


def check_file(file, suffix="", download=True, download_dir=".", hard=True):
    """
    Search/download file (if necessary), check suffix (if provided), and return path.

    Args:
        file (str): File name or path.
        suffix (str | tuple): Acceptable suffix or tuple of suffixes to validate against the file.
        download (bool): Whether to download the file if it doesn't exist locally.
        download_dir (str): Directory to download the file to.
        hard (bool): Whether to raise an error if the file is not found.

    Returns:
        (str): Path to the file.
    """
    check_suffix(file, suffix)  # optional
    file = str(file).strip()  # convert to string and strip spaces
    file = check_yolov5u_filename(file)  # yolov5n -> yolov5nu
    if (
        not file
        or ("://" not in file and Path(file).exists())  # '://' check required in Windows Python<3.10
        or file.lower().startswith("grpc://")
    ):  # file exists or gRPC Triton images
        return file
    elif download and file.lower().startswith(("https://", "http://", "rtsp://", "rtmp://", "tcp://")):  # download
        url = file  # warning: Pathlib turns :// -> :/
        file = Path(download_dir) / url2file(file)  # '%2F' to '/', split https://url.com/file.txt?auth
        if file.exists():
            print(f"Info: Found {clean_url(url)} locally at {file}")  # file already exists
        else:
            downloads.safe_download(url=url, file=file, unzip=False)
        return str(file)
    else:  # search
        files = glob.glob(str(ROOT / "**" / file), recursive=True) or glob.glob(str(ROOT.parent / file))  # find file
        if not files and hard:
            raise FileNotFoundError(f"'{file}' does not exist")
        elif len(files) > 1 and hard:
            raise FileNotFoundError(f"Multiple files match '{file}', specify exact path: {files}")
        return files[0] if len(files) else []  # return file


def check_yaml(file, suffix=(".yaml", ".yml"), hard=True):
    """
    Search/download YAML file (if necessary) and return path, checking suffix.

    Args:
        file (str | Path): File name or path.
        suffix (tuple): Tuple of acceptable YAML file suffixes.
        hard (bool): Whether to raise an error if the file is not found or multiple files are found.

    Returns:
        (str): Path to the YAML file.
    """
    return check_file(file, suffix, hard=hard)
