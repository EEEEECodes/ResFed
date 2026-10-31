"""Check the Python 3.11 CPU environment and an optional MNIST smoke result."""

import argparse
import importlib
from importlib import metadata
from pathlib import Path
import re
import struct
import sys


PACKAGES = ("numpy", "torch", "torchvision", "matplotlib", "tqdm", "scipy")
INSTALL_COMMAND = (
    r".\.venv\Scripts\python.exe -m pip install --only-binary=:all: "
    "-r requirements-cpu.txt"
)


class VerificationError(Exception):
    """A failed environment or saved-result check."""


def check_environment():
    if sys.version_info[:2] != (3, 11) or struct.calcsize("P") != 8:
        raise VerificationError(
            "Use 64-bit Python 3.11 to create .venv, then run this helper with "
            r".\.venv\Scripts\python.exe. "
            f"Current Python: {sys.version.split()[0]} ({struct.calcsize('P') * 8}-bit)."
        )

    requirements = Path(__file__).with_name("requirements.txt")
    pins = {}
    for line in requirements.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([\w-]+)==([^\s;]+)", line.strip())
        if match:
            pins[match[1]] = match[2]
    if any(package not in pins for package in PACKAGES):
        raise VerificationError("requirements.txt must pin all six smoke dependencies with ==.")

    modules = {}
    errors = []
    for package in PACKAGES:
        try:
            version = metadata.version(package)
            print(f"{package}: {version}")
            if version.split("+", 1)[0] != pins[package]:
                errors.append(f"{package} must be {pins[package]}, found {version}")
            modules[package] = importlib.import_module(package)
            if package == "matplotlib":
                modules[package].use("Agg", force=True)
            if package in ("torch", "torchvision") and "+" in version:
                if version.split("+", 1)[1] != "cpu":
                    errors.append(f"{package} must use a CPU wheel, found {version}")
        except metadata.PackageNotFoundError:
            errors.append(f"{package} is not installed")
        except Exception as exc:
            errors.append(f"cannot import {package}: {exc}")
    if errors:
        raise VerificationError(
            "Dependency check failed:\n- " + "\n- ".join(errors)
            + f"\nInstall the pinned CPU dependencies manually from the repository: {INSTALL_COMMAND}"
        )

    np, torch = modules["numpy"], modules["torch"]
    if torch.version.cuda is not None or torch.cuda.is_available():
        raise VerificationError(
            "A CPU-only PyTorch build is required. Reinstall the CPU requirements manually: "
            + INSTALL_COMMAND
        )
    try:
        tensor = torch.tensor([1.0, 2.0, 3.0], device="cpu")
        array = tensor.numpy()
        restored = torch.from_numpy(array)
        if not np.array_equal(array, [1.0, 2.0, 3.0]) or not torch.equal(restored, tensor):
            raise VerificationError("Torch/NumPy round trip changed the tensor values.")
        if restored.device.type != "cpu":
            raise VerificationError("Torch/NumPy round trip did not stay on CPU.")
    except VerificationError:
        raise
    except Exception as exc:
        raise VerificationError(f"Torch/NumPy round trip failed: {exc}") from exc
    print(f"Environment verified: Python {sys.version.split()[0]} (64-bit), CPU, Torch/NumPy round trip.")
    return np


def check_results(result_dir, output, np):
    result_dir = result_dir.resolve()
    output = output.resolve()
    if output.is_relative_to(result_dir):
        raise VerificationError("Save the plot outside the result-input directory, e.g. results/smoke.png.")
    if not result_dir.is_dir():
        raise VerificationError(f"Result directory does not exist: {result_dir}. Run the smoke experiment first.")
    files = [path for path in result_dir.iterdir() if path.is_file()]
    if not files:
        raise VerificationError(f"No result files found in {result_dir}. Run the smoke experiment first.")
    result_file = max(files, key=lambda path: (path.stat().st_mtime_ns, path.name))

    # Recorder.load uses the repository's existing decoder for NumPy scalar values.
    import matplotlib.pyplot as plt
    from postprocessing.recorder import Recorder

    recorder = Recorder()
    try:
        recorder.load(result_file, label=result_file.name)
    except Exception as exc:
        raise VerificationError(f"Cannot load result {result_file}: {exc}") from exc
    result = recorder.res_list[0][0]
    if not isinstance(result, dict):
        raise VerificationError(f"Expected a result object in {result_file}.")
    expected = {
        "dataset_id": "MNIST", "model_name": "LeNet", "fed_method": "ResFed",
        "num_client": 2, "num_comm_round": 3, "num_epoch": 1,
        "num_local_class": 10, "bs": 64, "lr": 0.01, "i_seed": 1,
        "ul_spar": 0.95, "dl_spar": 0.95, "resfed_upstream": 1,
        "resfed_downstream": 1, "ul_proto_id": 1, "dl_proto_id": 1,
        "qe": 1, "Param_01": 0,
    }
    hyperparam = result.get("hyperparam", {})
    if not isinstance(hyperparam, dict):
        raise VerificationError("Result hyperparam must be an object.")
    for key, value in expected.items():
        if hyperparam.get(key) != value:
            raise VerificationError(f"Expected {key}={value!r}; found {hyperparam.get(key)!r} in {result_file}.")

    def series(section, key, label):
        try:
            values = np.asarray(section[key], dtype=float)
        except (KeyError, TypeError, ValueError) as exc:
            raise VerificationError(f"Missing or invalid {label}.{key}.") from exc
        if values.shape != (3,) or not np.isfinite(values).all():
            raise VerificationError(f"{label}.{key} must contain exactly three finite values.")
        return values

    server = result.get("server", {})
    accuracy = series(server, "iid_accuracy", "server")
    loss = series(server, "train_loss", "server")
    if not ((accuracy >= 0) & (accuracy <= 1)).all() or not (loss >= 0).all():
        raise VerificationError("Server accuracy must be in [0, 1] and training loss must be nonnegative.")
    clients = result.get("clients", {})
    if not isinstance(clients, dict) or len(clients) != 2:
        raise VerificationError("The smoke result must contain exactly two clients.")
    for client_id, client in clients.items():
        upstream = series(client, "ul_bits", str(client_id))
        downstream = series(client, "dl_bits", str(client_id))
        if upstream[0] != 1 or not (upstream > 0).all():
            raise VerificationError(f"{client_id}: expected upstream ratios [1, positive, positive].")
        if downstream[0] != 1 or downstream[1] != 0 or downstream[2] <= 0:
            raise VerificationError(f"{client_id}: expected downstream ratios [1, 0, positive] (warm-up zero).")

    figure = None
    try:
        recorder.plot()
        figure = plt.gcf()
        if len(figure.axes) != 3 or [len(axis.lines) for axis in figure.axes] != [1, 1, 4]:
            raise VerificationError("Expected three plot axes: accuracy, loss, and both client communication directions.")
        for axis in figure.axes:
            for line in axis.lines:
                if len(line.get_xdata()) != 3 or len(line.get_ydata()) != 3:
                    raise VerificationError("Every plotted series must have exactly three rounds.")
        if not np.array_equal(figure.axes[0].lines[0].get_ydata(), accuracy):
            raise VerificationError("Plotted accuracy does not match the saved result.")
        if not np.array_equal(figure.axes[1].lines[0].get_ydata(), loss):
            raise VerificationError("Plotted loss does not match the saved result.")
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="png", bbox_inches="tight")
    finally:
        if figure is not None:
            plt.close(figure)
    if not output.is_file() or output.stat().st_size == 0:
        raise VerificationError(f"Plot was not saved successfully: {output}")
    print(f"Result verified: {result_file} (MNIST/LeNet, 2 clients, 3 rounds, finite metrics).")
    print(f"Plot saved: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, help="Validate the newest regular result file in this directory.")
    parser.add_argument("--output", type=Path, help="PNG path outside --results (default: results/smoke.png).")
    args = parser.parse_args()
    if args.output is not None and args.results is None:
        parser.error("--output requires --results")
    try:
        np = check_environment()
        if args.results is not None:
            check_results(args.results, args.output or Path("results/smoke.png"), np)
    except (VerificationError, OSError) as exc:
        print(f"Verification failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
