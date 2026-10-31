# Code Appendix for ResFed: Residual-based Federated Learning Framework

ResFed is a Residual-based Federated Learning Framework with deep residual compression for both up- or/and down-streaming.

In this code appendix, we provide three main modules as follows:
* **Dataset preprocessing**: Dividing the dataset into a number of clients w.r.t. federated settings
* **ResFed implementation**: Pytorch implementation of Residual-based Federated Learning Framework with deep residual compression
* **Postprocessing**: Visualization of the training results and compression ratio for evaluation.

## Framework Overview

![drawing](figures/ResFed_DRC.png)


## Installation

### Dependencies

The current dependency baseline targets **Python 3.11, 64-bit**, with CPU execution on Windows:

| Package | Version |
| --- | --- |
| NumPy | 1.26.4 |
| PyTorch | 2.13.0 |
| Torchvision | 0.28.0 |
| Matplotlib | 3.10.8 |
| tqdm | 4.67.1 |
| SciPy | 1.17.0 |

The original appendix listed Python 3.7 and older packages. NumPy 1.21.x requires Python below 3.11; its old requirement cannot be installed in a Python 3.11 environment. [NumPy 1.26.4 supports Python 3.11](https://numpy.org/doc/stable/release/1.26.4-notes.html), and the selected PyTorch/Torchvision pair has an [official CPU installation](https://pytorch.org/get-started/previous-versions/).

`requirements.txt` contains the common package pins. `requirements-cpu.txt` includes these pins and selects the matching `+cpu` builds from the official PyTorch index. SciPy is included for the SVHN dataset loader. Validate this baseline on your machine using the checks below before starting a longer experiment.

### Install requirements

Run the following commands manually in **PowerShell from the repository root**, one command at a time. Confirm that the first two commands report Python 3.11 and 64-bit before creating the environment:

```powershell
python --version
python -c "import struct, sys; print(sys.executable); print('Architecture:', struct.calcsize('P') * 8)"
```

Create the environment with this command on its own, and wait for it to finish:

```powershell
python -m venv .venv
```

Then install the packages:

```powershell
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: -r requirements-cpu.txt
```

After installation succeeds, check the environment:

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe verify_smoke.py
```

The environment check prints the installed versions and verifies Python 3.11, 64-bit, the package pins, CPU execution, and conversion between PyTorch tensors and NumPy arrays. It does not install packages or download datasets. All subsequent commands use the environment's Python directly, so PowerShell activation is unnecessary.

If `venv` reports unrecognized arguments such as `-m pip` or `--only-binary`, the commands were combined. Run only `python -m venv .venv` first, then enter the pip command separately.

If Python 3.11 is unavailable, resolve that before creating `.venv`. If pip cannot find a matching binary package, check the interpreter version, architecture, and access to PyPI and the PyTorch CPU index before training. The binary-only install deliberately avoids compiling packages from source.

## Federated Dataset Preprocessing

This preprocessing aims to divide the entire datasets into a dedicated number of clients with respect to federated settings.

Depending on the value of `-c` (the number of classes in each local dataset), the entire dataset are split into Non-IID datasets in terms of label distribution skew.

By calling the function `divide_data()`, one of the used datasets in this paper, i.e. MNIST, CIFAR-10, Fashion-MNIST, SVHN, CIFAR-100 is downloaded via Torchvision automatically.

The module has been integrated into the main functions of ResFed.

## Execute the ResFed

### CPU smoke test

To run the environment checks, training, result validation, and headless plotting as one user-invoked task, run the saved script when you are ready:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_smoke.ps1
```

The script uses the existing `.venv` and stops if a step fails. It does not install dependencies. Training can take a while and downloads MNIST on the first run. The individual commands are documented below for running each stage separately.

After the environment checks pass, run this single-line PowerShell command from the repository root:

```powershell
.\.venv\Scripts\python.exe resfed_main.py -n 2 -m 3 -c 10 -mn LeNet -ds MNIST -e 1 -bs 64 -uls 0.95 -dls 0.95 -ul 1 -dl 1 -ulp 1 -dlp 1 -qe 1 -rd results\smoke
```

The first execution downloads MNIST into `data/`. This run uses the full training dataset for three communication rounds; two clients reduce orchestration overhead but do not reduce the total training data. Three rounds exercise upstream compression and downstream recovery after the initial warm-up.

When training completes, validate the newest result and save its plots without opening a window:

```powershell
.\.venv\Scripts\python.exe verify_smoke.py --results results\smoke --output results\smoke.png
```

Success requires two clients, three finite server accuracy/loss measurements, and three communication measurements per client. Accuracy must be within `[0, 1]` and loss nonnegative; this check does not require a particular accuracy or increasing accuracy. The downstream warm-up zero in the second round is expected. The result keys `ul_bits` and `dl_bits` store compression ratios despite their names.

Results have timestamped filenames without a `.json` extension and use the repository's custom decoder. The verifier loads them through `Recorder` and writes a three-panel PNG to `results\smoke.png`. Keep plots outside the input result directory because the evaluation script attempts to load every entry in that directory.

### Longer experiment

The original README experiment uses ten clients and 300 communication rounds:

```powershell
.\.venv\Scripts\python.exe resfed_main.py -n 10 -m 300 -c 10 -lr 0.01 -bs 64 -mn LeNet -ds MNIST -rd results\full -e 1 -is 1 -uls 0.95 -dls 0.95 -ul 1 -dl 1 -qe 1 -ulp 1 -dlp 1 -cw 0
```

Run this after the smoke test passes. The smoke test verifies execution and result handling; reproducing the paper's accuracy requires separate evaluation.

### Explanations of Arguments

- `--num_client` `-n`: Number of the clients.
- `--num_comm_round` `-m`: Number of communication rounds.
- `--num_local_class` `-c`: Number of the classes in each client.
- `--lr` `-lr`: Learning rate.
- `--bs` `-bs`: Batch size.
- `--model_name` `-mn`: Model name, e.g. "LeNet", "CNN", "ResNet18", etc.
- `--dataset` `-ds`: Dataset name: "MNIST", "CIFAR10", "FashionMNIST", "SVHN", or "CIFAR100".
- `--res_dir` `-rd`: Directory for the result files, default directory is "./results".
- `--num_epoch` `-e`: Number of local epochs.
- `--i_seed` `-is`: Seed number.
- `--ul_sparsity` `-uls`: Sparsity for deep compression in up-streaming.
- `--dl_sparsity` `-dls`: Sparsity for deep compression in down-streaming.
- `--resfed_upstream` `-ul`: Flag for ResFed in up-streaming.
- `--resfed_downstream` `-dl`: Flag for ResFed in down-streaming.
- `--quan_coef` `-qe`: Quantization coefficient.
- `--ul_proto_id` `-ulp`: Prediction rule of the predictor for up-streaming.
- `--dl_proto_id` `-dlp`: Prediction rule of the predictor for down-streaming.
- `--comp_weight` `-cw`: Flag for model weight compression.


## Evaluation Procedures

To open the smoke test's accuracy, loss, and compression plots interactively, run:

```powershell
.\.venv\Scripts\python.exe postprocessing\eval_main.py -rr results\smoke
```

Use `-rr results\full` for the longer experiment. Figure labels use the result filenames. Interactive plotting requires a graphical Matplotlib backend; the verifier's PNG output uses the headless `Agg` backend.

# Citation
If you find ResFed is useful, please cite our work:
```bibtex
@ARTICLE{10283999,
  author={Song, Rui and Zhou, Liguo and Lyu, Lingjuan and Festag, Andreas and Knoll, Alois},
  journal={IEEE Internet of Things Journal}, 
  title={ResFed: Communication-Efficient Federated Learning With Deep Compressed Residuals}, 
  year={2024},
  volume={11},
  number={6},
  pages={9458-9472},
  doi={10.1109/JIOT.2023.3324079}}
```
