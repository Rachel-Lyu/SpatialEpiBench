# SpatialEpiBench
## 1. Project layout
```text
SpatialEpiBench/
├── README.md
├── requirements.txt
├── run_retrain.py
├── models/
│   ├── AGCRN.py
│   ├── DCRNN.py
│   └── ...
└── rawData/
    └── processed/
        ├── JHUcase.csv
        ├── JHUcase_adj.csv
        ├── ILI2019.csv
        ├── ILI2019_adj.csv
        └── ...
```

Each dataset needs two CSV files:

```text
rawData/processed/<dataset>.csv
rawData/processed/<dataset>_adj.csv
```

Example for `--dataset JHUcase`:

```text
rawData/processed/JHUcase.csv
rawData/processed/JHUcase_adj.csv
```

---

## 2. Basic usage

Run with default settings:

```bash
python run_retrain.py --dataset JHUcase
```

By default, this uses:

```text
model      = AGCRN
device     = cpu
epochs     = 50
lookback   = 28
horizon    = 7
train_rate = 0.6
val_rate   = 0.2
loss       = mse
```

Run on GPU:

```bash
python run_retrain.py \
  --dataset JHUcase \
  --model AGCRN \
  --device cuda
```

Run other models (examples):

```bash
python run_retrain.py \
  --dataset JHUcase \
  --model DCRNN \
  --device cuda \
  --rnn-units 32 \
  --num-rnn-layers 2 \
  --max-diffusion-step 2 \
  --dropout 0.1
```

```bash
python run_retrain.py \
  --dataset JHUcase \
  --model STGCN \
  --device cuda \
  --nhids 32
```

```bash
python run_retrain.py \
  --dataset JHUcase \
  --model GraphWaveNet \
  --device cuda \
  --epochs 30 \
  --blocks 2 \
  --nlayers 4 \
  --residual-channels 4 \
  --dilation-channels 4
```

---

## 3. Common command-line arguments

These arguments are available for all models.

| Argument | Default | Description |
|---|---:|---|
| `--dataset` | `JHUcase` | Dataset name under `rawData/processed/`. |
| `--model` | `AGCRN` | Model name to run. |
| `--device` | `cpu` | Device, for example `cpu`, `cuda`, or `cuda:0`. |
| `--epochs` | `50` | Number of training epochs per retraining window. |
| `--lookback` | `28` | Number of historical time steps used as input. |
| `--horizon` | `7` | Forecast horizon used when generating windows. |
| `--train_rate` | `0.6` | Initial train split ratio. |
| `--val_rate` | `0.2` | Initial validation split ratio. |
| `--loss` | `mse` | Loss function. Choices: `mse`, `mse_filtered`. |
| `--retrain-every` | `90` | Number of target time steps predicted before retraining again. |
| `--retrain-train-length` | `180` | Number of previous time steps used for each retraining window. |
| `--use-future-ti` | off | Use future time-index information if supported by the model. |
| `--epi-mode` | `none` | Epidemiological mode. Choices: `none`, `sir_incidence`, `sir_percent`, `ngm`. |
| `--use-einn` | off | Enable EINN alignment. Requires `--epi-mode sir_incidence`, `--epi-mode sir_percent` or `--epi-mode ngm`. |
| `--plot` | off | Plot predictions for one selected state/location. |
| `--state2plot` | `None` | State/location name to plot. Used with `--plot`. |
| `--model-kwargs-json` | `None` | Extra model-specific kwargs as a JSON object. |

---

## 4. Supported models

The runner currently supports:

```text
AGCRN
ARIMA
ColaGNN
DCRNN
Dlinear
EARTH
EpiGNN
GraphWaveNet
GTS
MTGNN
STGCN
STNorm
StemGNN
repeat_last
```

`ARIMA` and `repeat_last` are baseline models.

---

## 5. Model-specific arguments

Only the selected model's hyperparameters are added to the command-line parser.

For example, when you run:

```bash
python run_retrain.py --model GraphWaveNet --help
```

GraphWaveNet-specific arguments will appear.

---

## 6. Output files

The runner creates an output folder named:

```text
retrain_<dataset>/
```

Example:

```text
retrain_JHUcase/
```

The main prediction CSV is saved as:

```text
retrain_<dataset>/retrain_<dataset>_<tag>.csv
```

The CSV contains rows with fields such as:

| Column | Meaning |
|---|---|
| `retrain_id` | Retraining window index. |
| `timestamp` | Target timestamp. |
| `state_idx` | Node/region index. |
| `state` | Node/region name. |
| `train_start` | First timestamp used in the retraining window. |
| `train_end` | Last timestamp used in the retraining window. |
| `eval_start` | First timestamp evaluated after this retrain. |
| `eval_end` | Last timestamp evaluated after this retrain. |
| `y_true` | Ground-truth target value. |
| `y_pred` | Model prediction. |

If `--plot` and `--state2plot` are provided, a PNG file is also saved beside the CSV.

---

## 7. Adding a new model

To add a new model, edit `MODEL_REGISTRY` in `run_retrain.py`.

Example:

```python
MODEL_REGISTRY["NewModel"] = {
    "class_path": "models.NewModel:NewModel",
    "defaults": {
        "hidden_dim": 32,
        "dropout": 0.1,
    },
}
```

Then you can run:

```bash
python run_retrain.py \
  --dataset JHUcase \
  --model NewModel \
  --hidden-dim 64 \
  --dropout 0.2
```

Rules:

- The file should be importable from Python.
- The class path format is `module_path:ClassName`.
- Keys in `defaults` become command-line arguments.
- Underscores in parameter names become hyphens in the CLI.

Example:

```python
"hidden_dim": 32
```

becomes:

```bash
--hidden-dim 32
```