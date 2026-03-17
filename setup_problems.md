# DIRECTO — Conda Setup Problems & Solutions

A record of every issue encountered when setting up the DIRECTO environment
via the conda path documented in the README. All problems are specific to the
conda workflow; the pixi path is unaffected because pixi uses different
installation mechanisms and a more permissive TOML parser.

---

## Problem 1 — `- e .` in `environment.yaml` breaks conda env creation

**Reason.** The `pip:` section of `environment.yaml` contained the line `- e .`,
which is meant to install the `directo` package in editable mode. However, conda
translates the pip section of a YAML file into a plain requirements file and hands
it to pip. In a requirements file, `e .` is not valid syntax — the `-e` flag only
works on the command line, not as a package specifier. Conda's pip integration
does not understand this shorthand, so the entire environment creation fails.

**Solution.** Remove the `- e .` line from the `pip:` section of `environment.yaml`
before creating the environment, then run the editable install manually afterward
as a separate step once the environment is active.

```bash
# After removing the line and creating the environment:
conda env create -f environment.yaml
conda activate directo
pip install -e .
```

---

## Problem 2 — `pyproject.toml` uses a multiline inline table, breaking `pip install -e .`

**Reason.** The `authors` field in `pyproject.toml` was written as a multiline
inline table:

```toml
authors = [
    {
        name = "Alba Carballo-Castro, ...",
        email = "alba.carballocastro@epfl.ch"
    }
]
```

The TOML 1.0 specification requires inline tables to be on a single line. Python
3.11's built-in `tomllib` parser strictly follows TOML 1.0 and raises a
`TOMLDecodeError` when it encounters newlines inside the braces. The pixi
workflow is unaffected because pixi uses a more permissive TOML parser that
accepts multiline inline tables as an extension.

**Solution.** Collapse the inline table onto a single line in `pyproject.toml`:

```toml
authors = [
    {name = "Alba Carballo-Castro, Manuel Madeira, Yiming Qin, Dorina Thanou, Pascal Frossard", email = "alba.carballocastro@epfl.ch"}
]
```

---

## Problem 3 — `pip install -e .` succeeds but `import directo` fails

**Reason.** The `pyproject.toml` specifies `[tool.setuptools.packages.find]` with
`where = ["."]`, which tells setuptools to scan the project root for Python
packages. However, the actual source code lives under `src/`, not the root.
Setuptools found no importable package named `directo` in the root directory, so
the editable install registered an empty import hook — pip reported success, but
the hook pointed at nothing.

The correct fix for this project is to recognize that the codebase uses a flat
layout inside `src/` (files like `main.py`, `utils.py` and subdirectories like
`models/` sit directly in `src/` without a `directo/` subdirectory). The intended
usage is always to run `python main.py` from within `src/`, not to `import directo`
from elsewhere. The distribution name `directo` in `pyproject.toml` is a pip label,
not a Python import namespace.

**Solution.** Add a `package-dir` directive to `pyproject.toml` so that setuptools
adds `src/` to the Python path, making all modules inside it importable:

```toml
[tool.setuptools.package-dir]
"" = "src"
```

The practical verification test is not `import directo` but rather whether the
code runs correctly from `src/`:

```bash
cd src && python main.py +experiment=debug
```

---

## Problem 4 — `dgl` not installed despite being listed in `environment.yaml`

**Reason.** DGL is listed in `environment.yaml` under the `dglteam/label/th24_cu121`
channel, which is a niche channel hosting CUDA-version-specific builds. Conda's
solver can silently skip packages from slow or low-priority channels when resolving
the full dependency graph, without raising any error during environment creation.
The omission only surfaces at runtime when the code tries `import dgl`.

**Solution.** Install DGL explicitly via conda from the correct channel after
activating the environment. Using conda rather than pip is important here because
DGL has native CUDA extensions that must match the specific PyTorch + CUDA
combination (PyTorch 2.4.0, CUDA 12.1) used by this project:

```bash
conda install -c dglteam/label/th24_cu121 dgl
```

---

## Problem 5 — `cfg.general.conditional` key missing from Hydra config

**Reason.** The model initialization code in `graph_discrete_flow_model.py` reads
`cfg.general.conditional`, but this key is absent from
`configs/general/general_default.yaml`. Hydra uses OmegaConf in struct mode, which
raises a `ConfigAttributeError` when code accesses a key that was not declared in
the config — it does not silently return `None` as a regular Python dict would.
This is a version skew: the key was added to the code at some point but the
corresponding config entry was never added.

**Solution.** Add the missing key to `configs/general/general_default.yaml`:

```yaml
conditional: False
```

`False` is the correct default because conditional generation (where sampling is
guided by an external label or conditioning signal) is the opt-in mode; all
standard benchmark experiments use unconditional generation.

---

## Problem 6 — `self.spe_out_dim` used before assignment in `GraphTransformerDirected`

**Reason.** In `src/models/transformer_model_directed.py`, the three lines that
assign `self.spe_q_dim`, `self.spe_pe_dim`, and `self.spe_out_dim` were commented
out during a refactor, but `self.spe_out_dim` is still actively referenced two
lines later in the `mlp_in_X` and `mlp_in_E` layer definitions. Python resolves
attribute lookups sequentially at runtime, so accessing `self.spe_out_dim` before
it has been assigned raises an `AttributeError`. Python has no way to catch this at
parse time.

**Solution.** Uncomment the one line that is still needed (the other two can remain
commented if they are not referenced elsewhere):

```python
self.spe_out_dim = spe_dims["out_dim"] if pos_enc == "spe" else 0
```

This evaluates to `spe_dims["out_dim"]` when SPE positional encodings are used,
and to `0` for all other encoding modes (RRWP, magnetic eigenvalues, etc.), leaving
the linear layer dimensions unaffected in those cases.

---

## Problem 7 — NumPy 2.0 removed `np.Inf`, breaking PyTorch Lightning 2.0.4

**Reason.** NumPy 2.0 was a major release that deliberately removed several
long-deprecated aliases, including `np.Inf` (capital I). The correct modern form
is `np.inf`. PyTorch Lightning 2.0.4 — the version pinned in `environment.yaml` —
was released before NumPy 2.0 existed and still uses `np.Inf` internally in its
`ModelCheckpoint` callback. When the environment installs NumPy 2.3.1 (also pinned
in `environment.yaml`), this creates a silent mutual incompatibility that only
surfaces at runtime when the Trainer is initialized. Conda's solver does not detect
this because the conflict is semantic (a removed API) rather than a formal version
constraint.

**Solution.** Downgrade NumPy to the latest 1.x release, which is compatible with
PyTorch Lightning 2.0.4. Downgrading NumPy is safer than upgrading PyTorch
Lightning, because PyTorch Lightning is a higher-level framework whose API has
changed between versions and DIRECTO's code was written against 2.0.4 specifically:

```bash
pip install "numpy<2.0"
```

NumPy 1.26.4 (the latest 1.x release) is extremely well-tested against the
PyTorch ecosystem and all array operations used by DIRECTO exist identically in
both the 1.x and 2.x series.

---

## Summary

All seven problems only affect the conda installation path. The root cause in each
case is that the conda workflow has not been tested end-to-end against the current
state of the codebase — problems 1, 2, and 3 are authoring mistakes in packaging
files; problems 4 and 7 are dependency version incompatibilities that a lockfile
would have prevented; and problems 5 and 6 are code/config synchronization lapses.
The pixi path avoids most of these because pixi uses a lockfile (guaranteeing exact
dependency versions), a more permissive TOML parser, and its own editable install
mechanism that bypasses the `pyproject.toml` issues entirely.
