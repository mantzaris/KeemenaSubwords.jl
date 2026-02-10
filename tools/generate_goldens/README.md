# Golden Generator (Maintainer-Only)

This directory contains optional Python tooling for regenerating conformance goldens.

- Runtime tests do **not** use Python.
- CI tests consume committed JSON files under `test/golden/`.

## Usage

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python generate_goldens.py --out ../../test/golden
```

The generated files are meant to be reviewed and committed by maintainers.
