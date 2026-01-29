# Installation

## Requirements

- Python 3.8+
- NumPy ≥ 1.20
- SciPy ≥ 1.7

## Install from PyPI

```bash
pip install symbiont
```

## Install from Source

```bash
git clone https://github.com/yourusername/symbiont.git
cd symbiont
pip install -e .
```

## Development Installation

For running tests and contributing:

```bash
git clone https://github.com/yourusername/symbiont.git
cd symbiont
pip install -e ".[dev]"
```

This installs additional dependencies:
- pytest, pytest-cov (testing)
- black, flake8 (code formatting)
- mypy (type checking)

## Verify Installation

```python
import symbiont
print(symbiont.__version__)

# Quick functionality check
from symbiont import SMIS, ECTI, ECCP, TEFL
from symbiont.core import theorems, constants

print("SYMBIONT installed successfully!")
```

## Optional Dependencies

For visualization (optional):
```bash
pip install matplotlib seaborn
```

For advanced optimization (optional):
```bash
pip install cvxpy
```

## Platform Notes

### Linux
Full support, recommended platform.

### macOS
Full support on both Intel and Apple Silicon.

### Windows
Full support. Use Windows Terminal or PowerShell for best experience.

## Troubleshooting

### NumPy Version Conflicts
If you encounter NumPy-related errors:
```bash
pip install --upgrade numpy>=1.20
```

### SciPy Installation Issues
On some systems, SciPy may require additional dependencies:
```bash
# Ubuntu/Debian
sudo apt-get install libopenblas-dev liblapack-dev

# macOS
brew install openblas
```

## Docker

A Dockerfile is provided for containerized deployment:

```bash
docker build -t symbiont .
docker run -it symbiont python -c "from symbiont import SMIS; print('OK')"
```
