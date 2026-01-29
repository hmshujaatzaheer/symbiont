# Contributing to SYMBIONT

Thank you for your interest in contributing to SYMBIONT! This document provides guidelines for contributing.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/symbiont.git
   cd symbiont
   ```
3. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```
4. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=symbiont --cov-report=html

# Run specific test file
pytest tests/test_smis.py -v

# Run specific test
pytest tests/test_theorems.py::TestTheorem1MutualInformation -v
```

### Code Style

We use standard Python style conventions:

```bash
# Format code
black symbiont tests

# Check linting
flake8 symbiont tests

# Type checking
mypy symbiont
```

### Pre-commit Checks

Before committing, ensure:
1. All tests pass: `pytest`
2. Code is formatted: `black --check symbiont tests`
3. No lint errors: `flake8 symbiont tests`

## Contribution Types

### Bug Fixes

1. Create an issue describing the bug
2. Reference the issue in your PR
3. Include a test that reproduces the bug
4. Fix the bug and verify the test passes

### New Features

1. Open an issue to discuss the feature
2. Get feedback before implementing
3. Include comprehensive tests
4. Update documentation

### Documentation

- Fix typos or clarify explanations
- Add examples
- Improve API documentation

### Theorems

When adding or modifying theorems:

1. Ensure mathematical correctness
2. Add comprehensive tests including:
   - Basic functionality
   - Edge cases
   - Input validation
3. Update `docs/theorems.md`
4. Reference the paper section/equation

## Code Guidelines

### Module Structure

Each algorithm module (`smis`, `ecti`, `eccp`, `tefl`) follows this pattern:

```python
# symbiont/module/__init__.py

"""
Module docstring with paper reference.
"""

from typing import ...
import numpy as np

from ..core.constants import ...
from ..core.theorems import ...

# Public API
__all__ = ['MainClass', 'HelperClass', 'Result']

class MainClass:
    """
    Main algorithm implementation.
    
    From Section X.Y of the paper:
    [Brief description of the algorithm]
    
    References:
        - Author (Year): Paper title
    """
    
    def __init__(self, ...):
        """Initialize with clear parameter documentation."""
        pass
    
    def main_method(self, ...) -> Result:
        """
        Main functionality.
        
        Args:
            param: Description
        
        Returns:
            Description of return value
        
        Raises:
            ValueError: When input is invalid
        """
        pass
```

### Testing Guidelines

```python
# tests/test_module.py

import pytest
import numpy as np
from symbiont.module import MainClass

class TestMainClass:
    """Tests for MainClass."""
    
    @pytest.fixture
    def instance(self):
        """Create test instance."""
        return MainClass(...)
    
    def test_initialization(self, instance):
        """Test proper initialization."""
        assert instance is not None
    
    def test_main_functionality(self, instance):
        """Test core functionality."""
        result = instance.main_method(...)
        assert result.value > 0
    
    def test_edge_case(self, instance):
        """Test edge case handling."""
        with pytest.raises(ValueError):
            instance.main_method(invalid_input)
```

### Docstring Format

Use Google-style docstrings:

```python
def function(param1: float, param2: np.ndarray) -> Tuple[float, float]:
    """
    Brief description.
    
    Longer description if needed, including mathematical
    formulation or paper reference.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        Tuple containing:
            - First element description
            - Second element description
    
    Raises:
        ValueError: When param1 is negative
    
    Example:
        >>> result = function(1.0, np.array([1, 2, 3]))
        >>> print(result)
        (0.5, 0.3)
    """
    pass
```

## Pull Request Process

1. **Update documentation** for any changed functionality
2. **Add tests** for new features
3. **Ensure CI passes** (all tests, linting, type checks)
4. **Request review** from maintainers
5. **Address feedback** promptly

### PR Title Format

```
[TYPE] Brief description

Types:
- feat: New feature
- fix: Bug fix
- docs: Documentation only
- test: Tests only
- refactor: Code refactoring
- perf: Performance improvement
```

### PR Description Template

```markdown
## Description
Brief description of changes.

## Related Issue
Fixes #123

## Changes
- Change 1
- Change 2

## Testing
How was this tested?

## Checklist
- [ ] Tests pass locally
- [ ] Documentation updated
- [ ] Code formatted with black
- [ ] No new linting errors
```

## Questions?

- Open an issue for questions
- Tag maintainers for urgent matters
- Check existing issues before creating new ones

Thank you for contributing to SYMBIONT!
