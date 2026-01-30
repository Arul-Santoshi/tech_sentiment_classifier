# CLAUDE.md - AI Assistant Guide for tech_sentiment_classifier

This document provides essential context and guidelines for AI assistants working on this codebase.

## Project Overview

**tech_sentiment_classifier** is a Python-based machine learning project for sentiment analysis of technology-related content. The project classifies text (articles, reviews, social media posts, etc.) into sentiment categories (positive, negative, neutral) with a focus on the technology domain.

**Author**: Arul Santoshi
**License**: MIT (2026)
**Status**: Initial development phase

## Repository Structure

```
tech_sentiment_classifier/
├── .gitignore          # Comprehensive Python/ML gitignore
├── LICENSE             # MIT License
├── README.md           # Project documentation
├── CLAUDE.md           # This file - AI assistant guidelines
│
# Recommended structure for development:
├── src/                # Source code
│   └── tech_sentiment_classifier/
│       ├── __init__.py
│       ├── models/     # ML model definitions
│       ├── data/       # Data processing utilities
│       ├── training/   # Training scripts and configs
│       └── inference/  # Prediction/inference code
├── tests/              # Test suite
├── notebooks/          # Jupyter notebooks for exploration
├── data/               # Dataset storage (gitignored)
├── models/             # Trained model artifacts (gitignored)
├── configs/            # Configuration files
└── scripts/            # Utility scripts
```

## Development Environment

### Python Environment Setup

This project uses Python with virtual environment management. Supported tools (based on .gitignore):
- **venv** (standard library)
- **poetry** (recommended for dependency management)
- **pipenv**
- **uv** (fast Python package installer)
- **pdm**
- **pixi**

### Recommended Setup Commands

```bash
# Using venv
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
pip install -e ".[dev]"

# Using poetry
poetry install
poetry shell

# Using uv
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### IDE Support

The project is configured for:
- **VS Code** - Primary editor support
- **PyCharm** - JetBrains IDE support
- **Cursor** - AI-powered editor support

## Key Conventions

### Code Style

- Follow **PEP 8** for Python code style
- Use **type hints** for all function signatures
- Use **Ruff** for linting and formatting (indicated by `.ruff_cache/` in gitignore)
- Maximum line length: 88 characters (Black/Ruff default)

### File Naming

- Python modules: `snake_case.py`
- Test files: `test_<module_name>.py`
- Configuration files: `<name>.yaml` or `<name>.toml`
- Notebooks: `<number>_<descriptive_name>.ipynb`

### Import Order

1. Standard library imports
2. Third-party imports
3. Local application imports

Use `isort` or Ruff's import sorting.

### Docstrings

Use Google-style docstrings:

```python
def classify_sentiment(text: str, model: str = "default") -> dict:
    """Classify the sentiment of the given text.

    Args:
        text: The input text to classify.
        model: The model variant to use for classification.

    Returns:
        A dictionary containing sentiment label and confidence score.

    Raises:
        ValueError: If text is empty or model is not found.
    """
```

## Testing

### Test Framework

Use **pytest** for testing (indicated by `.pytest_cache/` in gitignore).

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/tech_sentiment_classifier

# Run specific test file
pytest tests/test_models.py

# Run tests matching pattern
pytest -k "test_sentiment"
```

### Test Organization

```
tests/
├── conftest.py         # Shared fixtures
├── test_models.py      # Model tests
├── test_data.py        # Data processing tests
├── test_training.py    # Training pipeline tests
└── test_inference.py   # Inference tests
```

### Coverage Requirements

- Aim for >80% code coverage
- Use `coverage.xml` for CI integration

## Type Checking

Use **mypy** for static type checking (indicated by `.mypy_cache/` in gitignore).

```bash
mypy src/
```

## Common Commands

```bash
# Linting
ruff check src/ tests/
ruff format src/ tests/

# Type checking
mypy src/

# Testing
pytest
pytest --cov=src/tech_sentiment_classifier --cov-report=html

# Documentation (if Sphinx is set up)
cd docs && make html
```

## Git Workflow

### Branch Naming

- `main` - Production-ready code
- `develop` - Integration branch
- `feature/<name>` - New features
- `fix/<name>` - Bug fixes
- `claude/<session-id>` - AI assistant development branches

### Commit Messages

Use conventional commits format:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Examples:
- `feat(model): add transformer-based sentiment classifier`
- `fix(data): handle empty text inputs gracefully`
- `docs: update API documentation`
- `test(inference): add edge case tests for batch prediction`

## AI Assistant Guidelines

### When Working on This Codebase

1. **Read before editing**: Always read existing files before making modifications
2. **Preserve style**: Match the existing code style and conventions
3. **Add tests**: Include tests for new functionality
4. **Type hints**: Add type annotations to all new code
5. **Documentation**: Update docstrings and README as needed

### Avoid These Patterns

- Don't create unnecessary abstraction layers
- Don't add features beyond what's requested
- Don't modify unrelated code during bug fixes
- Don't commit sensitive data (API keys, credentials)
- Don't ignore existing error handling patterns

### ML-Specific Guidelines

1. **Reproducibility**: Always set random seeds for reproducible results
2. **Model versioning**: Use clear versioning for trained models
3. **Data handling**: Never commit raw datasets to the repository
4. **Experiment tracking**: Document hyperparameters and results
5. **Resource awareness**: Be mindful of memory and compute requirements

### File Operations

- Prefer editing existing files over creating new ones
- Use the recommended project structure when creating new modules
- Keep configuration separate from code
- Store large files (models, data) outside the repository

## Dependencies (Expected)

Based on project type, common dependencies may include:

### Core ML/NLP
- `torch` or `tensorflow` - Deep learning framework
- `transformers` - Hugging Face transformers
- `scikit-learn` - Classical ML algorithms
- `numpy`, `pandas` - Data manipulation

### NLP-Specific
- `nltk` or `spacy` - NLP utilities
- `tokenizers` - Fast tokenization

### Development
- `pytest` - Testing
- `ruff` - Linting and formatting
- `mypy` - Type checking
- `pre-commit` - Git hooks

### Optional
- `jupyter` - Notebooks
- `mlflow` or `wandb` - Experiment tracking
- `dvc` - Data version control

## Environment Variables

Store sensitive configuration in `.env` (gitignored):

```bash
# .env.example
MODEL_PATH=/path/to/models
DATA_PATH=/path/to/data
API_KEY=your_api_key_here
LOG_LEVEL=INFO
```

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure the package is installed in editable mode (`pip install -e .`)
2. **CUDA issues**: Check GPU availability with `torch.cuda.is_available()`
3. **Memory errors**: Reduce batch size or use gradient accumulation
4. **Missing dependencies**: Run `pip install -e ".[dev]"` for all dependencies

### Getting Help

- Check existing documentation in `docs/`
- Review test files for usage examples
- Examine notebooks for exploration patterns

---

*Last updated: 2026-01-30*
