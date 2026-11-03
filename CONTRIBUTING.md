# Contributing to RadEx

Thank you for considering contributing to RadEx! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a new branch for your feature or bug fix
4. Make your changes
5. Test your changes
6. Commit and push to your fork
7. Submit a pull request

## 📋 Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR-USERNAME/RadEx.git
cd RadEx

# Add upstream remote
git remote add upstream https://github.com/David-Ajay/RadEx.git

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install -e .
pip install pytest flake8 black

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

## 🔧 Code Style

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and concise
- Use type hints where appropriate

### Formatting

We recommend using `black` for code formatting:

```bash
black src/ tests/
```

### Linting

Check your code with `flake8`:

```bash
flake8 src/ tests/ --max-line-length=127
```

## 🧪 Testing

- Write tests for new features
- Ensure existing tests pass
- Aim for high test coverage

Run tests with:

```bash
pytest tests/
```

## 📝 Commit Messages

Write clear, descriptive commit messages:

```
Add feature: Brief description

More detailed explanation if needed.
- Bullet points for multiple changes
- Keep first line under 50 characters
```

## 🎯 Pull Request Process

1. **Update documentation** - Update README.md if needed
2. **Add tests** - Include tests for new functionality
3. **Update CHANGELOG** - Document your changes
4. **Describe your PR** - Clearly explain what and why
5. **Link issues** - Reference related issues

### PR Title Format

```
[Type] Brief description

Types: Feature, Bugfix, Docs, Refactor, Test, Chore
```

## 🐛 Bug Reports

When filing a bug report, include:

- **Description** - Clear description of the bug
- **Steps to reproduce** - How to trigger the bug
- **Expected behavior** - What should happen
- **Actual behavior** - What actually happens
- **Environment** - OS, Python version, dependencies
- **Logs/Screenshots** - Any relevant output

## 💡 Feature Requests

For feature requests, include:

- **Problem description** - What problem does it solve?
- **Proposed solution** - How should it work?
- **Alternatives** - Other solutions considered
- **Additional context** - Any other relevant information

## 📚 Documentation

- Update docstrings for code changes
- Update README.md for user-facing changes
- Add inline comments for complex logic
- Keep documentation accurate and up-to-date

## 🔍 Code Review

All submissions require review. We use GitHub pull requests for this purpose.

Reviewers will check for:
- Code quality and style
- Test coverage
- Documentation
- Performance implications
- Breaking changes

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Thank You!

Your contributions make RadEx better for everyone. We appreciate your time and effort!

## 📧 Questions?

If you have questions, feel free to:
- Open an issue for discussion
- Contact the maintainers
- Check existing documentation

---

**Happy Contributing! 🎉**
