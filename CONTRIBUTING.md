# Contributing to Ultra Fast Backtester

Thank you for considering contributing to Ultra Fast Backtester! This project is in active development and we welcome contributions from the community.

## Development Setup

1. **Fork the repository**
   ```bash
   # Fork on GitHub first, then clone your fork
   git clone https://github.com/yourusername/ultra-fast-backtester.git
   cd ultra-fast-backtester
   ```

2. **Install in development mode**
   ```bash
   pip install -e ".[dev,ml,viz]"
   ```

3. **Run tests to ensure everything works**
   ```bash
   pytest
   ```

## Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes and add tests**
   - Follow the existing code style
   - Add tests for new functionality

3. **Run tests to ensure they pass**
   ```bash
   pytest
   ```

4. **Check code quality**
   ```bash
   black .  # Format code
   flake8   # Check for style issues
   ```

5. **Commit your changes**
   ```bash
   git commit -m "Add feature: brief description"
   ```

6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Submit a pull request**
   - Provide a clear description of the changes
   - Reference any related issues

## Code Style Guidelines

- Follow PEP 8 for Python code style
- Keep functions focused and reasonably sized
- Use meaningful variable and function names

## Testing Guidelines

- Write tests for all new functionality
- Use descriptive test names
- Test both success and failure cases

## Reporting Issues

When reporting issues, please include:

- Python version
- Operating system
- Steps to reproduce the issue
- Expected vs actual behavior
- Any error messages or stack traces

## Feature Requests

For feature requests, please:

- Describe the feature clearly
- Explain the use case
- Consider if it fits the project's scope
- Be open to discussion and iteration

## Questions and Discussion

- Use GitHub Issues for bug reports and feature requests
- Use GitHub Discussions for general questions and ideas
- Feel free to reach out via email: adi.siv@berkeley.edu

## License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing to Ultra Fast Backtester! 