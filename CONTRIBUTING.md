# Contributing to AIAgentLab

Thank you for your interest in contributing to AIAgentLab.

This project aims to provide a clean, modular foundation for document-grounded question answering using retrieval-augmented generation. Contributions that improve reliability, clarity, developer experience, testing, and documentation are welcome.

## Scope

At the current stage of the project, the most helpful contributions are:

- bug fixes
- test coverage improvements
- retrieval quality improvements
- documentation improvements
- UI and API usability improvements
- cleanup of configuration and project structure

Large feature additions should be discussed in an issue before implementation.

## Before you start

Please do the following before opening a pull request:

1. Search existing issues and pull requests.
2. Open an issue for significant changes.
3. Confirm that the change fits the current project roadmap.
4. Make sure your branch is up to date with the main branch.

## Development setup

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/kahramanfaruk/AIAgentLab.git
cd AIAgentLab
python -m venv .venv
source .venv/bin/activate
```

Create the environment file and install dependencies:

```bash
cp .env.example .env
pip install -e ".[dev]"
```

## Common commands

```bash
make install
make lint
make test
make run
```

## Branch and commit guidance

- Use small, focused pull requests.
- Prefer descriptive branch names such as `fix/retriever-scoring` or `docs/readme-update`.
- Write clear commit messages.
- Do not combine unrelated refactors and features in one pull request.

## Code style

Please follow these guidelines:

- Use Python 3.11+ compatible code.
- Prefer explicit, readable code over clever abstractions.
- Keep functions small and focused.
- Add type hints where practical.
- Avoid introducing unnecessary dependencies.
- Keep business logic out of the UI layer when possible.

Formatting and static checks should pass before submitting a pull request.

## Testing

All contributions should include appropriate testing where relevant.

Expected minimums:

- unit tests for isolated logic changes
- integration tests for end-to-end pipeline changes
- manual smoke testing for Streamlit or API changes

Run tests locally before opening a pull request:

```bash
pytest tests/ -v
```

Or:

```bash
make test
```

## Documentation

If your change affects setup, architecture, usage, or developer workflow, update the relevant documentation as part of the same pull request.

This includes files such as:

- `README.md`
- `docs/architecture/overview.md`
- `docs/architecture/rag_pipeline.md`
- `docs/guides/quickstart.md`

## Pull request checklist

Before submitting a pull request, please verify:

- the code runs locally
- tests pass
- linting passes
- documentation is updated if needed
- no secrets or local data files are committed
- the pull request description clearly explains the change

## Reporting bugs

When reporting a bug, please include:

- a short description of the problem
- steps to reproduce
- expected behavior
- actual behavior
- relevant logs or screenshots
- operating system and Python version

## Suggesting features

Feature suggestions are welcome. Please describe:

- the problem you are trying to solve
- the proposed solution
- why it fits the current project scope
- any tradeoffs or alternatives considered

## Data and security

Please do not commit:

- API keys
- `.env` files
- local vector database contents
- private or copyrighted documents without permission

Use `.env.example` as the template for configuration.

## Questions

If something is unclear, open an issue before making a large change. Early discussion is preferred over large misaligned pull requests.

Thank you for helping improve AIAgentLab.