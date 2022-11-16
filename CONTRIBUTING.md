# How to Contribute

We'd love to accept your patches and contributions to this project as it will take the joint efforts of the MARL community to ensure that the evaluation standard is raised! There are
just a few small guidelines you need to follow.

## Installing MARL-eval for developement

To develop features for marl-eval, clone the repository and install all the dependencies as follows:

```bash
git clone https://github.com/instadeepai/marl-eval.git
pip install -e .
```

## Installing Pre-Commit Hooks and Testing Dependencies

Install the pre-commit hooks and testing dependencies:
```bash
pip install .[testing_formatting]
pre-commit install
pre-commit install -t commit-msg
```
You can run all the pre-commit hooks on all files as follows:
```bash
pre-commit run --all-files
```

## Naming Conventions
### Branch Names
We name our feature and bugfix branches as follows - `feature/[BRANCH-NAME]`, `bugfix/[BRANCH-NAME]` or `maintenance/[BRANCH-NAME]`. Please ensure `[BRANCH-NAME]` is hyphen delimited.
### Commit Messages
We follow the conventional commits [standard](https://www.conventionalcommits.org/en/v1.0.0/).

## Code reviews

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.

## Community Guidelines

This project follows
[Google's Open Source Community Guidelines](https://opensource.google.com/conduct/).
