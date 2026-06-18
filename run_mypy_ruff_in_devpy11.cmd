call conda activate devpy11
mypy src/zernpy
ruff check --fix src tests

set /p dummy=Press Enter to close...
