# Data Science Handbook

## Requirements

- [pyenv](https://pypi.org/project/pyenv/)

## Environment Configuration

```sh
pyenv versions                      # Check the installed Python versions
pyenv install --list                # Show the available Python interpreters
pyenv install [version]             # Install a specific Python version
pyenv local [version]               # Set the Python version for the current local directory
pyenv global [version]              # Set the Python version globally on the machine
python3 -m venv .venv               # Create a virtual environment with the current Python version
source .venv/bin/activate           # Activate the virtual environment (Linux/macOS)
.venv\Scripts\activate              # Activate the virtual environment for Windows:
pip install -r requirements.txt     # Install project dependencies
pip3 freeze > requirements.txt      # Copy the python dependencies to requirements.txt
jupyter notebook                    # Run the local Jupiter notebook web server (http://localhost:8888/lab/)
```

## References:

- [Notebooks](https://colab.research.google.com/github/ageron/handson-ml3/blob/main/).
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 3rd Edition](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781098125967/ch02.html)
- [Machine Learning course Layers](https://online.fliphtml5.com/grdgl/hfrm/#p=36)