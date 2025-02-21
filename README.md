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
jupyter lab                         # Run the local Jupiter notebook web server (http://localhost:8888/lab/)
```

## Python Help
```sh
help(len)                           # Help of python
pip install ipython                 # Install I Python
ipython                             # Execute the terminal
len?                                # length function help
L = [1, 2, 3]
L.insert?                           # Show the object insert documentation
L?
def square(a):
    """Return the square of a."""
    return a ** 
square?                             # Show the square function documentation
square??                            # Show the implementation of square function
L.<TAB>                             # Show the options and allows to choose it
from itertools import co<TAB>       # Autocomplete the import or allow to choose it
import <TAB>
import h<TAB> 
*Warning?                           # Show and allow to chooice the matches with Warning
str.*find?     
Ctrl-a                              # Move cursor to beginning of line
Ctrl-e                              # Move cursor to end of the line
Ctrl-d                              # Delete next character in line
Ctrl-k                              # Cut text from cursor to end of line
Ctrl-u                              # Cut text from beginning of line to cursor
Ctrl-y                              # Yank (i.e., paste) text that was previously cut
Ctrl-t                              # Transpose (i.e., switch) previous two characters
Ctrl-p (or the up arrow key)        # Access previous command in history
Ctrl-n (or the down arrow key)      # Access next command in history
Ctrl-r                              # Reverse-search through command history
Ctrl-l                              # Clear terminal screen
Ctrl-c                              # Interrupt current Python command
Ctrl-d                              # Ctrl-d

%run myscript.py                                # Run a script and allow to use any defined function in the script.
%timeit L = [n ** 2 for n in range(1000)]       # Which will automatically determine the execution time of the single-line Python statement that follows it.
                                                # => 430 µs ± 3.21 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

%%timeit                                        # for multilpe lines    
    L = []
    for n in range(1000):
    L.append(n ** 2)
                                                # => 484 µs ± 5.67 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
                                                # total time = 484 µs * 1000 loops  = 484000 µs = 0.484 seconds  (10 ^ 6 µs)


from myscript import square_range               # Get the performance of a script
%timeit square_range(1000)

python -m timeit -s "from mi_script import calcular_cuadrados" "calcular_cuadrados(1000)"


```