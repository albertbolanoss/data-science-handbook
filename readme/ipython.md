# Ipython


## Ipython Help commands

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
In                                  # show the typed commands in ipython
Out                                 # show the output commands in ipython
Out[2] ** 2 + Out[3] ** 2           # get the output in the position 2 and operate
print(_)                            # the _ get the latest value of output      
print(__)                           # —you can use a double underscore to access the second-to-last output
print(___)                          # and a triple underscore to access the third-to-last output
_2                                  # it's the same that access to Out[2], the expresion it's _N (number)
 math.sin(2) + math.cos(2);         # Suppressing Output, add semicolon to the line
 14 in Out                          # Check if 14 is in Out
 %history -n 1-3                    # print the last four histort commands
```

## Ipython shortcuts

```sh
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

```

## Magic commands
```sh
%run myscript.py                                # Run a script and allow to use any defined function in the script.
%timeit L = [n ** 2 for n in range(1000)]       # Which will automatically determine the execution time                                            # of the single-line Python statement that follows it.
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

%magic                                          # show a completed list of magic commands (magic command are special commands that starts with % or %%)
%timeit?                                        # show information of a magic command

```

