# PyDJL
PyDJL solves an equation for a wave of permanent form travelling in a stratified fluid. The equation is the Dubreil-Jacotin-Long (DJL) equation and the waves of permanent form are known as internal solitary waves. Such waves travel at constant speed without changing their shape.

PyDJL is a Python 3 translation of the MATLAB-based project [DJLES](https://github.com/mdunphy/DJLES.git). The solution method used here is the same as in DJLES, with some differences in implementation details that lead to better performance. For example, PyDJL uses sine transforms while DJLES uses Fourier transforms. The performance is 6-20x better in PyDJL.

For details on the problem background and setup, please see the user guide and references from the DJLES project: [DJLES documentation](https://palang.ca/DJLES/djles.pdf)

# Requirements
Python 3.7, scipy, numpy, matplotlib.

# Code structure
The code is organized as a single PyDJL module and a set of test cases. The module contains a DJL class, Diagnostic class, plot function, and diffmatrix function.

* **DJL class**: This class is the core of project, which implements and stores the iterative procedure to solve the Dubreil-Jacotin-Long (DJL) equation.

* **Diagnostic class**: This class uses a DJL object to calculate quantities such as density, vorticity, etc.

* **plot function**: This function is called from the test cases to make pseudocolour plots of the solution. User can pass plottype=1 or 2. Plottype 1 shows two panels (eta and density), while plottype 2 plots uses the results of Diagnostics to show 8 panels (eta, density, uwave, w, kewave, ape density, Richardson number, and vorticity).

* **diffmatrix function**: A helper function used both in the DJL class and in the case files. It constructs a centred differentiation matrix with up/down wind at the ends.

The eight test cases demonstrate the problem set up and solution procedure. The cases are explained in more detail in the [DJLES documentation](https://palang.ca/DJLES/djles.pdf)


# How to use?
Download the code and run a case file, such as:
```
  python case_large_ape.py
```
and you should get a figure like this:![Image](https://raw.githubusercontent.com/golnazir/PyDJL/master/case_large_ape.png)

