# PyDJL
PyDJL solves an equation for a wave of permanent form travelling in a stratified fluid. The equation is the Dubreil-Jacotin-Long (DJL) equation and the waves of permanent form are known as internal solitary waves. Such waves travel at constant speed without changing their shape.

PyDJL is a Python 3 translation of the MATLAB-based project [DJLES](https://github.com/mdunphy/DJLES.git). The solution method used here is the same as in DJLES, with some differences in implementation details that lead to better performance. For example, PyDJL uses sine transforms while DJLES uses Fourier transforms. The performance is 6-20x better in PyDJL.

For details on the problem background and setup, please see the user guide and references from the DJLES project: https://www.math.uwaterloo.ca/~mdunphy/djles.pdf

# Requirements
Python 3.7, scipy, numpy, matplotlib.

# Code structure
The core part of code is DJL module. It includes Diagnostic class, DJL class, plot function, and diffmatrix function.
* **DJL class**: This class is the core of project. Implementing different mathematic equations to solve the Dubreil-Jacotin-Long Equation (DJLE)

* **Diagnostic class**: This class use the DJL object, calculating more parameters such as density, vorticity, residual.

* **plot function**: This function is called from case files after the equation has been solved and plot different variables for user.
User can pass plottype 1 or 2. Plottype 1 only plot eta and density, while plottype 2 plots those as well as uwave, w, kewave, ape density,
Ri, and vorticity.

* **diffmatrix function**: This function used both in DJL class and case files. It construct a centred differentiation matrix with up/down wind at the ends

There are 8 case files that can be run to be solved using DJL equation. Each case is explained in pdf provided Under Reference section. 
After the case file completes, the wave is plotted.


# How to use?
Download the code and run a case file, such as:
```
  python case_large_ape.py
```
and you should get a figure like this:![Image](https://raw.githubusercontent.com/golnazir/PyDJL/master/case_large_ape.png)

