# PyDJL
This project is translation of of DJLES project [mdunphy/DJLES](https://github.com/mdunphy/DJLES.git) from MATLAB to Python. 

# Motivation
The main motivation of converting the project to python was to speed up solving the DJL equation.

# Code structure
The core part of code is DJL module. It includes Diagnostic class, DJL class, plot function, and diffmatrix function.
* **DJL calss**: This class is the core of project. Implementing different mathematic equations to solve the Dubreil-Jacotin-Long Equation (DJLE)

* **Diagnostic class**: This class use the DJL object, calculating more parameters such as density, vorticity, residual.

* **plot function**: This function is called from case files after the equation has been solved and plot different variables for user.
User can pass plottype 1 or 2. Plottype 1 only plot eta and density, while plottype 2 plots those as well as uwave, w, kewave, ape density,
Ri, and vorticity.

* **diffmatrix function**: This function used both in DJL class and case files. It construct a centered differentiation matrix with up/down wind at the ends

There are 8 case files that can be run to be solved using DJL equation. Each case is explained in pdf provided Under Reference section. 
After the case file completes, the wave is plotted.

# Requirment
python 3 interpreter

# How to use?
Download the code and run one of the case files. User can set the value for different variables as well as verbose variable. Verbose enables a user
to print more information while the program is running.
By default the verbose is set to 0 and limited information is printed as program is running. 

# Reference
To see user guid from original project please see: https://www.math.uwaterloo.ca/~mdunphy/djles.pdf
