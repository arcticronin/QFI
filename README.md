# QFI

## Structure

Important files:

firt to look at:
- notebook.ipynb : jupyter notebook that is used to debug the process, uses everything that has been used in main

then:
- main.py : main script, that computes QFI using numpy
- density_matrix_from_exp : generation of the density matrices 

will be part of a future work:
qiskit_subroutines and sub2 : quantum circuits to get the fisher info (annex of the paper)
vqse.py : variational quantum eigensolver, that is also used in subroutine (must be fixed)

even later:
- optimization best alpha to get the U that optimizes the fisher info