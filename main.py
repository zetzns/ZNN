import numpy as np
import scipy
import sympy as sp
import matplotlib.pyplot as plt
from sympy.parsing.sympy_parser import parse_expr


# Check if a matrix is symmetric
def is_symmetric(matrix) -> bool:
    matrix_ = np.array(matrix)
    for i in range(n):
        for j in range(i,n):
            if matrix_[i][j] != matrix_[j][i]:
                return False
    return True


# Compute the operational matrix Z used in transforming vector operations
def compute_operational_matrix_Z(n: int) -> np.array:
    n_sq = n*n
    r = (n_sq + n)//2
    Z = np.zeros((n_sq,r))
    for h in range (1, n_sq+1):
        d = (h-1)%n + 1
        c = int(np.floor((h-1)/n) + 1)
        if c >= d:
            Z[h-1][d+c*(c-1)//2-1] = 1
        else:
            Z[h-1][c+d*(d-1)//2-1] = 1
    return Z


# Calculate the H matrix which is part of the Riccati differential equation
def H_matrix(S, R, Q, Y):
    return S.transpose()@Y + Y@S - Y@R@Y + Q


# Compute the mass matrix used in the ZNN transformation
def mass_matrix(S, Y, R, Z):
    I = np.eye(n)
    return (np.kron(I,S.transpose()) + np.kron(S.transpose(),I)-np.kron(I,Y@R)-np.kron((R@Y).transpose(),I))@Z


# Calculate w matrix for the neural network computation
def w_matrix(H, S, Y, R, Q, a=10):
    I = np.eye(n)
    S = S.diff()
    R = R.diff()
    Q = Q.diff()
    return -a*np.ravel(H) - (np.kron(I,S.transpose()))@np.ravel(Y) - (np.kron(I,Y))@np.ravel(S) + (np.kron(Y.transpose(),Y))@np.ravel(R) - np.ravel(Q)

# Differential function for integration
def y_prime(t, y, M, H, w, error_arr):
    for i in range(n*(n+1)//2):
        M = M.subs(y_arr[i], y[i])
        H = H.subs(y_arr[i], y[i])
        w = w.subs(y_arr[i], y[i])

    M_sub = np.array(M.subs('t', t), dtype=np.float64)
    w_sub = np.array(w.subs('t', t), dtype=np.float64)
    H_sub = np.array(H.subs('t', t), dtype=np.float64)
    error_arr.append((np.linalg.norm(H_sub, ord='fro'), t))
    if n == 1:
        return w_sub/M_sub
    else:
        return np.linalg.inv((M_sub.T)@M_sub)@(M_sub.T)@w_sub


# Input gathering from user
def get_input() -> None:
    global a, n, S, R, Q, Y0, t_start, t_end, num_eval
    n = int(input('Введите размерность задачи '))
    if (n <= 0):
        raise ValueError("Некорректный ввод")
    a = float(input('Введите коэффициент скорости обучения '))
    if (a <= 0):
        raise ValueError("Некорректный ввод")
    S = sp.Matrix(np.zeros((n, n)))
    R = sp.Matrix(np.zeros((n, n)))
    Q = sp.Matrix(np.zeros((n, n)))
    Y0 = np.zeros((n, n))

    n_elems = n*n

    for i in range(n_elems):
        element = input(f'Введите {i+1} элемент матрицы S ')
        S[i] = parse_expr(element)

    for i in range(n_elems):
        element = input(f'Введите {i+1} элемент матрицы R ')
        R[i] = parse_expr(element)
    if not (is_symmetric(R)):
        raise ValueError("Некорректный ввод")

    for i in range(n_elems):
        element = input(f'Введите {i+1} элемент матрицы Q ')
        Q[i] = parse_expr(element)
    if not (is_symmetric(Q)):
        raise ValueError("Некорректный ввод")

    # S = sp.Matrix([
    #     [5 - 2 * sp.cos(t), -2 + sp.cos(t)],
    #     [4 + sp.sin(2 * t), -20 + 5 * sp.sin(t)]
    # ])
    # R = sp.Matrix([
    #     [2 + sp.sin(3 * t), 1 + sp.cos(3 * t)],
    #     [1 + sp.cos(3 * t), 6 - sp.sin(3 * t)]
    # ])
    # Q = sp.Matrix([
    #     [9 + 3 * sp.sin(t), -2 - sp.sin(t)],
    #     [-2 - sp.sin(t), 3 + 2 * sp.cos(3 * t)]
    # ])
    for i in range(n_elems):
        element = float(input(f'Введите {i+1} элемент матрицы Y0 '))
        Y0[i//n][i%n] = element
    if not (is_symmetric(Y0)):
        raise ValueError("Некорректный ввод")

    ts = input('Введите начало, конец, количество точек вычисления в формате t_start, t_end, num_eval ')
    ts = ts.split(', ')
    t_start = float(ts[0])
    t_end = float(ts[1])
    num_eval = int(ts[2])


# Input gathering from user
def get_input_from_file(filename) -> None:
    global a, n, S, R, Q, Y0, t_start, t_end, num_eval
    with open(filename, 'r') as f:
        n = int(f.readline())
        if (n <= 0):
            raise ValueError("Некорректный ввод")
        a = float(f.readline())
        if (a <= 0):
            raise ValueError("Некорректный ввод")
        S = sp.Matrix(np.zeros((n, n)))
        R = sp.Matrix(np.zeros((n, n)))
        Q = sp.Matrix(np.zeros((n, n)))
        Y0 = np.zeros((n, n))

        n_elems = n*n

        for i in range(n_elems):
            element = f.readline()
            S[i] = parse_expr(element)

        for i in range(n_elems):
            element = f.readline()
            R[i] = parse_expr(element)
        if not (is_symmetric(R)):
            raise ValueError("Некорректный ввод")

        for i in range(n_elems):
            element = f.readline()
            Q[i] = parse_expr(element)
        if not (is_symmetric(Q)):
            raise ValueError("Некорректный ввод")


        for i in range(n_elems):
            element = float(f.readline())
            Y0[i//n][i%n] = element
        if not (is_symmetric(Y0)):
            raise ValueError("Некорректный ввод")

        ts = f.readline()
        ts = ts.split(', ')
        t_start = float(ts[0])
        t_end = float(ts[1])
        num_eval = int(ts[2])

# Main function to initialize and run the neural network solver
def main() -> None:
    """
    main func to realize ZNNTV-ARE
    :return:
    """
    try:
        global t
        t = sp.symbols('t')                                    # Define the symbolic variable for time used in sympy expressions
        ReadType = input('Введите тип считывания: консоль/файл: ') # Request input type from user
        if ReadType == 'консоль':
            get_input()
        elif ReadType == 'файл':
            FileName = input('Введите название файла: ')
            get_input_from_file(FileName)
        else:
            raise ValueError("Некорректный ввод")

    except ValueError:
        raise ValueError("Некорректный ввод")

    global y_arr
    y_arr = [sp.Function(f'y{i}')(t) for i in range(0, (n*n+n)//2)]  # Define symbolic functions for each matrix element as a function of time
    Y = np.eye(n, dtype=object)                             # Create an upper triangular matrix of symbolic variables (this includes diagonal)
    countr = 0
    for i in range(n):
        for j in range(i, n):
            Y[i][j] = y_arr[countr]
            Y[j][i] = y_arr[countr]
            countr += 1

    Y0_vec = []                                             # Flatten the initial condition matrix Y0 into a vector for integration
    for i in range(n):
        for j in range(i, n):
            Y0_vec.append(Y0[i][j])

    Z = compute_operational_matrix_Z(n)                 # Calculate the operational matrix Z which maps flattened Y to the differential equations
    M = sp.Array(mass_matrix(S, Y, R, Z))               # Compute mass matrix M using the current S, Y, R matrices and Z
    H = H_matrix(S, R, Q, Y)                            # Calculate the H matrix for the neural network
    w = sp.Array(w_matrix(H, S, Y, R, Q, a))            # Compute the w vector for the neural network's differential equations
    error_arr = []                                      # Initialize error tracking array
    t_evalval = np.linspace(t_start, t_end, num_eval)   # Define the time evaluation values for the numerical solution
                                                        # Solve the differential equations using scipy's solve_ivp function
    sol = scipy.integrate.solve_ivp(fun=y_prime, t_span=[t_start, t_end], y0=np.array(Y0_vec), t_eval=t_evalval,
                                    args=(M, H, w, error_arr), method='RK23')

    # Plot results for each state variable
    f,ax = plt.subplots()
    for i in range((n*n+n)//2):
        ax.plot(sol.t, sol.y[i], label=f'y{i}')
    f.show()

    # Extract and plot the error values over time
    error_val = [x[0] for x in error_arr]
    error_time = [x[1] for x in error_arr]
    f_er, ax_er = plt.subplots()
    ax_er.plot(error_time, error_val, label='Error')
    f_er.show()

    print(error_val)


if __name__ == "__main__":
    main()
