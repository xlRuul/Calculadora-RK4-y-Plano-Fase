import numpy as np
import matplotlib.pyplot as plt
import re
import math
import sys
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QFormLayout, QPushButton, QLineEdit, QLabel, QTableWidget, QTableWidgetItem, QMessageBox
from PyQt6.QtCore import Qt
import sympy as sp

class MainMenu(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Menú Principal")
        self.setGeometry(100, 100, 400, 250)  # Ajusta el tamaño para incluir el cuarto botón

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        self.ode_button = QPushButton("Resolver Ecuación Diferencial de Primer Orden")
        self.ode_button.clicked.connect(self.open_ode_window)
        self.layout.addWidget(self.ode_button)

        self.sys_button = QPushButton("Resolver Sistema de Ecuaciones Diferenciales")
        self.sys_button.clicked.connect(self.open_sys_window)
        self.layout.addWidget(self.sys_button)

        self.second_order_button = QPushButton("Resolver Ecuación Diferencial de Segundo Orden")
        self.second_order_button.clicked.connect(self.open_second_order_window)
        self.layout.addWidget(self.second_order_button)


        self.men_button = QPushButton("Regresar al Menú")
        self.men_button.clicked.connect(self.close)
        self.layout.addWidget(self.men_button)

    def open_ode_window(self):
        self.hide()
        self.ode_window = RK4Window(mode="ode", main_menu=self)
        self.ode_window.show()

    def open_sys_window(self):
        self.hide()
        self.sys_window = RK4Window(mode="system", main_menu=self)
        self.sys_window.show()

    def open_second_order_window(self):
        self.hide()
        self.second_order_window = SecondOrderDifferentialEquationSolver()
        self.second_order_window.show()


def preprocess_equation(equation):
    equation = re.sub(r'\^', '**', equation)
    equation = re.sub(r'(\d*)e\*\*', r'\1*exp(', equation)
    equation = re.sub(r'e\*\*', 'exp(', equation)
    equation = re.sub(r'exp\(([^)]+)\)', r'exp(\1)', equation)
    if equation.count('exp(') > equation.count(')'):
        equation += ')' * (equation.count('exp(') - equation.count(')'))
    equation = re.sub(r'(\d)([xyv])', r'\1 * \2', equation)
    equation = re.sub(r'([xyv])(\d)', r'\1 * \2', equation)
    equation = re.sub(r'([xyv])([xyv])', r'\1 * \2', equation)
    equation = re.sub(r'[^0-9a-zA-Z\+\-\*/\(\)\^\s\.\*\^e]', '', equation)
    equation = re.sub(r'\s+', ' ', equation).strip()
    return equation

def rk4_ode(f, y0, x0, h, steps):
    x, y = sp.symbols('x y')
    f = preprocess_equation(f)
    f = sp.sympify(f)
    
    f_lambda = sp.lambdify((x, y), f, 'numpy')

    xs = [x0]
    ys = [y0]
    
    current_x = x0
    current_y = y0
    
    for _ in range(steps):
        k1 = f_lambda(current_x, current_y)
        k2 = f_lambda(current_x + 0.5 * h, current_y + 0.5 * h * k1)
        k3 = f_lambda(current_x + 0.5 * h, current_y + 0.5 * h * k2)
        k4 = f_lambda(current_x + h, current_y + h * k3)
        print(f"Calculating new y using: current_y + (h * ({k1:.4f} + 2 * {k2:.4f} + 2 * {k3:.4f} + {k4:.4f})) / 6")

        current_y = current_y + (h * (k1 + 2 * k2 + 2 * k3 + k4)) / 6
        current_x = current_x + h
        
        xs.append(current_x)
        ys.append(current_y)
        print(f"Step {steps + 1}:")
        print(f"    k1 = {k1:.4f}, k2 = {k2:.4f}, k3 = {k3:.4f}, k4 = {k4:.4f}")
    
        print(f"    current_x = {current_x:.4f}, current_y = {current_y:.4f}")
        print("-" * 40)
    return xs, ys


def rk4_system(f1, f2, x0, y0, t0, h, steps):
    t, x, y = sp.symbols('t x y')
    f1 = preprocess_equation(f1)
    f2 = preprocess_equation(f2)
    f1 = sp.sympify(f1)
    f2 = sp.sympify(f2)

    f1_lambda = sp.lambdify((x, y), f1, 'numpy')
    f2_lambda = sp.lambdify((x, y), f2, 'numpy')

    xs = [x0]
    ys = [y0]
    ts = [t0]

    for step in range(steps):
        k1_x = f1_lambda(xs[-1], ys[-1])
        k1_y = f2_lambda(xs[-1], ys[-1])

        k2_x = f1_lambda(xs[-1] + 0.5 * k1_x * h, ys[-1] + 0.5 * k1_y * h)
        k2_y = f2_lambda(xs[-1] + 0.5 * k1_x * h, ys[-1] + 0.5 * k1_y * h)

        k3_x = f1_lambda(xs[-1] + 0.5 * k2_x * h, ys[-1] + 0.5 * k2_y * h)
        k3_y = f2_lambda(xs[-1] + 0.5 * k2_x * h, ys[-1] + 0.5 * k2_y * h)

        k4_x = f1_lambda(xs[-1] + k3_x * h, ys[-1] + k3_y * h)
        k4_y = f2_lambda(xs[-1] + k3_x * h, ys[-1] + k3_y * h)

        x_new = xs[-1] + (k1_x + 2 * k2_x + 2 * k3_x + k4_x) * h / 6
        y_new = ys[-1] + (k1_y + 2 * k2_y + 2 * k3_y + k4_y) * h / 6
        t_new = ts[-1] + h

        # Agrega los nuevos valores a las listas
        xs.append(x_new)
        ys.append(y_new)
        ts.append(t_new)

        # Imprime los resultados de cada iteración
        print(f"Step {step + 1}:")
        print(f"    k1_x = {k1_x:.4f}, k1_y = {k1_y:.4f}")
        print(f"    k2_x = {k2_x:.4f}, k2_y = {k2_y:.4f}")
        print(f"    k3_x = {k3_x:.4f}, k3_y = {k3_y:.4f}")
        print(f"    k4_x = {k4_x:.4f}, k4_y = {k4_y:.4f}")
        print(f"    current_x = {x_new:.4f}, current_y = {y_new:.4f}, t = {t_new:.4f}")
        print("-" * 40)

    return ts, xs, ys

class RK4Window(QWidget):
    def __init__(self, mode, main_menu):
        super().__init__()

        self.setWindowTitle("Resolver Ecuaciones Diferenciales")
        self.setGeometry(100, 100, 500, 400)

        self.layout = QVBoxLayout(self)
        self.mode = mode
        self.main_menu = main_menu  # Almacenar la referencia al MainMenu
        self.create_ui()

    def create_ui(self):
        if self.mode == "ode":
            self.create_ode_ui()
        elif self.mode == "system":
            self.create_system_ui()
        elif self.mode == "second_order":
            self.create_second_order_ui()

    def create_ode_ui(self):
        self.ode_frame = QWidget()
        self.ode_layout = QFormLayout(self.ode_frame)

        # Definición de campos de entrada
        self.eq_label = QLabel("Ecuación Diferencial (dy/dx = ...):")
        self.eq_entry = QLineEdit()
        self.y0_label = QLabel("Valor inicial de y:")
        self.y0_entry = QLineEdit()
        self.x0_label = QLabel("Valor inicial de x:")
        self.x0_entry = QLineEdit()
        self.xf_label = QLabel("Valor final de x:")
        self.xf_entry = QLineEdit()
        self.steps_label = QLabel("Valor de paso:")
        self.steps_entry = QLineEdit()

        # Botones
        self.solve_button = QPushButton("Resolver")
        self.solve_button.clicked.connect(self.solve_ode)
        self.clear_button = QPushButton("Borrar")
        self.clear_button.clicked.connect(self.clear_ode_fields)
        self.back_button = QPushButton("Regresar al Menú Principal")
        self.back_button.clicked.connect(self.go_back_to_menu)
        
        # Etiqueta de resultados
        self.results_label = QLabel()

        # Tabla de resultados
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(2)
        self.results_table.setHorizontalHeaderLabels(['x', 'y'])
        
        # Añadir elementos al layout
        self.ode_layout.addRow(self.eq_label, self.eq_entry)
        self.ode_layout.addRow(self.y0_label, self.y0_entry)
        self.ode_layout.addRow(self.x0_label, self.x0_entry)
        self.ode_layout.addRow(self.xf_label, self.xf_entry)
        self.ode_layout.addRow(self.steps_label, self.steps_entry)
        self.ode_layout.addRow(self.solve_button)
        self.ode_layout.addRow(self.clear_button)
        self.ode_layout.addRow(self.back_button)
        self.ode_layout.addRow(self.results_label)
        self.ode_layout.addRow(self.results_table)

        self.layout.addWidget(self.ode_frame)


    def solve_ode(self):
        try:
            eq = self.eq_entry.text()
            y0 = float(self.y0_entry.text())
            x0 = float(self.x0_entry.text())
            xf = float(self.xf_entry.text())
            h = float(self.steps_entry.text())

            if h <= 0:
                raise ValueError("El tamaño del paso debe ser un número positivo.")
            
            steps = math.ceil((xf - x0) / h)
            xf = x0 + steps * h

            xs, ys = rk4_ode(eq, y0, x0, h, steps)
            
            # Actualizar la tabla de resultados
            self.results_table.setRowCount(len(xs))
            for i, (x, y) in enumerate(zip(xs, ys)):
                self.results_table.setItem(i, 0, QTableWidgetItem(f"{x:.4f}"))
                self.results_table.setItem(i, 1, QTableWidgetItem(f"{y:.4f}"))

            # Graficar resultados
            plt.figure()
            plt.plot(xs, ys, label='Trayectoria')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(f'Ecuación Diferencial: {eq}')
            plt.grid(True)
            plt.legend()
            plt.show()

        except ValueError as ve:
            QMessageBox.warning(self, 'Entrada no válida', str(ve))
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e))


    
    def clear_ode_fields(self):
        self.eq_entry.clear()
        self.y0_entry.clear()
        self.x0_entry.clear()
        self.xf_entry.clear()
        self.steps_entry.clear()
        self.results_label.clear()
    
    def go_back_to_menu(self):
        self.main_menu.show()  # Mostrar el menú principal
        self.close()  # Cerrar la ventana secundaria


    def create_system_ui(self):
        self.sys_frame = QWidget()
        self.sys_layout = QFormLayout(self.sys_frame)
        
        # Definición de campos de entrada
        self.eq1_label = QLabel("Primera Ecuación (dx/dt = ...):")
        self.eq1_entry = QLineEdit()
        self.eq2_label = QLabel("Segunda Ecuación (dy/dt = ...):")
        self.eq2_entry = QLineEdit()
        self.x0_label = QLabel("Valor inicial de x:")
        self.x0_entry = QLineEdit()
        self.y0_label = QLabel("Valor inicial de y:")
        self.y0_entry = QLineEdit()
        self.t0_label = QLabel("Valor inicial de t:")
        self.t0_entry = QLineEdit()
        self.tf_label = QLabel("Valor final de t:")
        self.tf_entry = QLineEdit()
        self.steps_label = QLabel("Valor de paso:")
        self.steps_entry = QLineEdit()

        # Botones
        self.solve_button = QPushButton("Resolver")
        self.solve_button.clicked.connect(self.solve_system)
        self.clear_button = QPushButton("Borrar")
        self.clear_button.clicked.connect(self.clear_system_fields)
        self.back_button = QPushButton("Regresar al Menú Principal")
        self.back_button.clicked.connect(self.go_back_to_menu)
        
        # Etiqueta de resultados
        self.results_label = QLabel()

        # Tabla de resultados
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(3)
        self.results_table.setHorizontalHeaderLabels(['t', 'x', 'y'])
        
        # Añadir elementos al layout
        self.sys_layout.addRow(self.eq1_label, self.eq1_entry)
        self.sys_layout.addRow(self.eq2_label, self.eq2_entry)
        self.sys_layout.addRow(self.x0_label, self.x0_entry)
        self.sys_layout.addRow(self.y0_label, self.y0_entry)
        self.sys_layout.addRow(self.t0_label, self.t0_entry)
        self.sys_layout.addRow(self.tf_label, self.tf_entry)
        self.sys_layout.addRow(self.steps_label, self.steps_entry)
        self.sys_layout.addRow(self.solve_button)
        self.sys_layout.addRow(self.clear_button)
        self.sys_layout.addRow(self.back_button)
        self.sys_layout.addRow(self.results_label)
        self.sys_layout.addRow(self.results_table)

        self.layout.addWidget(self.sys_frame)


    def solve_system(self):
        try:
            eq1 = self.eq1_entry.text()
            eq2 = self.eq2_entry.text()
            x0 = float(self.x0_entry.text())
            y0 = float(self.y0_entry.text())
            t0 = float(self.t0_entry.text())
            tf = float(self.tf_entry.text())
            h = float(self.steps_entry.text())

            if h <= 0:
                raise ValueError("El tamaño del paso debe ser un número positivo.")
            
            steps = int((tf - t0) / h) + 1

            ts, xs, ys = rk4_system(eq1, eq2, x0, y0, t0, h, steps)

            # Actualizar la tabla de resultados
            self.results_table.setRowCount(len(ts))
            for i, (t, x, y) in enumerate(zip(ts, xs, ys)):
                self.results_table.setItem(i, 0, QTableWidgetItem(f"{t:.4f}"))
                self.results_table.setItem(i, 1, QTableWidgetItem(f"{x:.4f}"))
                self.results_table.setItem(i, 2, QTableWidgetItem(f"{y:.4f}"))

            # Graficar resultados
            plt.figure()
            plt.plot(xs, ys, label='Trayectoria')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(f'Sistema de Ecuaciones Diferenciales:\n{eq1} & {eq2}')
            plt.grid(True)
            plt.legend()
            plt.show()

        except ValueError as ve:
            QMessageBox.warning(self, 'Entrada no válida', str(ve))
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e))

    def clear_system_fields(self):
        self.eq1_entry.clear()
        self.eq2_entry.clear()
        self.x0_entry.clear()
        self.y0_entry.clear()
        self.t0_entry.clear()
        self.tf_entry.clear()
        self.steps_entry.clear()
        self.results_label.clear()

class SecondOrderDifferentialEquationSolver(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Resolver Ecuación Diferencial de Segundo Orden")
        self.setGeometry(100, 100, 600, 400)
        
        layout = QVBoxLayout()
        self.main_menu = main_menu 
        form_layout = QFormLayout()
        
        self.coef_a = QLineEdit()
        self.coef_b = QLineEdit()
        self.coef_c = QLineEdit()
        self.equation_rhs = QLineEdit()
        self.initial_x = QLineEdit()
        self.initial_v = QLineEdit()
        self.end_time = QLineEdit()
        self.time_step = QLineEdit()
        
        form_layout.addRow(QLabel("Coeficiente de x'' (a):"), self.coef_a)
        form_layout.addRow(QLabel("Coeficiente de x' (b):"), self.coef_b)
        form_layout.addRow(QLabel("Coeficiente de x (c):"), self.coef_c)
        form_layout.addRow(QLabel("Término no homogéneo (f(t)):") , self.equation_rhs)
        form_layout.addRow(QLabel("Posición inicial (x0):"), self.initial_x)
        form_layout.addRow(QLabel("Velocidad inicial (v0):"), self.initial_v)
        form_layout.addRow(QLabel("Tiempo final (t_end):"), self.end_time)
        form_layout.addRow(QLabel("Paso de tiempo (dt):"), self.time_step)
        
        layout.addLayout(form_layout)
        
        self.solve_button = QPushButton("Resolver")
        self.solve_button.clicked.connect(self.solve_equation)
        layout.addWidget(self.solve_button)
        self.clear_button = QPushButton("Borrar")
        self.clear_button.clicked.connect(self.clear_second_order_fields)
        layout.addWidget(self.clear_button)
        self.back_button = QPushButton("Regresar al Menú Principal")
        self.back_button.clicked.connect(self.go_back_to_menu)
        layout.addWidget(self.back_button)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(3)
        self.results_table.setHorizontalHeaderLabels(['Tiempo', 'Posición (x)', 'Velocidad (v)'])
        layout.addWidget(self.results_table)
        
        self.setLayout(layout)
    
    def solve_equation(self):
        try:
            # Obtener coeficientes de la ecuación
            a_expr = self.parse_expression(self.coef_a.text())
            b_expr = self.parse_expression(self.coef_b.text())
            c_expr = self.parse_expression(self.coef_c.text())
            f_expr = self.parse_expression(self.equation_rhs.text())
            x0 = float(self.initial_x.text())
            v0 = float(self.initial_v.text())
            t_end = float(self.end_time.text())
            dt = float(self.time_step.text())

            # Definir la ecuación diferencial
            t = sp.Symbol('t')
            x = sp.Symbol('x')
            v = sp.Symbol('v')
            a = sp.lambdify(t, a_expr, 'numpy')
            b = sp.lambdify(t, b_expr, 'numpy')
            c = sp.lambdify(t, c_expr, 'numpy')
            f = sp.lambdify(t, f_expr, 'numpy')
            
            def equation(t, x, v):
                return - (b(t) * v + c(t) * x) / a(t) + f(t) / a(t)
            
            # Resolver la ecuación diferencial
            tiempos, posiciones, velocidades = self.rk4_second_order(equation, 0, x0, v0, t_end, dt)
            
            # Actualizar la tabla de resultados
            self.update_results_table(tiempos, posiciones, velocidades)
            
            # Graficar resultados
            self.plot_results(tiempos, posiciones, velocidades)
            
        except ValueError as e:
            print(f"Error en los valores de entrada: {e}")
        except Exception as e:
            print(f"Se produjo un error inesperado: {e}")


    def parse_expression(self, expr):
        # Reemplazar caracteres y operadores en la expresión
        expr = expr.replace('^', '**')
        expr = expr.replace('e**t', 'exp(t)')
        expr = expr.replace('e**x', 'exp(x)')
        expr = expr.replace('e**', 'exp(')
        
        if expr.count('exp(') > expr.count(')'):
            expr += ')'
        
        # Intentar convertir la cadena en una expresión simbólica
        try:
            return sp.sympify(expr)
        except sp.SympifyError:
            print(f"Error al interpretar la expresión: {expr}")
            return 0

    def rk4_second_order(self, equation, t0, x0, v0, t_end, dt):
        num_steps = int((t_end - t0) / dt)
        tiempos = np.linspace(t0, t_end, num_steps + 1)
        posiciones = np.zeros(num_steps + 1)
        velocidades = np.zeros(num_steps + 1)
        
        posiciones[0] = x0
        velocidades[0] = v0

        for i in range(num_steps):
            t = tiempos[i]
            x = posiciones[i]
            v = velocidades[i]

            # Calcular k1
            k1_x = dt * v
            k1_v = dt * equation(t, x, v)
            
            # Calcular k2
            k2_x = dt * (v + 0.5 * k1_v)
            k2_v = dt * equation(t + 0.5 * dt, x + 0.5 * k1_x, v + 0.5 * k1_v)
            
            # Calcular k3
            k3_x = dt * (v + 0.5 * k2_v)
            k3_v = dt * equation(t + 0.5 * dt, x + 0.5 * k2_x, v + 0.5 * k2_v)
            
            # Calcular k4
            k4_x = dt * (v + k3_v)
            k4_v = dt * equation(t + dt, x + k3_x, v + k3_v)
            
            # Actualizar posiciones y velocidades
            posiciones[i + 1] = x + (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6
            velocidades[i + 1] = v + (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6
            
            # Imprimir resultados de cada paso
            print(f"Step {i + 1}:")
            print(f"    t = {t:.1f}, x = {x:.4f}, v = {v:.4f}")
            print(f"    k1_x = {k1_x:.4f}, k1_v = {k1_v:.4f}")
            print(f"    k2_x = {k2_x:.4f}, k2_v = {k2_v:.4f}")
            print(f"    k3_x = {k3_x:.4f}, k3_v = {k3_v:.4f}")
            print(f"    k4_x = {k4_x:.4f}, k4_v = {k4_v:.4f}")
            print(f"    new_x = {posiciones[i + 1]:.4f}, new_v = {velocidades[i + 1]:.4f}")
            print("-" * 40)

        return tiempos, posiciones, velocidades

    def update_results_table(self, tiempos, posiciones, velocidades):
        self.results_table.setRowCount(len(tiempos))
        for i in range(len(tiempos)):
            self.results_table.setItem(i, 0, QTableWidgetItem(f"{tiempos[i]:.2f}"))
            self.results_table.setItem(i, 1, QTableWidgetItem(f"{posiciones[i]:.4f}"))
            self.results_table.setItem(i, 2, QTableWidgetItem(f"{velocidades[i]:.4f}"))

    def plot_results(self, tiempos, posiciones, velocidades):
        a_expr = self.parse_expression(self.coef_a.text())
        b_expr = self.parse_expression(self.coef_b.text())
        c_expr = self.parse_expression(self.coef_c.text())
        f_expr = self.parse_expression(self.equation_rhs.text())

        # Formatear expresiones
        def format_expression(expr, var):
            expr_str = str(expr)
            if expr_str.startswith('+'):
                expr_str = expr_str[1:]  # Eliminar el signo '+' al principio si existe
            expr_str = expr_str.replace('**', '^')
            expr_str = expr_str.replace('exp(', 'e^')
            expr_str = expr_str.replace(')', '')
            expr_str = expr_str.replace(var, f'{var}')
            return expr_str
        
        a_str = format_expression(a_expr, 'x')
        b_str = format_expression(b_expr, 'x')
        c_str = format_expression(c_expr, 'x')
        f_str = format_expression(f_expr, 'x')

        # Construir el título de la gráfica
        def clean_signs(expr):
            expr = expr.replace(' + -', ' - ')
            return expr

        title_expr = f"{clean_signs(a_str)}x'' + {clean_signs(b_str)}x' + {clean_signs(c_str)}x = {f_str}"

        plt.figure(figsize=(12, 6))

        plt.subplot(2, 1, 1)
        plt.plot(tiempos, posiciones, label='Posición (x)')
        plt.xlabel('Tiempo')
        plt.ylabel('Posición (x)')
        plt.grid(True)
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(tiempos, velocidades, label='Velocidad (v)', color='r')
        plt.xlabel('Tiempo')
        plt.ylabel('Velocidad (v)')
        plt.title(f'Ecuación Diferencial de Segundo Orden\n{title_expr}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def clear_second_order_fields(self):
        self.coef_a.clear()
        self.coef_b.clear()
        self.coef_c.clear()
        self.equation_rhs.clear()
        self.initial_x.clear()
        self.initial_v.clear()
        self.end_time.clear()
        self.time_step.clear()


    def go_back_to_menu(self):
            self.main_menu.show()  # Mostrar el menú principal
            self.close()  # Cerrar la ventana secundaria
if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_menu = MainMenu()
    main_menu.show()
    sys.exit(app.exec())