import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLineEdit, QPushButton, QLabel, QComboBox)
from fractions import Fraction

def parse_valores(valores, variables):
    matriz = np.zeros((2, 2))
    
    for i in range(2):
        for j in range(2):
            valor = valores[i * 2 + j].text()
            if valor:
                try:
                    # Convertir valores fraccionarios a flotantes
                    matriz[i, j] = float(Fraction(valor))
                except ValueError:
                    matriz[i, j] = 0
            else:
                matriz[i, j] = 0
    
    return matriz

def calcular_autovalores(A):
    a11, a12 = A[0]
    a21, a22 = A[1]
    a = 1
    b = -(a11 + a22)
    c = a11 * a22 - a12 * a21
    discriminante = b**2 - 4*a*c
    
    if discriminante >= 0:
        lambda1 = (-b + discriminante**0.5) / (2*a)
        lambda2 = (-b - discriminante**0.5) / (2*a)
        return lambda1, lambda2
    else:
        real_part = -b / (2*a)
        imag_part = (abs(discriminante)**0.5) / (2*a)
        lambda1 = complex(real_part, imag_part)
        lambda2 = complex(real_part, -imag_part)
        return lambda1, lambda2

def calcular_autovector(A, autovalor):
    A_minus_lambda_I = [
        [A[0][0] - autovalor.real, A[0][1]],
        [A[1][0], A[1][1] - autovalor.real]
    ]
    
    A_minus_lambda_I = np.array(A_minus_lambda_I, dtype=complex)
    _, _, vh = np.linalg.svd(A_minus_lambda_I)
    autovector = vh[-1]

    # Verificamos si el primer componente es cero
    if autovector[0] != 0:
        autovector = autovector / autovector[0]
    else:
        # Si el primer componente es cero, normalizamos usando el segundo componente
        autovector = autovector / autovector[1]
    
    return autovector

def graficar_autovectores_y_plano_fase(A, autovectores):
    t = np.linspace(-10, 10, 100)
    plt.figure()
    ax = plt.gca()
    
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    
    for vec in autovectores:
        if vec is not None:
            if np.isclose(vec[0].imag, 0) and np.isclose(vec[1].imag, 0):
                x_values = vec[0].real * t
                y_values = vec[1].real * t
                plt.plot(x_values, y_values, label=f'Autovector [x: {vec[0].real:.2f}, y: {vec[1].real:.2f}]')
            else:
                x_values = vec[0].real * t
                y_values = vec[1].real * t
                plt.plot(x_values, y_values, label=f'Autovector [x: {vec[0].real:.2f} + {vec[0].imag:.2f}j, y: {vec[1].real:.2f} + {vec[1].imag:.2f}j]')
    
    x = np.linspace(-10, 10, 400)
    y = np.linspace(-10, 10, 400)
    X, Y = np.meshgrid(x, y)
    U = A[0, 0] * X + A[0, 1] * Y
    V = A[1, 0] * X + A[1, 1] * Y
    plt.streamplot(X, Y, U, V, color='blue', linewidth=1, density=1)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Autovectores y Plano Fase')
    plt.legend()
    plt.grid(True)
    plt.show()

class Interfaz(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setGeometry(100, 100, 400, 400)
        self.setWindowTitle('Cálculo de Autovalores y Autovectores')

        layout = QVBoxLayout()

        # Fila superior para la primera ecuación
        fila_superior_layout = QHBoxLayout()
        
        self.valor11 = QLineEdit(self)
        self.valor11.setPlaceholderText('Ingrese valor')
        fila_superior_layout.addWidget(self.valor11)
        
        self.combo11 = QComboBox(self)
        self.combo11.addItems(['X', 'Y'])
        self.combo11.setCurrentText('X')  # Por defecto 'X'
        fila_superior_layout.addWidget(self.combo11)
        
        self.valor12 = QLineEdit(self)
        self.valor12.setPlaceholderText('Ingrese valor')
        fila_superior_layout.addWidget(self.valor12)
        
        self.combo12 = QComboBox(self)
        self.combo12.addItems(['X', 'Y'])
        self.combo12.setCurrentText('Y')  # Por defecto 'Y'
        fila_superior_layout.addWidget(self.combo12)
        
        layout.addLayout(fila_superior_layout)

        # Fila inferior para la segunda ecuación
        fila_inferior_layout = QHBoxLayout()
        
        self.valor21 = QLineEdit(self)
        self.valor21.setPlaceholderText('Ingrese valor')
        fila_inferior_layout.addWidget(self.valor21)
        
        self.combo21 = QComboBox(self)
        self.combo21.addItems(['X', 'Y'])
        self.combo21.setCurrentText('X')  # Por defecto 'X'
        fila_inferior_layout.addWidget(self.combo21)
        
        self.valor22 = QLineEdit(self)
        self.valor22.setPlaceholderText('Ingrese valor')
        fila_inferior_layout.addWidget(self.valor22)
        
        self.combo22 = QComboBox(self)
        self.combo22.addItems(['X', 'Y'])
        self.combo22.setCurrentText('Y')  # Por defecto 'Y'
        fila_inferior_layout.addWidget(self.combo22)
        
        layout.addLayout(fila_inferior_layout)

        self.boton_calcular = QPushButton('Calcular', self)
        self.boton_calcular.clicked.connect(self.calcular)
        layout.addWidget(self.boton_calcular)

        self.boton_borrar = QPushButton('Borrar', self)
        self.boton_borrar.clicked.connect(self.borrar)
        layout.addWidget(self.boton_borrar)

        self.resultado_label = QLabel(self)
        layout.addWidget(self.resultado_label)

        self.boton_regresar = QPushButton('Regresar al Menú', self)
        self.boton_regresar.clicked.connect(self.regresar_al_menu)
        layout.addWidget(self.boton_regresar)

        self.setLayout(layout)

    def calcular(self):
        # Crear lista de cajas de texto y combos
        valores = [self.valor11, self.valor12, self.valor21, self.valor22]
        combos = [self.combo11, self.combo12, self.combo21, self.combo22]
        
        # Procesar las entradas y construir la matriz
        A = parse_valores(valores, combos)
        
        # Imprimir en consola para depuración
        print("Matriz A:")
        print(A)
        
        try:
            autovalores = calcular_autovalores(A)
            autovectores = [calcular_autovector(A, val) for val in autovalores]
            
            resultado = "Autovalores:\n"
            for i, val in enumerate(autovalores):
                if isinstance(val, complex):
                    resultado += f"λ{i + 1} = {val.real:.2f} + {val.imag:.2f}j\n"
                else:
                    resultado += f"λ{i + 1} = {val:.2f}\n"

            resultado += "\nAutovectores:\n"
            for i, vec in enumerate(autovectores):
                if vec is not None:
                    if np.isclose(vec[0].imag, 0) and np.isclose(vec[1].imag, 0):
                        resultado += f"Vector asociado a λ{i + 1}: [x: {vec[0].real:.2f}, y: {vec[1].real:.2f}]\n"
                    else:
                        resultado += f"Vector asociado a λ{i + 1}: [x: {vec[0].real:.2f} + {vec[0].imag:.2f}j, y: {vec[1].real:.2f} + {vec[1].imag:.2f}j]\n"
            
            self.resultado_label.setText(resultado)
            graficar_autovectores_y_plano_fase(A, autovectores)
        
        except ValueError as e:
            self.resultado_label.setText(str(e))
    
    def borrar(self):
        self.valor11.clear()
        self.valor12.clear()
        self.valor21.clear()
        self.valor22.clear()

        self.combo11.setCurrentText('X')
        self.combo12.setCurrentText('Y')
        self.combo21.setCurrentText('X')
        self.combo22.setCurrentText('Y')

    def regresar_al_menu(self):
        self.close()
        sys.exit()
    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    interfaz = Interfaz()
    interfaz.show()
    sys.exit(app.exec())