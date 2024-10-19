import sys
import os
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget

class MainMenu(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Menú Principal")
        self.setGeometry(100, 100, 400, 200)

        # Crear el layout principal
        self.layout = QVBoxLayout()

        # Botón para abrir el programa de Interfaz.py
        self.interfaz_button = QPushButton("Abrir Interfaz - Plano Fase", self)
        self.interfaz_button.clicked.connect(self.open_fase)
        self.layout.addWidget(self.interfaz_button)

        # Botón para abrir el programa de RK4.py
        self.rk4_button = QPushButton("Abrir RK4 - Ecuaciones Diferenciales", self)
        self.rk4_button.clicked.connect(self.open_rk4)
        self.layout.addWidget(self.rk4_button)

        # Botón para salir del menú
        self.exit_button = QPushButton("Salir", self)
        self.exit_button.clicked.connect(self.close)
        self.layout.addWidget(self.exit_button)

        # Configurar el widget central
        central_widget = QWidget()
        central_widget.setLayout(self.layout)
        self.setCentralWidget(central_widget)

    def open_fase(self):
        # Abrir el programa Fase.py
        os.system("python Fase.py")

    def open_rk4(self):
        # Abrir el programa RK4.py
        os.system("python RK4.py")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainMenu()
    window.show()
    sys.exit(app.exec())
