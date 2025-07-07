import matplotlib.pyplot as plt
from collections import deque

class Graficador:
    def __init__(self, max_puntos=60):
        self.tiempos   = deque(maxlen=max_puntos)
        self.humedades = deque(maxlen=max_puntos)
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.fig.show()

    def actualizar(self, ts, pct):
        self.tiempos.append(ts)
        self.humedades.append(pct)

        self.ax.clear()
        self.ax.plot(self.tiempos, self.humedades, marker='o')
        self.ax.set_ylim(0, 100)
        self.ax.set_ylabel('Humedad (%)')
        self.ax.set_xlabel('Hora')
        self.ax.set_title('Humedad en tiempo real')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.pause(0.1)
