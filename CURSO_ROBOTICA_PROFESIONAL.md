# Curso de Robótica Profesional
## Python · Rust · ESP32 · Inteligencia Artificial

> Curso de paga con kit físico incluido.
> Cada proyecto viene con sus componentes electrónicos listos y las piezas
> fabricadas en nuestra impresora 3D (200 × 200 × 200 mm) y cortadora láser (400 × 400 mm).
> Sin MATLAB. El alumno programa, ensambla y hace funcionar su robot desde el día uno.

---

## Modelo del curso

| Elemento | Detalle |
|----------|---------|
| **Formato** | Video + documentación + soporte activo |
| **Kit físico** | Componentes electrónicos + piezas fabricadas enviados por proyecto |
| **Impresión 3D** | Volumen máximo de pieza: **200 × 200 × 200 mm** (PETG / TPU / PLA) |
| **Corte láser** | Área máxima de hoja: **400 × 400 mm** (MDF 4 mm / acrílico 3 mm) |
| **Diseño incluido** | Archivos STL, DXF y STEP de cada pieza disponibles en el campus |
| **Lenguajes** | Python · Rust · Swift (iOS) · React Native / TypeScript (Android + iOS) |
| **Control remoto** | Radio RC (ELRS/LoRa) · Web secuencial · App móvil nativa y multiplataforma |
| **Sin prerequisito de taller** | Las piezas llegan fabricadas; el alumno aprende a rediseñarlas |

---

## Stack tecnológico

| Capa | Tecnología | Rol |
|------|-----------|-----|
| Firmware | **Rust** + esp-idf-hal | Control en tiempo real, seguridad de memoria |
| Microcontrolador | **ESP32** | Hardware embebido, WiFi/BLE, FreeRTOS |
| Algoritmos | **Python** | Cinemática, dinámica, IA, simulación |
| Comunicación | micro-ROS / MQTT / BLE | ESP32 ↔ Python ↔ ROS2 |
| Simulación | Gazebo / Webots | Digital twin antes del hardware |
| IA | PyTorch / ONNX / Stable-Baselines3 | Visión, control adaptativo, RL |

---

## Módulo 0 — Fundamentos transversales
*Prerrequisito para todo lo demás*

### 0.1 Matemáticas para Robótica — Fundamentos Profundos en Python

> El robot es matemática ejecutada en silicio. Esta sección cubre todos los
> conceptos matemáticos que aparecerán a lo largo del curso, implementados
> en Python para que sean tangibles desde el primer día.

---

#### 0.1.1 Álgebra Lineal — el lenguaje de la geometría del robot

```python
import numpy as np

# ── Vectores y operaciones básicas ──────────────────────────────────────────
p = np.array([1.0, 2.0, 3.0])        # punto en 3D
v = np.array([0.0, 1.0, 0.0])        # vector de dirección
print(np.linalg.norm(p))             # magnitud = √(1+4+9) = 3.742
print(np.dot(p, v))                  # producto punto = 2.0
print(np.cross(p, v))                # producto cruz = [-3, 0, 1]

# ── Matrices de rotación SO(3) ───────────────────────────────────────────────
def rot_x(θ): 
    c, s = np.cos(θ), np.sin(θ)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])

def rot_y(θ): 
    c, s = np.cos(θ), np.sin(θ)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])

def rot_z(θ): 
    c, s = np.cos(θ), np.sin(θ)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

R = rot_z(np.pi/4) @ rot_x(np.pi/6)   # rotación compuesta (multiplicación de matrices)
print("Determinante:", np.linalg.det(R))  # siempre +1 para SO(3)
print("Inversa = Transpuesta:", np.allclose(R.T, np.linalg.inv(R)))  # True

# ── Valores y vectores propios (eigenvalues) ─────────────────────────────────
# Útiles en análisis de sistemas de control y vibración
A = np.array([[4, 1], [2, 3]])
eigenvalues, eigenvectors = np.linalg.eig(A)
print("Valores propios:", eigenvalues)    # λ₁=5, λ₂=2
# El sistema A converge si max(|λ|) < 1

# ── Descomposición SVD — para análisis de manipulabilidad ────────────────────
J = np.array([[1,0,0.5],[0,1,0.3],[0.2,0.1,1]])  # Jacobiana de ejemplo
U, s, Vt = np.linalg.svd(J)
print("Valores singulares:", s)
# Índice de manipulabilidad de Yoshikawa:
w = np.sqrt(np.linalg.det(J @ J.T))
print(f"Manipulabilidad w = {w:.4f}")   # 0 = singularidad
```

---

#### 0.1.2 Transformaciones Homogéneas SE(3)

```python
import numpy as np

def T_from_Rp(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Construye matriz homogénea 4×4 a partir de R (3×3) y p (3,)."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = p
    return T

def T_inv(T: np.ndarray) -> np.ndarray:
    """Inversa eficiente de transformación homogénea (no usar np.linalg.inv)."""
    R, p = T[:3,:3], T[:3,3]
    T_i = np.eye(4)
    T_i[:3,:3] = R.T
    T_i[:3, 3] = -R.T @ p
    return T_i

# Cadena cinemática: 3 eslabones
T01 = T_from_Rp(rot_z(np.pi/4), [0.3, 0.0, 0.0])   # base → eslabon 1
T12 = T_from_Rp(rot_z(np.pi/6), [0.25, 0.0, 0.0])  # eslabon 1 → 2
T23 = T_from_Rp(rot_z(0.0),     [0.2, 0.0, 0.0])   # eslabon 2 → efector

T_EE = T01 @ T12 @ T23    # posición y orientación del efector final
print("Posición del efector:", T_EE[:3, 3])
```

---

#### 0.1.3 Cuaterniones — Rotación sin Gimbal Lock

```python
from scipy.spatial.transform import Rotation as Rot

# ── Crear cuaternión desde eje-ángulo ────────────────────────────────────────
r = Rot.from_rotvec(np.pi/4 * np.array([0, 0, 1]))  # 45° alrededor de Z
q = r.as_quat()    # [x, y, z, w] — scipy usa esta convención
print("Cuaternión:", q)

# ── Composición de rotaciones (sin gimbal lock) ──────────────────────────────
r1 = Rot.from_euler('z',  45, degrees=True)
r2 = Rot.from_euler('x',  30, degrees=True)
r_total = r2 * r1    # primero r1, luego r2
print("Ángulos Euler resultantes:", r_total.as_euler('zyx', degrees=True))

# ── SLERP — interpolación suave entre dos orientaciones ──────────────────────
from scipy.spatial.transform import Slerp

times = [0, 1]
rots  = Rot.concatenate([Rot.from_euler('z', 0, degrees=True),
                          Rot.from_euler('z', 90, degrees=True)])
slerp = Slerp(times, rots)

# 5 puntos intermedios: interpolación esférica lineal
for t in np.linspace(0, 1, 5):
    angle = slerp(t).as_euler('z', degrees=True)
    print(f"t={t:.2f} → {angle[0]:.1f}°")
# → 0.0°, 22.5°, 45.0°, 67.5°, 90.0°
```

---

#### 0.1.4 Álgebra Simbólica con SymPy — Jacobiana y Cinemática

```python
from sympy import symbols, cos, sin, Matrix, simplify, pi, atan2, sqrt

# ── Parámetros DH simbólicos para robot 2-DOF ────────────────────────────────
θ1, θ2, l1, l2 = symbols('θ1 θ2 l1 l2', real=True)

# Cinemática directa simbólica
x_ee = l1*cos(θ1) + l2*cos(θ1 + θ2)
y_ee = l1*sin(θ1) + l2*sin(θ1 + θ2)

# ── Jacobiana geométrica — derivada parcial de la posición ───────────────────
J_sym = Matrix([
    [x_ee.diff(θ1), x_ee.diff(θ2)],
    [y_ee.diff(θ1), y_ee.diff(θ2)],
])
print("Jacobiana:")
print(simplify(J_sym))

# ── Velocidad del efector final ──────────────────────────────────────────────
# [ẋ, ẏ] = J × [θ̇₁, θ̇₂]
θ1_dot, θ2_dot = symbols('θ1_dot θ2_dot')
vel_ee = J_sym @ Matrix([θ1_dot, θ2_dot])
print("Velocidad del efector:", simplify(vel_ee))

# ── Singularidades: det(J) = 0 ───────────────────────────────────────────────
det_J = simplify(J_sym.det())
print("Determinante de J:", det_J)
# det_J = l1*l2*sin(θ2) → singularidad cuando θ2 = 0 o π (brazo extendido/doblado)
```

---

#### 0.1.5 Ecuaciones Diferenciales — Dinámica del Robot

```python
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ── Péndulo simple — modelo del eslabón de un brazo ─────────────────────────
def pendulum(t, state, g=9.81, l=0.3, b=0.1):
    """
    state = [θ, θ̇]
    g: gravedad, l: longitud, b: fricción viscosa
    θ̈ = -(g/l)·sin(θ) - b·θ̇
    """
    θ, θ_dot = state
    return [θ_dot, -(g/l)*np.sin(θ) - b*θ_dot]

sol = solve_ivp(pendulum, t_span=[0, 5], y0=[np.pi/3, 0],
                t_eval=np.linspace(0, 5, 500), method='RK45')

plt.plot(sol.t, np.degrees(sol.y[0]), label='θ (deg)')
plt.plot(sol.t, np.degrees(sol.y[1]), label='θ̇ (deg/s)')
plt.xlabel('Tiempo (s)'); plt.legend(); plt.grid(True)
plt.title('Péndulo simple: respuesta libre')
plt.savefig('pendulum.png', dpi=150)

# ── Ecuaciones de Euler-Lagrange para robot 2-DOF ────────────────────────────
def robot_2dof_dynamics(t, state, τ1=0.0, τ2=0.0):
    """
    Dinámica simplificada con masas iguales (m=1 kg) y l=0.3 m
    M(θ)·θ̈ + C(θ,θ̇)·θ̇ + G(θ) = τ
    """
    θ1, θ2, dθ1, dθ2 = state
    m, l, g = 1.0, 0.3, 9.81

    # Matriz de inercia M
    M11 = (m*l**2) + (m*(2*l)**2)
    M12 = m*l**2*np.cos(θ2)
    M = np.array([[M11, M12], [M12, m*l**2]])

    # Fuerzas de Coriolis/centrífugas C
    h = -m*l**2*np.sin(θ2)
    C = np.array([[h*dθ2, h*(dθ1+dθ2)], [-h*dθ1, 0]])

    # Gravedad G
    G = np.array([m*g*l*np.cos(θ1) + m*g*2*l*np.cos(θ1+θ2),
                  m*g*l*np.cos(θ1+θ2)])

    τ = np.array([τ1, τ2])
    ddθ = np.linalg.solve(M, τ - C @ [dθ1,dθ2] - G)
    return [dθ1, dθ2, ddθ[0], ddθ[1]]
```

---

#### 0.1.6 Probabilidad y Estadística para Localización

```python
import numpy as np
from scipy.stats import multivariate_normal

# ── Distribución gaussiana 2D — representación de posición del robot ─────────
mean  = np.array([2.0, 3.0])                     # posición estimada
cov   = np.array([[0.5, 0.1], [0.1, 0.3]])        # incertidumbre (covarianza)
dist  = multivariate_normal(mean=mean, cov=cov)

# Probabilidad de estar en una posición específica
p = dist.pdf([2.1, 3.05])
print(f"P(robot en [2.1, 3.05]) = {p:.4f}")

# ── Filtro de Kalman — estimación de posición con ruido ──────────────────────
class KalmanFilter1D:
    def __init__(self, x0=0.0, P0=1.0, Q=0.1, R=1.0):
        self.x = x0    # estado estimado
        self.P = P0    # covarianza del error
        self.Q = Q     # ruido del proceso
        self.R = R     # ruido de la medición

    def predict(self, u=0.0, dt=0.1):
        """Predicción: el robot se movió con control u en dt segundos."""
        self.x = self.x + u * dt
        self.P = self.P + self.Q

    def update(self, z):
        """Corrección: nueva medición z (GPS, LiDAR, encoder)."""
        K    = self.P / (self.P + self.R)   # ganancia de Kalman
        self.x = self.x + K * (z - self.x) # corrección del estado
        self.P = (1 - K) * self.P          # reducción de incertidumbre
        return self.x

kf = KalmanFilter1D(x0=0.0, P0=1.0, Q=0.01, R=0.5)
for z in [0.1, 0.18, 0.31, 0.42, 0.52]:   # mediciones ruidosas
    kf.predict(u=0.1, dt=1.0)
    x_est = kf.update(z)
    print(f"Medición: {z:.2f} → Estimación: {x_est:.3f} (P={kf.P:.4f})")
```

---

#### 0.1.7 Optimización — Para Control y Planificación

```python
from scipy.optimize import minimize, differential_evolution
import numpy as np

# ── Cinemática inversa por optimización ──────────────────────────────────────
def fk_2dof(θ, l=[0.3, 0.25]):
    """Cinemática directa 2-DOF: devuelve posición del efector."""
    x = l[0]*np.cos(θ[0]) + l[1]*np.cos(θ[0]+θ[1])
    y = l[0]*np.sin(θ[0]) + l[1]*np.sin(θ[0]+θ[1])
    return np.array([x, y])

def ik_cost(θ, target):
    """Función de costo: distancia entre efector actual y objetivo."""
    return np.linalg.norm(fk_2dof(θ) - target)

target = np.array([0.45, 0.10])
result = minimize(ik_cost, x0=[0.5, -0.5], args=(target,),
                  method='SLSQP',
                  bounds=[(-np.pi, np.pi), (-np.pi, np.pi)])

print(f"IK para {target}: θ = [{np.degrees(result.x[0]):.1f}°, "
      f"{np.degrees(result.x[1]):.1f}°]")
print(f"Error posición: {result.fun*1000:.2f} mm")

# ── Optimización de ganancias PID con Nelder-Mead ────────────────────────────
def simulate_pid(gains):
    kp, ki, kd = gains
    e, ei, prev_e, x = 0.0, 0.0, 0.0, 0.0
    overshoot, settling = 0.0, 0.0
    for t in range(500):
        target = 1.0
        e = target - x
        ei += e * 0.01
        de = (e - prev_e) / 0.01
        u = kp*e + ki*ei + kd*de
        x += u * 0.01 - 0.05*x   # planta 1er orden simple
        prev_e = e
        if x > 1.0: overshoot = max(overshoot, x - 1.0)
        if t > 200 and abs(e) < 0.02: settling = t * 0.01
    return overshoot * 10 + (settling if settling else 5.0)

result_pid = differential_evolution(simulate_pid,
                                    bounds=[(0,20),(0,5),(0,2)],
                                    seed=42, maxiter=100)
kp_opt, ki_opt, kd_opt = result_pid.x
print(f"PID óptimo: Kp={kp_opt:.2f}, Ki={ki_opt:.2f}, Kd={kd_opt:.2f}")
```

---

#### 0.1.8 Señales y Sistemas — Para Control Digital

```python
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# ── Transformada de Fourier — analizar vibración del motor ───────────────────
t  = np.linspace(0, 1, 1000)
# Señal de vibración: 50 Hz (motor) + ruido + 150 Hz (resonancia mecánica)
vibration = (np.sin(2*np.pi*50*t) + 0.3*np.sin(2*np.pi*150*t)
             + 0.2*np.random.randn(len(t)))

freqs = np.fft.rfftfreq(len(t), 1/1000)
fft   = np.abs(np.fft.rfft(vibration))
peak_freq = freqs[np.argmax(fft)]
print(f"Frecuencia dominante: {peak_freq:.1f} Hz")

# ── Discretización de controladores (Tustin / bilineal) ──────────────────────
# Controlador PID continuo → discreto para ESP32
Ts = 0.01  # 100 Hz de control
Kp, Ki, Kd = 5.0, 1.0, 0.5
# PID en forma de función de transferencia:
# C(s) = Kp + Ki/s + Kd*s → C(s) = (Kd*s² + Kp*s + Ki) / s
num = [Kd, Kp, Ki]
den = [1, 0]
sys_c = signal.TransferFunction(num, den)
sys_d = sys_c.to_discrete(Ts, method='bilinear')
print("Coef. digitales del PID (b, a):", sys_d.num, sys_d.den)

# ── Respuesta en frecuencia (Bode plot) ──────────────────────────────────────
# Planta motor DC: G(s) = 1 / (τs + 1),  τ = 0.1 s
tau   = 0.1
plant = signal.TransferFunction([1], [tau, 1])
w, mag, phase = signal.bode(plant)
# Frecuencia de cruce de ganancia (donde |G(jω)| = 1):
gain_crossover = w[np.argmin(np.abs(mag))]
print(f"Frecuencia de cruce: {gain_crossover/(2*np.pi):.2f} Hz")

# ── Filtro paso-bajo para señales de sensores ────────────────────────────────
def low_pass_filter(cutoff_hz: float, fs: float, order: int = 4):
    """Butterworth LP digital para filtrar ruido de IMU o encoders."""
    nyq  = fs / 2
    norm = cutoff_hz / nyq
    b, a = signal.butter(order, norm, btype='low')
    return b, a

b, a = low_pass_filter(cutoff_hz=20, fs=1000)   # corte 20 Hz, muestreo 1 kHz
# Aplicar a señal de IMU:
# imu_filtered = signal.lfilter(b, a, imu_raw_signal)
```

---

#### 0.1.9 Geometría Computacional — Para Planificación y Detección

```python
import numpy as np

# ── Transformación de coordenadas polares ↔ cartesianas ──────────────────────
# LiDAR entrega datos en polares: (r, θ) → convertir a (x, y)
def polar_to_cartesian(ranges: np.ndarray, angles: np.ndarray) -> np.ndarray:
    x = ranges * np.cos(angles)
    y = ranges * np.sin(angles)
    return np.stack([x, y], axis=1)

# ── Intersección de segmentos — detección de colisión 2D ─────────────────────
def segments_intersect(p1, p2, p3, p4) -> bool:
    """¿Se cruzan los segmentos p1-p2 y p3-p4?"""
    d1, d2 = p2 - p1, p4 - p3
    cross = d1[0]*d2[1] - d1[1]*d2[0]
    if abs(cross) < 1e-10:
        return False   # paralelos
    t = ((p3[0]-p1[0])*d2[1] - (p3[1]-p1[1])*d2[0]) / cross
    u = ((p3[0]-p1[0])*d1[1] - (p3[1]-p1[1])*d1[0]) / cross
    return (0 <= t <= 1) and (0 <= u <= 1)

# ── Convex Hull — envoltura convexa de obstáculos detectados ─────────────────
from scipy.spatial import ConvexHull
points = np.random.rand(15, 2) * 10
hull = ConvexHull(points)
print(f"Vértices de la envoltura: {hull.vertices}")
print(f"Área del obstáculo: {hull.volume:.3f} m²")

# ── Distancia punto-segmento (para path following) ───────────────────────────
def point_to_segment_distance(p, a, b):
    """Distancia del punto p al segmento de recta a-b."""
    ab = b - a
    t  = np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-10)
    t  = np.clip(t, 0, 1)
    closest = a + t * ab
    return np.linalg.norm(p - closest), closest

p = np.array([3.0, 4.0])
dist, foot = point_to_segment_distance(p, np.array([0.,0.]), np.array([5.,0.]))
print(f"Distancia al camino: {dist:.3f} m, pie: {foot}")
```

---

#### 0.1.10 Resumen: Matemáticas → Módulo del curso

| Concepto matemático | Aparece en módulo | Herramienta Python |
|--------------------|-------------------|--------------------|
| Matrices rotación SO(3) | M2 Cinemática | `numpy`, `scipy.spatial.transform` |
| Transformaciones homogéneas SE(3) | M2, M3 | `numpy` arrays 4×4 |
| Cuaterniones + SLERP | M2, M6 | `scipy.spatial.transform.Rotation` |
| Jacobiana simbólica | M2, M3 | `sympy.Matrix` |
| Euler-Lagrange | M3 Dinámica | `scipy.integrate.solve_ivp` |
| Kalman Filter | M4 SLAM | `numpy` / `filterpy` |
| Optimización (IK, PID) | M2, M3 | `scipy.optimize` |
| FFT + Bode | M3 Control | `scipy.signal` |
| Filtros digitales | M3, M4 | `scipy.signal.butter` |
| SVD / Manipulabilidad | M2 | `numpy.linalg.svd` |
| Convex Hull, geometría | M4 Nav | `scipy.spatial` |
| Probabilidad gaussiana | M4 SLAM | `scipy.stats` |

### 0.2 Python Profesional para Robótica
- Entornos virtuales, `pyproject.toml`, gestión de dependencias
- Programación orientada a objetos aplicada a robots
- Comunicación serial Python ↔ ESP32 (`pyserial`)
- Async I/O con `asyncio` para múltiples sensores en paralelo
- Testing con `pytest` para algoritmos de control

### 0.3 Rust Fundamentals para Embebidos
- Ownership, borrowing y lifetimes aplicados a drivers de hardware
- Traits como interfaces de sensores y actuadores
- `no_std`: Rust sin sistema operativo
- Manejo de errores con `Result` y `Option` en firmware
- Herramientas: `cargo`, `probe-rs`, `defmt` (logging embebido)

### 0.4 ESP32 como Plataforma Robótica
- Arquitectura ESP32: dual-core Xtensa, periféricos, memoria
- Configuración con `esp-idf` y `esp-idf-hal` en Rust
- FreeRTOS: tareas, colas, semáforos desde Rust
- Modo low-power para robots con batería
- OTA (Over-The-Air) updates en campo

---

## Módulo 1 — Hardware y Actuadores
*Controla el mundo físico desde Rust en ESP32*

### 1.1 Motores y Drivers
- Motores DC con encoder: PWM, H-bridge (L298N, DRV8833) en Rust
- Motores paso a paso: control por pasos y microstepping
- Servomotores: posición absoluta vía PWM
- BLDC: fundamentos de ESC y control vectorial (Field Oriented Control)

### 1.2 Encoders y Retroalimentación
- Encoder cuadrático: lectura de interrupciones en ESP32 con Rust
- Cálculo de velocidad y posición en tiempo real
- Filtro anti-rebote por software en Rust

### 1.3 Sensores para Robótica
- IMU (MPU-6050, BNO055): lectura I2C con `esp-idf-hal`
- LiDAR (RPLIDAR, TF-Luna): UART desde Rust, parsing de tramas
- Ultrasonido: medición por tiempo de vuelo con timers ESP32
- Cámara OV2640 con ESP32-CAM: streaming MJPEG a Python

### 1.4 Comunicación ESP32 ↔ Python
- Protocolo binario personalizado sobre UART (serialización con `serde`)
- WebSocket WiFi: ESP32 como servidor, Python como cliente
- BLE GATT: servicios y características desde Rust (`esp32-nimble`)
- MQTT con `rumqttc` en Python y `esp-mqtt` en Rust

---

### 1.5 Construcción de Motores Eléctricos con Cobre e Imanes

> Fabricar un motor desde cero transforma al estudiante de usuario a diseñador.
> Esta sección cubre la física, el bobinado a mano y el diseño de motores
> BLDC/brushless impresos en 3D con imanes de neodimio.

---

#### 1.5.1 Física del motor eléctrico

**Ley de Lorentz — la fuerza que mueve el rotor:**
```
F = I × L × B × sin(θ)

donde:
  I = corriente en el conductor (A)
  L = longitud del conductor en el campo (m)
  B = densidad de flujo magnético (T)
  θ = ángulo entre conductor y campo (90° = máxima fuerza)
```

**Ley de inducción de Faraday — la FEM generada:**
```
ε = −N × dΦ/dt

donde:
  N = número de espiras
  Φ = flujo magnético (Wb)
  dΦ/dt = tasa de cambio del flujo (cuando el rotor gira)
```

**Fórmulas de diseño clave:**

| Parámetro | Fórmula | Unidades |
|-----------|---------|---------|
| Constante de par (Kt) | Kt = 60 / (2π × Kv) | N·m/A |
| Constante de velocidad (Kv) | Kv = RPM/V sin carga | RPM/V |
| Torque | T = Kt × I | N·m |
| Velocidad sin carga | ω₀ = Kv × V | RPM |
| Potencia mecánica | P = T × ω | W |
| Resistencia de bobinado | R = ρ × L / A | Ω |
| Calor disipado | P_calor = I² × R | W |

---

#### 1.5.2 Tipos de motores y cuándo construirlos

| Tipo | Bobinado | Imanes | Kv típico | Uso en robótica | Construible a mano |
|------|---------|--------|-----------|-----------------|-------------------|
| **DC con escobillas** | Rotor (armadura) | Estátor (ferrita) | — | Motores pequeños, aprendizaje | ✅ Fácil |
| **BLDC outrunner** | Estátor | Rotor (neodimio) | 100–2000 RPM/V | Drones, robots rápidos | ✅ Medio |
| **BLDC inrunner** | Estátor | Rotor interno | 1000–5000 RPM/V | Velocidad, precisión | ✅ Medio |
| **Stepper** | Dos fases, estátor | Rotor mixto | — | Posición exacta, impresoras | ⚠️ Difícil |
| **Motor lineal** | Bobina traslacional | Imanes lineales | — | Actuadores lineales | ⚠️ Avanzado |

---

#### 1.5.3 Motor DC con escobillas — construcción desde cero

**Componentes necesarios:**
- Núcleo de rotor: carrete de PETG impreso en 3D (Ø30 mm, 3 ranuras)
- Alambre de cobre esmaltado: 0.3 mm Ø, longitud ≈ 10 m
- Imanes de neodimio N35: 2× arco o 4× bloque (para estátor)
- Escobillas: carbón de una pila AA cortado o cobre-grafito
- Colector: cobre de PCB cortado, 3 segmentos

**Proceso de bobinado:**
```
1. Imprimir rotor con 3 dientes en PETG
   → CadQuery: 3 ranuras de 3 mm × 10 mm × 25 mm de longitud

2. Bobinar cada ranura con alambre 0.3 mm:
   - Bobina A: ranura 1-2, sentido horario, 30 espiras
   - Bobina B: ranura 2-3, sentido horario, 30 espiras
   - Bobina C: ranura 3-1, sentido horario, 30 espiras
   → Aislar capas con cinta de papel o barniz
   → Terminales a los 3 segmentos del colector

3. Instalar escobillas en contacto con el colector
   → Presión suave (~0.1 N) para buen contacto sin desgaste excesivo

4. Montar estátor con 2 imanes de neodimio opuestos:
   → Norte hacia la armadura, Sur hacia la armadura (campo radial)
   → Distancia imán-rotor: 0.5–1.0 mm (entrehierro)

5. Test: aplicar 3–5 V DC → debe girar libremente
   → Medir Kv: RPM / V (ej: 500 RPM a 3V → Kv = 167 RPM/V)
```

---

#### 1.5.4 Motor BLDC outrunner impreso en 3D

El outrunner es el tipo más usado en drones y robots rápidos.
El estátor es fijo (bobinas), el rotor exterior gira alrededor (imanes pegados).

**Diseño típico 12N14P (12 ranuras, 14 polos):**
```
Ranuras (N) = número de bobinas del estátor
Polos  (P) = número de imanes del rotor

Frecuencia eléctrica: f_elec = RPM × (P/2) / 60
Para 1000 RPM con 14 polos: f_elec = 1000 × 7 / 60 = 116.7 Hz

Paso eléctrico entre ranuras: 360° × P / (2 × N) = 360 × 14 / 24 = 210°
→ Las bobinas se ubican con desfase de 210° eléctricos
```

**Esquema de bobinado LRK (estándar BLDC):**
```
Ranuras:  1   2   3   4   5   6   7   8   9   10  11  12
Bobinas: +A  +B  +C  -A  -B  -C  +A  +B  +C  -A  -B  -C

Fases: A = ranuras 1, 4, 7, 10
       B = ranuras 2, 5, 8, 11
       C = ranuras 3, 6, 9, 12
Conexión estrella (Y): los tres negativos al nodo común
```

```python
# Calculador de motor BLDC en Python
import math

def bldc_design(
    target_kv: float,       # RPM/V deseadas
    supply_voltage: float,  # V
    target_current: float,  # A (corriente nominal)
    poles: int = 14,        # número de imanes
    slots: int = 12,        # número de ranuras
    wire_gauge_mm: float = 0.5,  # diámetro del alambre de cobre
):
    # Estimación de espiras por bobina para Kv objetivo
    # Kv ≈ 60000 / (N_turns × poles × B × Area) (simplificado)
    kv_ref = target_kv
    turns_estimate = round(1000 / kv_ref)

    # Resistencia de fase
    turns_total = turns_estimate * slots
    wire_length_m = turns_total * 0.08   # 80mm por espira (estimado)
    rho_cu = 1.72e-8                     # Ω·m cobre
    area_wire = math.pi * (wire_gauge_mm / 2000) ** 2
    R_phase = rho_cu * wire_length_m / area_wire

    # Pérdidas en cobre
    P_copper = 3 * (target_current ** 2) * R_phase

    # Potencia mecánica estimada
    rpm_no_load = kv_ref * supply_voltage
    torque_kt   = 60 / (2 * math.pi * kv_ref)
    power_mech  = torque_kt * target_current * (rpm_no_load * 2 * math.pi / 60)

    return {
        "turns_per_slot":    turns_estimate,
        "wire_length_m":     round(wire_length_m, 2),
        "R_phase_ohm":       round(R_phase, 3),
        "copper_losses_W":   round(P_copper, 2),
        "torque_Nm":         round(torque_kt * target_current, 4),
        "rpm_no_load":       round(rpm_no_load),
        "power_mechanical_W": round(power_mech, 2),
    }

# Ejemplo: motor BLDC Kv=1000, 11.1V, 5A
d = bldc_design(target_kv=1000, supply_voltage=11.1, target_current=5.0)
for k, v in d.items():
    print(f"  {k}: {v}")
```

**Diseño CadQuery del estátor y rotor:**
```python
import cadquery as cq

def bldc_stator(slots: int = 12, od_mm: float = 30, id_mm: float = 15,
                height: float = 10, tooth_w: float = 3.5):
    """Estátor BLDC impreso en PETG para bobinar a mano."""
    slot_angle = 360 / slots
    stator = (
        cq.Workplane("XY")
        .circle(od_mm / 2)
        .circle(id_mm / 2)
        .extrude(height)
    )
    # Ranuras radiales para el bobinado
    for i in range(slots):
        angle = math.radians(i * slot_angle)
        stator = stator.cut(
            cq.Workplane("XY")
            .transformed(rotate=(0, 0, math.degrees(angle)))
            .rect(tooth_w, od_mm / 2 - id_mm / 2)
            .extrude(height)
            .translate((0, (od_mm + id_mm) / 4, 0))
        )
    return stator

def bldc_rotor_bell(poles: int = 14, od_mm: float = 36, id_mm: float = 31,
                    height: float = 12, magnet_w: float = 5, magnet_h: float = 3):
    """Campana exterior del rotor con ranuras para imanes N35."""
    bell = (
        cq.Workplane("XY")
        .circle(od_mm / 2)
        .circle(id_mm / 2)
        .extrude(height)
    )
    # Ranuras para imanes de neodimio
    for i in range(poles):
        angle = 360 / poles * i
        bell = bell.cut(
            cq.Workplane("XY")
            .transformed(rotate=(0, 0, angle))
            .rect(magnet_w, magnet_h)
            .extrude(height)
            .translate((0, (od_mm + id_mm) / 4, 0))
        )
    return bell

stator = bldc_stator()
rotor  = bldc_rotor_bell()
cq.exporters.export(stator, "bldc_stator_12N.stl")
cq.exporters.export(rotor,  "bldc_rotor_14P.stl")
```

---

#### 1.5.5 Imanes de neodimio — tipos y manejo

| Tipo | Grado | Br (T) | Hc (kA/m) | Temp. máx. | Uso |
|------|-------|--------|-----------|-----------|-----|
| **N35** | NdFeB | 1.17–1.22 | ≥ 868 | 80°C | Motor pequeño, bajo costo |
| **N52** | NdFeB | 1.42–1.48 | ≥ 876 | 80°C | Mayor densidad de energía |
| **N35SH** | NdFeB-SH | 1.17 | — | 150°C | Motor en ambiente caliente |
| **SmCo** | Samario-Cobalto | 1.05–1.15 | — | 300°C | Temperatura extrema, caro |

**Formas de imanes para motores:**

| Forma | Uso |
|-------|-----|
| **Disco Ø × h** | Rotor axial, pequeños motores |
| **Bloque rectangular** | Rotor BLDC pegado internamente |
| **Arco/segmento** | Rotor BLDC premium (campo más uniforme) |
| **Anillo** | Estátor de imán permanente |

**Pegado de imanes al rotor:**
```
1. Limpiar superficie con IPA 99%
2. Usar epoxy bicomponente de alta temperatura (Loctite EA 9466)
3. Alternar polaridades: N-S-N-S alrededor del rotor
   (verificar con brújula antes de pegar)
4. Jig de alineación: imprimir en 3D para posicionamiento preciso
5. Curar 24h bajo presión (pinzas o cinta temporal)
6. Verificar con gaussímetro o app de smartphone (menor precisión)
```

---

#### 1.5.6 Test y caracterización del motor fabricado

```python
import numpy as np
import matplotlib.pyplot as plt

def characterize_motor(rpm_data: list, voltage_data: list,
                       current_data: list, torque_data: list):
    """
    Genera curvas características del motor fabricado.
    Datos obtenidos con: dinamómetro DIY (celda de carga + encoder).
    """
    kv_measurements = [r / v for r, v in zip(rpm_data, voltage_data)]
    kv_avg = np.mean(kv_measurements)

    efficiency = [
        (t * r * 2 * np.pi / 60) / (v * i)
        for t, r, v, i in zip(torque_data, rpm_data, voltage_data, current_data)
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(current_data, torque_data, 'b-o')
    axes[0].set(xlabel='Corriente (A)', ylabel='Torque (N·m)',
                title=f'Curva T-I | Kt = {0.60/kv_avg:.4f} N·m/A')

    axes[1].plot(torque_data, rpm_data, 'r-o')
    axes[1].set(xlabel='Torque (N·m)', ylabel='RPM',
                title=f'Curva mecánica | Kv = {kv_avg:.1f} RPM/V')

    axes[2].plot(torque_data, [e*100 for e in efficiency], 'g-o')
    axes[2].set(xlabel='Torque (N·m)', ylabel='Eficiencia (%)',
                title=f'η_max = {max(efficiency)*100:.1f}%')

    plt.tight_layout()
    plt.savefig('motor_characterization.png', dpi=150)
    return {"kv": round(kv_avg, 1), "efficiency_max_pct": round(max(efficiency)*100, 1)}
```

---

#### 1.5.7 BOM: Kit de construcción de motor BLDC

| Componente | Qty | Costo aprox. | Fuente |
|------------|-----|-------------|--------|
| Imanes N35 bloque 5×5×3 mm | 14 | $3 | AliExpress |
| Alambre Cu esmaltado 0.5 mm, 50 g | 1 | $4 | Electrónica local |
| Rodamientos 608ZZ | 2 | $2 | AliExpress |
| Eje de acero 8 mm × 60 mm | 1 | $1.5 | Ferretería |
| Filamento PETG (estátor + rotor) | ~80 g | $2 | — |
| Epoxy alta temperatura | 1 tubo | $5 | Ferretería |
| ESC 30A (para test con BLDC) | 1 | $8 | AliExpress |

**Total: ~$25 por kit de motor completo**

---

## Módulo 2 — Cinemática
*Modelar el movimiento antes de controlarlo*

### 2.1 Representación de Cuerpos Rígidos
- Matrices de rotación SO(3): propiedades y operaciones en `numpy`
- Cuaterniones unitarios: evitar el gimbal lock, interpolación SLERP
- Transformaciones homogéneas SE(3)
- Cuaterniones duales: representación compacta de postura completa

### 2.2 Cinemática de Robots Móviles
- **Diferencial**: modelo cinemático, radio efectivo, odometría
- **Omnidireccional 3 ruedas**: jacobiana omnidireccional, control de dirección
- **Omnidireccional 4 ruedas (mecanum)**: análisis de deslizamiento
- Odometría: estimación de pose sin GPS con `numpy`
- Simulación 3D de trayectorias con `matplotlib` y `vpython`

### 2.3 Cinemática de Manipuladores
- Parámetros Denavit-Hartenberg (DH): convención estándar y modificada
- Cinemática directa: producto de matrices DH con `robotics-toolbox-python`
- Cinemática inversa numérica: realimentación del error (axis-angle)
- Cinemática inversa con cuaterniones duales
- Jacobiana geométrica y analítica
- Singularidades: detección con determinante y condición de la jacobiana

### 2.4 Planificación de Trayectorias
- Polinomios de grado 3 y 5 para movimiento suave
- Splines cúbicas en espacio de juntas (Joint Space)
- Trayectorias en espacio cartesiano (Task Space)
- Perfiles trapezoidal y de jerk mínimo
- Implementación en Rust para ejecución en tiempo real en ESP32

---

## Módulo 3 — Dinámica y Control
*De la física al torque exacto en cada motor*

### 3.1 Dinámica de Robots
- Formulación Euler-Lagrange: energía cinética y potencial
- Matrices de inercia, Coriolis y gravedad
- Modelo dinámico de robot diferencial y manipulador 2-DOF
- Algoritmo recursivo Newton-Euler para manipuladores N-DOF
- Implementación numérica en Python con `numpy`

### 3.2 Control Clásico
- PID discreto: implementación en Rust sobre ESP32 con control de tasa
- Sintonización automática: método Lambda y Ziegler-Nichols
- Control en cascada (velocidad + posición)
- Anti-windup y limitación de salida
- Control computed-torque (par calculado) con modelo dinámico

### 3.3 Control Avanzado
- Control con incertidumbre: control robusto H∞ básico
- Control adaptativo: estimación online de parámetros
- Control en espacio cartesiano vs. espacio de juntas
- Control de fuerza/impedancia: interacción robot-entorno
- Model Predictive Control (MPC) básico con `do-mpc`

### 3.4 Control en Tiempo Real con Rust
- Loop de control a frecuencia fija con `embassy` (async embebido)
- Comunicación entre tareas FreeRTOS: queues tipadas en Rust
- Watchdog y recuperación de fallos en firmware
- Perfilado de tiempo de ejecución con `defmt` y `probe-rs`

---

## Módulo 4 — Localización y Navegación
*El robot sabe dónde está y cómo llegar a cualquier punto*

### 4.1 Fusión de Sensores
- Filtro de Kalman Extendido (EKF): fusión IMU + odometría
- Filtro de partículas (MCL) para localización probabilística
- Implementación en Python con `filterpy`
- Comparativa: EKF vs. UKF vs. partículas en escenarios reales

### 4.2 SLAM — Simultaneous Localization and Mapping
- SLAM 2D con `cartographer` y `gmapping` (ROS2)
- Graph-SLAM: optimización de poses con `g2o` / `gtsam`
- Visual SLAM básico: ORB-SLAM3 como referencia
- SLAM con LiDAR en entornos estructurados

### 4.3 Planificación de Rutas
- Algoritmo A* sobre mapa de ocupación (Python)
- RRT y RRT* para espacios de alta dimensión
- D* Lite para replanificación dinámica
- Potential Fields: navegación reactiva simple
- Implementación embebida de A* en Rust para ESP32

### 4.4 Navegación Autónoma Completa
- Stack: percepción → localización → planificación → control → actuación
- Evitación dinámica de obstáculos (Dynamic Window Approach)
- Integración con ROS2 Nav2
- micro-ROS en ESP32 para recibir comandos de velocidad de ROS2

---

## Módulo 5 — Visión Artificial
*El robot ve, interpreta y actúa sobre lo que percibe*

### 5.1 Fundamentos de Visión con OpenCV
- Calibración de cámara: modelo pinhole, distorsión, `cv2.calibrateCamera`
- Transformaciones geométricas: homografía, perspectiva
- Detección de bordes, contornos y formas
- Tracking de color HSV y centroide de objeto

### 5.2 Visión Profunda con PyTorch
- CNN para clasificación de objetos en escena robótica
- Detección de objetos: YOLOv8 con `ultralytics`
- Segmentación semántica: SAM (Segment Anything Model)
- Estimación de pose 6D: FoundPose / MegaPose
- Exportar modelos a ONNX para inferencia ligera

### 5.3 Control Visual (Visual Servoing)
- IBVS (Image-Based Visual Servoing): jacobiana de imagen
- PBVS (Position-Based Visual Servoing): pose 3D → control
- Seguimiento de marcadores ArUco con OpenCV
- Pipeline completo: cámara ESP32-CAM → Python → control ESP32

### 5.4 Percepción 3D
- Nube de puntos con cámara de profundidad (Intel RealSense, OAK-D)
- Open3D: filtrado, registro ICP, reconstrucción de superficies
- Detección de planos y objetos en 3D
- Fusión LiDAR + cámara (calibración extrínseca)

---

## Módulo 6 — Inteligencia Artificial para Robótica
*Del modelo matemático al comportamiento aprendido*

### 6.1 Aprendizaje por Refuerzo (RL)
- Fundamentos: MDP, política, recompensa, valor
- Entornos de simulación: Gymnasium, PyBullet, Isaac Gym
- Algoritmos: PPO, SAC, TD3 con `stable-baselines3`
- Sim-to-real transfer: domain randomization, adaptación
- RL para control de manipuladores y navegación

### 6.2 Aprendizaje por Imitación
- Behavioral Cloning: red neuronal sobre demostraciones humanas
- DAgger: corrección iterativa de la política
- Captura de demostraciones con teleoperation sobre ESP32
- Fine-tuning de modelos preentrenados para tareas específicas

### 6.3 Modelos de Lenguaje en Robótica
- Instrucciones en lenguaje natural → acciones del robot
- LLM como planificador de tareas de alto nivel (LLM + Skills)
- Vision-Language Models (VLM): describir la escena → decidir acción
- Integración con `langchain` y llamadas a funciones de control

### 6.4 Inferencia en el Edge
- Cuantización de modelos: INT8, FP16 con `torch.quantization`
- ONNX Runtime en Python para inferencia sin GPU
- TFLite / Edge Impulse en ESP32 para modelos ligeros
- Pipeline: entrenamiento en GPU → exportar ONNX → ejecutar en ESP32

---

## Módulo 7 — Integración y Producción
*Del prototipo al sistema robusto*

### 7.1 ROS2 — Robot Operating System 2
- Nodos, tópicos, servicios y acciones en Python
- `rclrs`: cliente ROS2 oficial en Rust
- micro-ROS en ESP32 (Rust): publicar sensores, suscribir comandos
- TF2: árbol de transformaciones del robot
- URDF: descripción del robot para simulación y visualización

### 7.2 Simulación y Digital Twin
- Gazebo Harmonic: modelado URDF + plugins de sensores
- Webots: alternativa multiplataforma, buena para educación
- Hardware-in-the-loop (HIL): ESP32 real + mundo simulado
- Digital twin: replicar estado del robot real en simulación en tiempo real

### 7.3 Rust en Producción Robótica
- FFI Rust-Python: exponer algoritmos Rust como módulos Python (`PyO3`)
- `nalgebra`: álgebra lineal de alto rendimiento para cinemática en Rust
- Comparativa de rendimiento: Rust vs. Python en loops de control
- Gestión de memoria predecible: latencias deterministas en Rust

### 7.4 Calidad y Despliegue
- Testing embebido con `defmt-test` en ESP32 real
- CI/CD para firmware: GitHub Actions + `probe-rs` + HIL
- Monitoreo en producción: métricas de telemetría vía MQTT → Grafana
- Seguridad: TLS en MQTT, firma de firmware, secure boot ESP32

---

## Módulo 8 — Fase Integradora: Diseño 3D para Robótica
*Del modelo cinemático a la pieza física: Blender, impresión 3D y corte láser*

> Este módulo cierra el ciclo completo: las dimensiones del robot salen del modelo
> matemático (M2), las piezas se diseñan en software, se validan en simulación (M7)
> y se fabrican antes de ensamblar electrónica (M1) y código (M3–M6).

### 8.1 CAD Paramétrico con Python — CadQuery y Build123d
- **CadQuery**: diseño de piezas mecánicas como código Python puro
- **Build123d**: sucesor moderno, API más expresiva, compatible con STEP/STL/DXF
- Ventaja clave: las piezas viven en git, se generan desde parámetros del modelo cinemático
- Exportar STL (impresión), STEP (intercambio CAD), DXF (corte láser) desde un mismo script
- Biblioteca de piezas reutilizables: ruedas, soportes de motor, monturas de sensor, carcasas ESP32
- Generación procedural: un script que genera toda la familia de piezas de un robot

```python
# Ejemplo: soporte de motor parametrizable
import build123d as bd

def motor_mount(motor_d: float, wall: float = 3.0):
    with bd.BuildPart() as mount:
        bd.Cylinder(radius=motor_d/2 + wall, height=20)
        bd.Cylinder(radius=motor_d/2, height=20, mode=bd.Mode.SUBTRACT)
    return mount.part
```

### 8.2 Blender para Robótica
- Interfaz esencial: mesh, modifier stack, boolean, bevel
- Modelado mecánico de eslabones, articulaciones y chasis
- **Addon Phobos**: exportar modelo Blender → URDF completo para ROS2 y Gazebo
  - Definición de joints, links, masas e inercias directamente en Blender
  - Mallas visuales (alta resolución) vs. mallas de colisión (convex hull simplificado)
- Rigging de robot: armatures para animar y visualizar cinemática
- Renderizado técnico: documentación, presentaciones, portfolio
- Workflow Blender → Gazebo: validar el URDF en simulación antes de imprimir una sola pieza

### 8.3 OpenSCAD — CAD Scriptable (complemento)
- Primitivas, diferencias, uniones e intersecciones como operaciones de código
- Módulos reutilizables: biblioteca de engranajes, correas, ruedas dentadas
- Generación desde Python: escribir `.scad` programáticamente con `solidpython2`
- Cuándo elegir OpenSCAD vs. CadQuery: simplicidad vs. potencia paramétrica

### 8.4 Diseño para Impresión 3D
> **Restricción de producción: volumen máximo 200 × 200 × 200 mm.**
> Piezas más grandes se parten en segmentos con ensamble por pins o encajes.

**Materiales disponibles en el curso:**
| Material | Uso en los kits | T° cama | Notas |
|----------|----------------|---------|-------|
| PLA | Mock-ups, prototipos rápidos | 60°C | Fácil, frágil bajo carga |
| PETG | Estructura principal de todos los robots | 80°C | Balance óptimo para kits |
| TPU 95A | Ruedas, gripper, amortiguadores | 45°C | Flexible, grip excelente |
| PA (Nylon) | Engranajes, articulaciones de alta carga | 90°C | Para kits avanzados |

**Reglas de diseño dentro del volumen 200 × 200 × 200 mm:**
- Tolerancias: agujeros pasantes +0.2 mm, encajes de presión −0.1 mm
- Orientación de impresión: capas perpendiculares a la carga principal
- Voladizos: máximo 45° sin soporte — rediseñar si supera eso
- Costillas y nervios: grosor = 0.6× la pared principal
- **Heat inserts** M3/M4: uniones desmontables profesionales con soldador de punta fina
- **Snap fits** y press fits: diseño con deflexión calculada en CadQuery
- Relleno: gyroid 25% para carga, rectilinear 40% para precisión dimensional

**Estrategia para piezas que superan 200 mm:**
- Partir en segmentos con solapamiento de 15–20 mm
- Unión con pins M3 + heat insert o adhesivo estructural (Loctite 425)
- CadQuery genera los cortes automáticamente según el parámetro `max_dim=200`

**Piezas fabricadas en cada kit:**
- Chasis diferencial (si < 200 mm) → impresión; si no → láser MDF
- Eslabones de brazo: cada eslabón cabe en 200 × 200 × 200 mm (diseño modular)
- Ruedas: TPU, Ø60–80 mm, impresión vertical para resistencia al desgaste
- Gripper: PETG + TPU (dedos flexibles), conjunto < 120 × 80 × 60 mm
- Carcasa ESP32 + electrónica: PETG, < 120 × 80 × 40 mm

**Slicing profesional con PrusaSlicer:**
- Perfiles guardados por material y función (kit_petg_structural, kit_tpu_wheel...)
- Modifier meshes: relleno local 80% en zonas de estrés (monturas de motor)
- Pausas para insertar heat inserts o magnetos en mitad de impresión
- Estimación de masa exportada → actualiza automáticamente el modelo dinámico en Python

### 8.5 Diseño para Corte Láser
> **Restricción de producción: área máxima de hoja 400 × 400 mm.**
> Un chasis de robot diferencial ocupa ~300 × 250 mm → caben 2 por hoja.

**Conceptos fundamentales:**
- **Kerf en nuestra máquina**: 0.15–0.25 mm según material → compensar en DXF
- Velocidad y potencia configuradas por material en LightBurn (perfiles guardados)

**Materiales disponibles en los kits:**
| Material | Espesor | Uso en kits | Piezas por hoja 400×400 |
|----------|---------|-------------|------------------------|
| MDF | 4 mm | Chasis principal, estructuras | 2 chasis completos |
| Acrílico | 3 mm | Paneles superiores, estético | 3–4 paneles |
| Triplay (birch) | 4 mm | Estructural ligero (brazo, AGV) | 2–3 piezas grandes |
| Cartón | 3 mm | Mock-up rápido antes de cortar MDF | 4–6 prototipos |

**Técnicas de ensamble usadas en los kits:**
- **Finger joints**: encaje preciso en esquinas, sin pegamento — calibrado para kerf real
- **T-slot + tuerca M3**: desmontable, reutilizable entre proyectos
- **Press fit layered**: capas apiladas + pines para estructuras 3D dentro de 400×400
- **Living hinges**: tapa de acceso a electrónica en una sola pieza (no necesita bisagra)

**Diseño de chasis tipo "sandwich" (estándar de los kits):**
- **Capa base** (MDF 4mm): monturas de motor, soporte de encoders, guía de cables
- **Capas medias** (MDF 4mm): bandeja de electrónica (ESP32, driver, batería)
- **Capa superior** (acrílico 3mm): soporte de sensores, tapa transparente

> Nota de producción: todos los DXF del curso están calibrados con kerf = 0.2 mm
> para nuestra máquina. Si el alumno tiene otra cortadora, CadQuery regenera el DXF
> con el kerf correcto en un parámetro.

**Software del flujo de producción:**
- **CadQuery / Build123d** → genera DXF con kerf compensado automáticamente
- **Inkscape**: ajustes visuales finales, agrupación de piezas en la hoja 400×400
- **LightBurn**: control de la máquina, perfiles de corte guardados por material

### 8.6 Integración FreeCAD como CAD Mecánico Completo
- FreeCAD con Python API: automatizar generación de piezas desde scripts
- Módulo Assembly: ensambles completos con restricciones
- Análisis FEM básico: verificar resistencia de piezas antes de imprimir
- Exportar a Blender para visualización y a Gazebo para simulación

### 8.7 Flujo de Trabajo Completo: Diseño → Simulación → Fabricación

```
1. Modelo cinemático (Python + robotics-toolbox)
        │  dimensiones exactas de eslabones, ejes, distancias
        ▼
2. CAD paramétrico (CadQuery/Build123d en Python)
        │  STL para impresión · DXF para láser · STEP para referencia
        ▼
3. URDF automático (Blender + Phobos)
        │  masas e inercias desde el CAD
        ▼
4. Validación en simulación (Gazebo / Webots)
        │  cinemática, colisiones, sensores — sin imprimir nada aún
        ▼
5. Fabricación
        ├── Impresión 3D: eslabones, brackets, carcasas (PETG/TPU)
        └── Corte láser: chasis, paneles, estructura (MDF/acrílico)
        ▼
6. Ensamble + electrónica (ESP32, motores, sensores)
        ▼
7. Calibración física vs. modelo → ajustar parámetros dinámicos
```

### 8.8 Diseño de Robots Completos — Ejercicios de Proyecto
- **Robot diferencial**: chasis cortado a láser + ruedas impresas en TPU + brackets en PETG
- **Brazo 4-DOF**: eslabones en PETG + articulaciones con rodamientos + gripper TPU
- **Rover omnidireccional**: chasis sandwich láser + ruedas mecanum impresas
- Cable management integrado: canales en diseño, sin cables sueltos
- BOM (Bill of Materials) generado automáticamente desde CadQuery

---

### 8.9 Engranajes para Robótica — Teoría, Diseño e Impresión 3D

> Los engranajes multiplican torque, reducen velocidad y transmiten fuerza entre ejes.
> Diseñarlos con Python y fabricarlos en nuestra impresora es el cierre perfecto entre
> el cálculo mecánico y la pieza física.

---

#### 8.9.1 Conceptos fundamentales de engranajes

**Terminología clave:**

| Parámetro | Símbolo | Definición | Fórmula |
|-----------|---------|-----------|---------|
| **Módulo** | m | Tamaño relativo del diente | m = d / z |
| **Número de dientes** | z | Cuántos dientes tiene el engrane | — |
| **Diámetro primitivo** | d | Círculo de rodadura | d = m × z |
| **Diámetro exterior** | da | Círculo de cabeza | da = m × (z + 2) |
| **Diámetro interior** | df | Círculo de pie | df = m × (z − 2.5) |
| **Paso circular** | p | Longitud de arco entre dientes | p = π × m |
| **Ángulo de presión** | α | Ángulo de contacto normal | 20° estándar |
| **Razón de transmisión** | i | Relación de velocidades | i = z₂ / z₁ = ω₁ / ω₂ |
| **Distancia entre centros** | a | Distancia entre ejes | a = m × (z₁ + z₂) / 2 |

**Relación torque-velocidad:**
```
Torque salida  = Torque entrada × (z₂ / z₁) × η
Velocidad salida = Velocidad entrada × (z₁ / z₂)

donde η = eficiencia mecánica del par (0.95–0.98 engranajes rectos)
```

---

#### 8.9.2 Tipos de engranajes en robótica

| Tipo | Forma | Ejes | Eficiencia | Uso robótico |
|------|-------|------|-----------|-------------|
| **Recto (spur)** | Dientes paralelos al eje | Paralelos | 97–99 % | Caja reductora brazo, robot móvil |
| **Helicoidal** | Dientes en espiral | Paralelos | 96–98 % | Menos ruido, mayor carga |
| **Cónico** | Dientes en cono | Perpendiculares | 95–97 % | Cambio de dirección 90° |
| **Corona-sinfín (worm)** | Tornillo + rueda | 90° cruzados | 50–90 % | Autofreno, alta reducción en poco espacio |
| **Cremallera** | Lineal | Rotación → traslación | 95 % | Actuadores lineales, impresoras 3D |
| **Planetario** | Satélites + corona | Coaxiales | 97–98 % | Máxima reducción compacta (jointas brazo) |

---

#### 8.9.3 Diseño de engranajes en Python con CadQuery

```python
import cadquery as cq
import math

def involute_point(r_base: float, t: float) -> tuple[float, float]:
    """Punto en la curva evolvente del círculo base."""
    x = r_base * (math.cos(t) + t * math.sin(t))
    y = r_base * (math.sin(t) - t * math.cos(t))
    return x, y

def spur_gear(module: float, teeth: int, thickness: float,
              bore_d: float = 5.0, pressure_angle: float = 20.0) -> cq.Workplane:
    """
    Genera un engranaje recto con perfil evolvente.
    module      : módulo del engranaje (ej. 1.0, 1.5, 2.0)
    teeth       : número de dientes
    thickness   : espesor del engrane en mm
    bore_d      : diámetro del agujero central en mm
    """
    r_pitch   = module * teeth / 2           # radio primitivo
    r_base    = r_pitch * math.cos(math.radians(pressure_angle))
    r_addend  = r_pitch + module             # radio de cabeza
    r_dedend  = r_pitch - 1.25 * module     # radio de pie

    # Generar perfil de UN diente usando puntos evolventes
    n_pts = 20
    t_max = math.sqrt((r_addend / r_base) ** 2 - 1)
    pts_right = [involute_point(r_base, t * t_max / n_pts) for t in range(n_pts + 1)]

    # Reflexión para el flanco izquierdo
    pitch_angle = 2 * math.pi / teeth
    half_tooth  = math.pi / teeth + math.atan(
        math.sqrt((r_addend / r_base)**2 - 1) - math.acos(r_base / r_addend)
    )
    pts_left = [
        (x * math.cos(-2 * half_tooth) - y * math.sin(-2 * half_tooth),
         x * math.sin(-2 * half_tooth) + y * math.cos(-2 * half_tooth))
        for x, y in pts_right
    ]

    # Construir perfil de diente y extruir
    tooth_profile = (
        cq.Workplane("XY")
        .spline(pts_right)
        .lineTo(pts_left[-1][0], pts_left[-1][1])
        .spline(list(reversed(pts_left)))
        .close()
    )

    # Ensamblar todos los dientes en patrón polar
    gear = (
        cq.Workplane("XY")
        .circle(r_dedend)
        .extrude(thickness)
    )
    # Nota: en producción usar la librería `cq-gears` que maneja el perfil completo
    return gear.circle(bore_d / 2).cutThruAll()


def gear_train(module: float, ratio: float, pinion_teeth: int,
               thickness: float) -> dict:
    """
    Diseña un par de engranajes con la razón deseada.
    Retorna diccionario con geometría de ambos engranajes y distancia entre centros.
    """
    wheel_teeth = round(pinion_teeth * ratio)
    actual_ratio = wheel_teeth / pinion_teeth
    center_distance = module * (pinion_teeth + wheel_teeth) / 2

    return {
        "pinion_teeth":    pinion_teeth,
        "wheel_teeth":     wheel_teeth,
        "actual_ratio":    actual_ratio,
        "center_distance": center_distance,
        "pinion_od":       module * (pinion_teeth + 2),
        "wheel_od":        module * (wheel_teeth + 2),
        "pitch_diameter":  {"pinion": module * pinion_teeth,
                            "wheel":  module * wheel_teeth},
    }

# Ejemplo: reducción 5:1 para articulación de brazo robótico
params = gear_train(module=1.5, ratio=5.0, pinion_teeth=12, thickness=8.0)
print(f"Engranaje conductor: z={params['pinion_teeth']}, Ø={params['pinion_od']} mm")
print(f"Engranaje conducido: z={params['wheel_teeth']}, Ø={params['wheel_od']} mm")
print(f"Distancia entre centros: {params['center_distance']} mm")
print(f"Razón real: {params['actual_ratio']:.3f}:1")

# Verificar que caben en la impresora (Ø < 200 mm)
assert params['wheel_od'] < 200, "El engranaje no cabe en la impresora 3D"
```

**Librería recomendada: cq-gears**
```bash
pip install cq-gears
```
```python
from cq_gears import SpurGear, HerringboneGear, WormGear, RingGear

# Engranaje recto estándar
gear = SpurGear(module=1.5, teeth_number=24, width=8.0, bore_d=5.0)
solid = gear.build()
cq.exporters.export(solid, "gear_z24_m15.stl")

# Engranaje para impresión: módulo ≥ 1.0 (dientes imprimibles a 0.2 mm layer)
# Módulo 1.5 → paso = 4.71 mm → detalle imprimible a 0.15 mm layer
```

---

#### 8.9.4 Selección de módulo para impresión 3D

| Módulo | Paso (mm) | Diente mínimo (mm) | Layer recomendado | Aplicación |
|--------|-----------|-------------------|------------------|-----------|
| 0.5 | 1.57 | 0.4 | Imposible en FDM | Solo SLA/resina |
| 1.0 | 3.14 | 0.8 | 0.10 mm | Mecanismo ligero, servo |
| **1.5** | **4.71** | **1.2** | **0.15 mm** | **Uso general en el curso** |
| 2.0 | 6.28 | 1.6 | 0.20 mm | Carga media, reducción robot |
| 3.0 | 9.42 | 2.4 | 0.20 mm | Alta carga, AGV |

**Configuración de slicer para engranajes (Bambu Studio / PrusaSlicer):**
```
Material: PETG (mejor resistencia al desgaste que PLA)
Capa:     0.15 mm
Perímetros: 4 (paredes gruesas para resistencia del diente)
Relleno:  40 % Gyroid (absorbe impacto)
Velocidad: 40 mm/s (precisión dimensional)
Orientación: eje del engranaje vertical (capas perpendiculares a la carga)
Tolerancia de agujero: +0.2 mm en diámetro (ajuste con calibración previa)
```

---

#### 8.9.5 Diseño de cajas reductoras para el curso

**Reducción en etapas para brazo robótico (articulación del codo):**
```python
# Reducción total 25:1 en dos etapas (5:1 × 5:1)
# Ventaja: engranajes más pequeños → caben en impresora

etapa_1 = gear_train(module=1.5, ratio=5.0, pinion_teeth=12, thickness=8.0)
etapa_2 = gear_train(module=1.5, ratio=5.0, pinion_teeth=12, thickness=8.0)

print(f"Reducción total: {etapa_1['actual_ratio'] * etapa_2['actual_ratio']:.1f}:1")
# Diámetro máximo de engranaje: 1.5 × (12×5 + 2) = 96 mm ✅ (< 200 mm)
```

**Tipos de cajas reductoras en el curso:**

| Tipo | Etapas | Reducción típica | Dónde usarla en el robot |
|------|--------|-----------------|--------------------------|
| **Par simple** | 1 | 2:1 – 8:1 | Rueda de robot diferencial |
| **Tren de engranajes** | 2–3 | 10:1 – 100:1 | Articulación de brazo |
| **Planetaria impresa** | 1 | 4:1 – 12:1 | Junta compacta, coaxial |
| **Sinfín impreso** | 1 | 10:1 – 60:1 | Autofreno (gripper, elevador) |
| **Correa GT2** | 1 | 1:1 – 5:1 | Eje con distancia (eje de robot) |

---

#### 8.9.6 Engranajes planetarios impresos

```python
from cq_gears import PlanetaryGearSet

# Sistema planetario: sol + 3 planetas + corona
planetary = PlanetaryGearSet(
    module         = 1.5,
    sun_teeth      = 12,
    planet_teeth   = 12,
    n_planets      = 3,
    width          = 10.0,
    bore_d         = 6.0,
)
# Reducción: (corona / sol) + 1 = (36/12) + 1 = 4:1
assembly = planetary.build()
cq.exporters.export(assembly["sun"],    "sun_gear.stl")
cq.exporters.export(assembly["planet"], "planet_gear.stl")
cq.exporters.export(assembly["ring"],   "ring_gear.stl")
```

**Restricción de la impresora (200×200×200 mm):**
- Corona máxima: Ø = módulo × (sol + 2×planeta + 2) × 2 ≤ 200 mm
- Con módulo 1.5 y 36 dientes de corona: Ø = 1.5 × 38 = 57 mm ✅

---

#### 8.9.7 Lubricación, tolerancias y durabilidad

**Tolerancias de juego (backlash) para impresión FDM:**
```
Backlash recomendado: 0.1–0.2 mm por lado del diente
Cómo aplicarlo en CadQuery:
  - Reducir el radio primitivo del conducido en 0.1 mm
  - O aumentar la distancia entre centros en 0.1 mm
  - Calibrar con dos dientes impresos de prueba antes del conjunto final
```

**Lubricación:**
| Material | Lubricante recomendado | Frecuencia |
|----------|----------------------|-----------|
| PETG–PETG | Grasa de litio blanca | Cada 50 h |
| PETG–aluminio | Aceite de silicona | Cada 20 h |
| PLA–PLA | Sin lubricante (autolubricante) | — |
| PETG–POM | Sin lubricante (POM es autolubricante) | — |

**Resistencia al desgaste:**
- PLA: blanda, para prototipos y prueba de concepto
- PETG: uso general, temperatura < 70°C
- ASA / ABS: exposición exterior o alta temperatura (hasta 90°C)
- POM (Delrin): el más resistente para engranajes industriales (pedir piezas mecanizadas)
- Metálicos (impresos en Markforged / DMLS): sólo para producción o alta carga

---

#### 8.9.8 Proyecto: caja reductora para brazo robótico

**Especificación:**
- Motor: N20 DC 6V 600 RPM
- Reducción objetivo: 10:1 → salida ≈ 60 RPM, torque × 10
- Módulo: 1.5 (imprimible a 0.15 mm layer)
- Material: PETG

```python
# Diseño automatizado
specs = gear_train(module=1.5, ratio=10.0, pinion_teeth=10, thickness=6.0)
# → pinion z=10, wheel z=100, centro = 82.5 mm
# ⚠️ wheel_od = 1.5×102 = 153 mm → ✅ cabe en 200×200

# Carcasa en CadQuery
caja = (
    cq.Workplane("XY")
    .box(170, 170, 20)  # caben ambos engranajes + 8 mm de pared
    .shell(-2.5)         # 2.5 mm de pared
    .faces(">Z").hole(specs["pinion_od"] / 2 + 2)  # ventana de acceso
)
cq.exporters.export(caja, "gearbox_housing.stl")
```

**BOM del proyecto engranajes:**

| Pieza | Qty | Fabricación | Material | Dimensión | Tiempo impresión |
|-------|-----|-------------|----------|-----------|-----------------|
| Engranaje conductor (z=10) | 1 | Impresora 3D | PETG | Ø17 × 6 mm | 15 min |
| Engranaje conducido (z=100) | 1 | Impresora 3D | PETG | Ø153 × 6 mm | 3.5 h |
| Carcasa inferior | 1 | Impresora 3D | PETG | 170×170×10 mm | 2 h |
| Carcasa superior | 1 | Impresora 3D | PETG | 170×170×12 mm | 2.5 h |
| Rodamiento 608ZZ | 2 | Comprado | Acero | Ø22×Ø8×7 mm | — |
| Eje de 8 mm | 1 | Comprado | Acero | Ø8×120 mm | — |
| Eje de 5 mm | 1 | Comprado | Acero | Ø5×80 mm | — |

---

## Módulo 9 — Aplicaciones Robóticas por Dominio
*Todo el stack aprendido aterriza en robots reales con propósito real*

> Cada sección aplica módulos anteriores a un dominio específico.
> No son ejercicios académicos: son los robots que existen en el mercado hoy.

---

### 9.1 Robótica de Competición — Catálogo Completo

> Mapeado al stack del curso: ESP32 (Rust) · Python · impresión 3D 200³ · corte láser 400×400.
> Cada categoría indica el kit base y los módulos del curso que aplican.

---

#### A — Combate y Batalla de Robots

La categoría más intensa del mundo amateur. El robot debe destruir, inmovilizar
o sacar del arena al oponente. Cero margen de error en firmware — un bug es perder.

**Clases de peso (estándar internacional):**

| Clase | Peso máx. | Notas para el curso |
|-------|----------|-------------------|
| Fairyweight | 150 g | 100% 3D imprimible en nuestras máquinas ✅ |
| **Antweight** | **453 g (1 lb)** | **Clase recomendada — todo entra en 200³** ✅ |
| **Beetleweight** | **1.36 kg (3 lbs)** | **Clase estrella del curso** ✅ |
| Hobbyweight | 5.4 kg (12 lbs) | Requiere piezas de acero/aluminio |
| Featherweight | 13.6 kg (30 lbs) | Fuera del scope de fabricación propia |
| Lightweight | 27 kg (60 lbs) | Profesional |
| Middleweight | 54 kg (120 lbs) | Semi-profesional / TV |
| Heavyweight | 110 kg (250 lbs) | BattleBots / Robot Wars / TV |

**Clase especial — Plastic Antweight / 3D Printed:**
Todas las piezas estructurales, armadura y arma deben ser 100% impresas en 3D.
Es exactamente lo que podemos hacer con nuestra impresora 200×200×200 mm.

**Tipos de armas:**

| Arma | Descripción | Dificultad de diseño |
|------|------------|---------------------|
| **Drum spinner** | Tambor horizontal de alta inercia | Media — eje + rodamiento |
| **Disc spinner** | Disco vertical, muy destructivo | Media |
| **Bar spinner** | Barra horizontal, área grande | Fácil de imprimir |
| **Shell spinner** | Cuerpo giratorio completo como arma | Alta |
| **Flipper eléctrico** | Lanzar al oponente al aire | Requiere MOSFET de alta corriente |
| **Flipper neumático** | CO₂ o aire comprimido | Complejo en Antweight |
| **Hammer / Axe** | Golpe vertical o diagonal | Servo de alto torque |
| **Thwackbot** | Cuerpo gira y la cola es el arma | Meltybrain algorithm |
| **Wedge / Rambot** | Sin arma activa, solo cuña baja | Más fácil, estrategia de empuje |
| **Crusher** | Pinza que aplasta | Control de fuerza preciso |
| **Walker** | Patas articuladas (+100% peso bonus) | Cinemática compleja |

**Materiales de armadura:**
- **UHMWPE** (Polietileno ultra alta densidad): imprimible, excelente anti-spinner
- **Policarbonato** (PC): transparente, absorbe impactos, imprimible
- **PETG**: estructura interna — sacrificial por diseño (cheap to reprint)
- **TPU**: capas exteriores que absorben impacto sin romperse
- Acero AR500 / Titanio 6Al4V: para Featherweight en adelante

**Sistemas de transmisión para combate:**
- **Brushless + ESC bidireccional**: máxima potencia por gramo
- **4WD tracción total**: imposible de voltear y mayor empuje
- **Meltybrain**: el robot gira sobre sí mismo como arma mientras se traslada

```rust
// Meltybrain: translación mientras el robot gira como spinner
// El heading se corrige con beacon óptico o IMU en cada vuelta completa
struct Meltybrain {
    gyro_z: f32,          // velocidad angular actual (rad/s)
    target_heading: f32,  // dirección de traslación deseada
    throttle: f32,
}

impl Meltybrain {
    fn compute_motor_mix(&self, current_angle: f32) -> (f32, f32) {
        let phase = current_angle - self.target_heading;
        let left  = self.throttle * (1.0 + phase.sin());
        let right = self.throttle * (1.0 - phase.sin());
        (left.clamp(0.0, 1.0), right.clamp(0.0, 1.0))
    }
}
```

**Electrónica de combate (Antweight/Beetleweight):**
- ESP32 + receptor ELRS: control con failsafe hardware (< 100 ms de timeout)
- ESC brushless 30A × 2 (tracción) + ESC 20A × 1 (arma)
- LiPo 2S 650mAh 75C: pico de corriente brutal durante impactos
- Armado por hardware: switch físico + LED de estado de arma
- Fusible reseteable PPTC en línea de arma

**Competencias relevantes:**
- **BattleBots** (USA): el más famoso, TV, Heavyweight
- **Robot Wars** (UK): formato similar
- **RoboGames** (San Francisco): 50+ categorías
- **Robotic People Fest** (Colombia): batalla 150g, 1L, 3L
- **Liga Robótica Argentina / LNR**: múltiples clases
- **Robomatrix** (LATAM): liga latinoamericana
- **Robot Challenge Colombia**: categorías de batalla

---

#### B — Sumo Robótico

El robot debe empujar al oponente fuera del dohyo (tatami circular)
sin salirse él mismo. Prohibido volar, usar armas o dañar al oponente.

**Categorías por peso:**

| Categoría | Peso máx. | Tamaño máx. | Tatami | Notas |
|-----------|----------|-------------|--------|-------|
| Micro Sumo | 100 g | 5×5 cm | 38 cm | Muy delicado |
| **Mini Sumo** | **500 g** | **10×10 cm** | **77 cm** | **Clase del curso Kit 2** ✅ |
| **Sumo Estándar** | **3 kg** | **20×20 cm** | **154 cm** | **Clásico** ✅ |
| Sumo Libre | Sin límite | Sin límite | Variable | Creatividad total |
| Sumo RC | 3 kg | 20×20 cm | 154 cm | Control manual |
| Sumo Humanoid | Robot bípedo | — | — | ROBO-ONE |

**Estrategias de búsqueda:**
```rust
enum SearchPattern { Spiral, Zigzag, PivotTurn, EdgeSweep }

fn search_opponent(pattern: SearchPattern, time_ms: u32) -> MotorCmd {
    match pattern {
        SearchPattern::Spiral    => spiral_outward(time_ms),
        SearchPattern::Zigzag    => zigzag(time_ms, angle_deg: 30.0),
        SearchPattern::PivotTurn => rotate_in_place(speed: 0.8),
        SearchPattern::EdgeSweep => sweep_perimeter(),
    }
}
```

**Detección del oponente:**
- IR de largo alcance (Sharp GP2Y0A21): 10–80 cm, 3 frontales + 2 laterales
- ToF VL53L1X: más preciso, hasta 400 cm, menor ángulo de apertura
- Ultrasonido HC-SR04: económico, 2–400 cm, más lento
- Cámara + visión (avanzado): OpenCV en ESP32-S3 con OV2640

**Competencias LATAM:**
- Liga Nacional de Robótica Argentina (LNR): Sumo Libre, Mini Sumo, Mega Sumo
- Robomatrix.org: circuito latinoamericano oficial
- WRO RoboDeportes: peso máx. 1.2 kg
- Universidad Autónoma de Occidente (Colombia): Sumo Libre
- Tecnológico de Monterrey (México): campeón LATAM 2025

---

#### C — Seguidor de Línea

Robot autónomo que sigue una línea (normalmente negra sobre blanco)
a la máxima velocidad posible. Categoría con la mayor cantidad de sub-categorías en LATAM.

**Niveles de dificultad:**

| Nivel | Descripción | Algoritmo clave |
|-------|------------|----------------|
| Básico | Línea simple, curvas suaves | PID clásico |
| Intermedio | Intersecciones T y +, cruces | Detección de intersección + mapa |
| Avanzado | Líneas múltiples, detección de color | PID adaptativo + visión |
| Velocista | Máxima velocidad, turbina de aire | Feed-forward + mapeo |
| Velocista persecución | Dos robots en la misma pista | Control de brecha (gap control) |
| 3D | Pista con rampas y cambios de plano | IMU + compensación de inclinación |

```python
# PID adaptativo con ganancia variable según velocidad
def adaptive_pid(error: float, speed: float) -> float:
    kp = 0.4 + speed * 0.3   # más agresivo en rectas rápidas
    ki = 0.001
    kd = 0.15 + speed * 0.1
    return kp * error + ki * integral + kd * derivative
```

**Sensores según nivel:**
- QTR-8RC (8 sensores IR): estándar, rápido, analógico+digital
- TCRT5000 × 5–8: económico, fácil de soldar
- Cámara + línea: para pistas complejas, visión computacional

**Categoría "velocista con turbina":**
- Motor de turbina de aeromodelismo empuja el robot al suelo (downforce)
- Permite curvar a velocidades que normalmente derraparían
- ESP32 controla turbina por PWM según velocidad estimada

---

#### D — Micromouse

Robot completamente autónomo que resuelve un laberinto 16×16 celdas (cada celda 18 cm)
y llega al centro en el menor tiempo posible. Competencia de referencia IEEE.

**Especificaciones del laberinto:**
- Dimensión: 16×16 celdas de 18×18 cm
- Paredes: 5 cm de altura, 1.2 cm de grosor
- Pasillo: 16.8 cm de ancho
- Objetivo: celda central (8,8) — 4 celdas combinadas

**Algoritmo Flood-Fill:**
```python
def flood_fill(maze: Maze, start: Cell, goal: Cell) -> list[Cell]:
    distances = {goal: 0}
    queue = deque([goal])
    while queue:
        cell = queue.popleft()
        for neighbor in maze.open_neighbors(cell):
            if neighbor not in distances:
                distances[neighbor] = distances[cell] + 1
                queue.append(neighbor)
    # Seguir gradiente descendente desde start hasta goal
    path, current = [start], start
    while current != goal:
        current = min(maze.open_neighbors(current), key=lambda c: distances[c])
        path.append(current)
    return path
```

**Hardware típico:**
- 4 motores DC o paso a paso con encoders de alta resolución
- 4 sensores IR a 45°: detectan paredes izquierda, derecha, frontal
- IMU para corrección de giro
- Baterías LiPo 1S / LiIon 18650

**Fases de carrera:**
1. **Exploración**: recorrer el laberinto mapeando paredes (lento pero completo)
2. **Speed run**: usar el mapa aprendido para el camino óptimo a máxima velocidad
3. **Record mundial**: bajo 3.5 segundos (top tier, requiere hardware de precisión)

---

#### E — FPV Drone Racing

Quadrotor diseñado para recorrer circuitos de puertas (gates) a máxima velocidad.

**Clases de competición:**

| Clase | Tamaño frame | Motor | Uso |
|-------|-------------|-------|-----|
| Toothpick | 2.5" | 1103 / 1202 | Indoor, ultra-ligero |
| **CineWhoop** | **3"** | **1404** | **Interior, seguro** |
| **Standard** | **5"** | **2306/2207** | **Clase principal** ✅ |
| Long-range | 7" | 2807 | Distancia, no velocidad |
| Heavy | 10"+ | 3110+ | Payload, trabajo |

**Stack de control (ESP32-S3 como FC):**
- Loop de rate: 8 kHz (giroscopio → PID → DShot → ESC)
- Filtros: RPM filter + Biquad notch (eliminar resonancias del frame)
- Telemetría: ExpressLRS CRSF bidireccional — batería, RSSI, velocidad
- Blackbox: log a flash SPI para análisis post-vuelo en Python

**Competencias:**
- **MultiGP**: liga mundial FPV, clasificatorias locales → mundiales
- **FAI World Drone Racing Championship**: federación internacional
- **DRL (Drone Racing League)**: semiprofesional, drones idénticos
- **Drone Soccer (FAI)**: 5v5, drones en esferas protectoras
- **AlphaPilot**: vuelo autónomo sin piloto humano

---

#### F — Soccer Robótico (RoboCup y afines)

**Ligas RoboCup:**

| Liga | Robots | Visión | Control | Nivel |
|------|--------|--------|---------|-------|
| Small Size (SSL) | 6 × Ø15cm | Cámara cenital | Centralizado PC | Universidad |
| Middle Size (MSL) | 5 × Ø40cm | Onboard 360° | Distribuido | Universidad |
| Standard Platform (SPL) | 5 NAO | Onboard | Onboard | Universidad |
| Humanoid KidSize | 4 bípedos | Onboard | Onboard | Universidad |
| Humanoid AdultSize | 2 bípedos | Onboard | Onboard | Profesional |
| Simulation 2D | Software | — | — | Investigación |
| Junior Soccer | 2 × pequeños | IR/Cámara | Onboard | Secundaria |

**SSL — Small Size League (más accesible):**
- Robots en zona azul/amarilla en campo 9×6 m
- Cámara cenital compartida → PC central → comandos WiFi a cada robot
- Predicción de trayectoria de pelota: filtro de Kalman + física en Python
- Control de velocidad y orientación: kinemática omnidireccional (4 ruedas)

---

#### G — Robots de Rescate

**RoboCup Rescue Maze:**
- Arena con colinas, pasos estrechos, víctimas a detectar (calor IR, visual)
- Mapa autónomo: SLAM con LiDAR y cámara
- Identificación de víctimas: termocámara (MLX90640) + visión YOLOv8

**RoboCup Rescue Simulation:**
- Entorno virtual en ROS2 / Gazebo
- Algoritmos de exploración: frontier-based exploration, RRT

**Trinity College Fire Fighting (TCFF):**
- Robot recorre habitaciones de una casa a escala
- Detecta y apaga una vela sin tocar las paredes
- Sensores: UV flame detector, termocámara, ultrasonido
- Actuador: mini ventilador o bomba de CO₂

---

#### H — Competencias FIRST, VEX y WRO

**FIRST Robotics Competition (FRC):**
- Robot de ~55 kg, construido en 6 semanas, campo de juego con alianzas 3v3
- Control: roboRIO (Java/Python/C++) — adaptable a ESP32 como coprocessor
- Game design cambia cada año (2026: Reefscape — manipulación de algas submarinas)
- Habilidades: mecánica, programación, visión, trabajo en equipo

**FIRST Tech Challenge (FTC):**
- Robot ≤ 18"×18"×18", control Android
- Más accesible que FRC, ideal para iniciarse
- Python y Java como lenguajes principales

**VEX V5 Robotics (VRC):**
- Campo 12×12 pies, robot ≤ 18"×18"×18"
- Control: VEX Brain con VEXcode (Blocks/C++/Python)
- **VEX AI**: versión con visión computacional — muy alineado con este curso

**WRO — World Robot Olympiad:**

| Categoría | Edad | Robot | Lenguaje |
|-----------|------|-------|----------|
| RoboMission Starter | 8–12 | Lego | Blocks |
| RoboMission Elementary | 10–13 | Lego | Blocks/Python |
| RoboMission Junior | 13–15 | Lego | Python |
| RoboMission Senior | 14–19 | Lego EV3/Spike | Python/Java |
| **Future Engineers** | **14–19** | **Libre** | **Libre** | ← Coche autónomo |
| RoboDeportes | 12–19 | ≤1.2 kg | Libre | Fútbol/sumo |

**WRO Future Engineers** — la más relevante para este curso:
- Coche autónomo en pista oval con obstáculos de colores
- Sin Lego: hardware libre → ESP32 + cámara + Python
- Detectar y esquivar conos por color (rojo izquierda, verde derecha)
- OpenCV + YOLOv8 en modo ligero para detección en tiempo real

---

#### I — Eurobot y ABU Robocon

**Eurobot (Europa, ≥ 1998):**
- Robots autónomos (+ 1 pequeño RC permitido desde 2021)
- Tarea cambia cada año — 2024: "Plants vs. Robots"
- Robot principal: cubo de 100×100×100 cm máx.
- Interacción con objetos, zonas de puntaje, tiempo de juego: 100 segundos
- Zona LATAM: Eurobot Open incluye equipos latinoamericanos

**ABU Robocon (Asia-Pacífico):**
- Competencia anual entre países de la región ABU (radiodifusión)
- 1 robot manual + 1 robot autónomo por equipo
- Tarea temática cultural cambia cada año
- Altamente espectacular y mediático

---

#### J — Robots Submarinos (ROV / AUV)

**MATE ROV Competition:**

| División | Nivel | Tarea |
|----------|-------|-------|
| Scout | Básico | Tareas simples submarinas |
| Navigator | Intermedio | Control fino + reportes |
| Ranger | Avanzado | Tareas complejas + innovación |
| Explorer | Universitario | Desafíos de ingeniería completos |

- ROV: tethered (cable), control manual desde superficie
- Sistema de control: ESP32 + Python en PC → serial sobre tether
- Actuadores: thruster brushless T100 (BlueRobotics), gripper, cámara

**AUVSI RoboSub (AUV Autónomo):**
- Vehículo completamente autónomo, sin cable
- Tareas: pasar por compuertas, disparar torpedos, recuperar objetos
- Stack: ROS2 + ArduSub + visión + sonar
- Hardware: torpedos neumáticos, gripper, hidrófonos

---

#### K — Robots Humanoides

**ROBO-ONE (Japón):**
- Robot bípedo de combate deportivo (boxeo, sumo, carreras)
- Clases: Standard (≤ 3 kg), Light (≤ 1 kg)
- Balance dinámico: ZMP (Zero Moment Point) en Rust a 1 kHz

**World Humanoid Robot Games (2025, Pekín):**
- 280 equipos, 500+ robots, 26 eventos
- Fútbol, boxeo, limpieza, clasificación de medicamentos, carrera
- Próxima edición: agosto 2026

**RoboCup Humanoid KidSize:**
- Robot bípedo ≤ 80 cm, juega fútbol autónomo
- Visión onboard: detectar pelota, portería, compañeros, rivales
- Walking engine: ZMP + captura de movimiento

---

#### L — Competencias Especiales y Emergentes

| Competencia | Descripción | Tech clave |
|-------------|-------------|-----------|
| **Rubik's Cube** | Resolver cubo en < 1 s | Visión + brazo 6-DOF + algorithms |
| **Robot Dibujante** | Dibujar imágenes con precisión | Path planning + control de fuerza |
| **Agricultural Robot** | Plantar, cosechar, monitorear | YOLOv8 + fuerza adaptativa |
| **Climbing Robot** | Escalar paredes verticales | Ventosas neumáticas + ESP32 |
| **Balancing Robot** | Equilibrio sobre 2 ruedas (Segway) | PID + LQR + IMU |
| **Drone Soccer** | 5v5 drones en esferas de plástico | Autonomous + RC híbrido |
| **Robot Orchestra** | Tocar instrumentos musicales | MIDI + actuadores precisos |
| **Bristlebot** | Robots vibratorios simples | Primeros pasos hardware |
| **Mini Golf Robot** | Hoyo en uno autónomo | Visión + cinemática inversa |
| **Sumo Libre** | Sin reglas de tamaño/diseño | Máxima creatividad |

---

#### M — Mapa de Competencias LATAM 2025–2026

| País | Evento principal | Categorías destacadas |
|------|-----------------|----------------------|
| **Colombia** | Robotic People Fest (30 categorías) | Sumo, batalla, seguidor, FPV |
| **Colombia** | Robot Challenge Colombia | Sumo colegios 1.5 kg |
| **Argentina** | Liga Nacional Robótica (LNR) | Sumo libre, mini sumo, batalla |
| **México** | WRO México 2026 | Future Engineers, RoboMission |
| **Venezuela** | WRO Venezuela | Todas las categorías WRO |
| **Perú** | WRO Perú | Conecta Hub |
| **Brasil** | RoboCup Brasil 2026 | Todas las ligas RoboCup |
| **Chile** | Liga Robótica 2026 Combarbalá | Sumo, seguidor |
| **Regional** | Robomatrix.org | Liga latinoamericana unificada |
| **Regional** | Rharobotics | Torneos universitarios |

---

#### N — Mapa de Módulos del Curso por Competencia

| Competencia | M1 | M2 | M3 | M5 | M6 | M8 | M9 | M10 | M12 |
|-------------|----|----|----|----|----|----|-----|-----|-----|
| Batalla (Beetle) | ✅ | — | ✅ | — | — | ✅ | ✅ | ✅ RC | ✅ PCB |
| Mini Sumo | ✅ | — | ✅ | — | — | ✅ | ✅ | ✅ RC | ✅ |
| Seguidor de línea | ✅ | — | ✅ | — | — | ✅ | ✅ | — | ✅ |
| Micromouse | ✅ | ✅ | ✅ | — | — | ✅ | ✅ | — | ✅ |
| FPV Racing | ✅ | — | ✅ | — | — | ✅ | ✅ | ✅ RC | ✅ FC |
| WRO Future Eng. | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| RoboCup SSL | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Rescate | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | ✅ |
| Eurobot | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| ROV submarino | ✅ | — | ✅ | ✅ | — | ✅ | ✅ | ✅ | ✅ |
| Humanoid | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | ✅ |

---

### 9.2 Drones de Trabajo

#### Arquitectura general de dron profesional

```
Payload (cámara/sensor/lidar)
    ↕ (UART/SPI/Ethernet)
Computadora de misión (Raspberry Pi / Jetson Nano)
    ↕ (MAVLink / UART)
Autopilot (PX4 / ArduPilot sobre ESP32 o Pixhawk)
    ↕ (PWM / DShot)
ESCs → Motores BLDC
```

**Stack de software:**
- **PX4**: autopilot open-source en C++, configurable vía parámetros
- **MAVSDK-Python**: control de alto nivel desde Python (`DroneKit` alternativo)
- **MAVLink**: protocolo de mensajes telemetría/comando entre capas
- **QGroundControl**: GCS (Ground Control Station) open-source

```python
import asyncio
from mavsdk import System

async def fly_mission(waypoints: list[tuple]):
    drone = System()
    await drone.connect()
    await drone.action.arm()
    await drone.action.takeoff()
    for lat, lon, alt in waypoints:
        await drone.offboard.set_position_global(lat, lon, alt)
    await drone.action.return_to_launch()
```

#### Inspección de Infraestructura
- Planificación de vuelo orbital alrededor de torre/puente con Python
- Detección automática de grietas y corrosión: YOLOv8 fine-tuned
- Generación de reporte PDF con coordenadas GPS de anomalías
- Modo seguidor de estructura: mantener distancia constante con ToF

#### Agricultura de Precisión
- **Fotogrametría**: vuelo en grid → imágenes solapadas → reconstrucción 3D con OpenDroneMap
- **NDVI**: cámara multiespectral → índice de vegetación → mapa de salud del cultivo
- **Fumigación de precisión**: activar bomba solo sobre zonas identificadas como afectadas
- Integración con sensores de suelo IoT: correlación vuelo + datos de campo

#### Cartografía y Fotogrametría
- Vuelo en grid automático: `mavsdk` + algoritmo de cobertura
- Procesamiento con **OpenDroneMap** (Python API): punto de nube, mesh, ortofoto
- Generación de DEM (Modelo de Elevación Digital) para planificación de rutas terrestres
- Exportar a GeoTIFF / KMZ para sistemas GIS

#### Entrega de Paquetes (Last-Mile Delivery)
- Navegación punto a punto con geofencing de seguridad
- Detección de zona de aterrizaje libre con visión artificial (no personas, no obstáculos)
- Winch motorizado: descender paquete sin aterrizar
- Sistema de identificación del destinatario: QR code o ArUco en zona de entrega

---

### 9.3 Automatización Logística

#### AGV — Automated Guided Vehicle (Almacén)
Robot de transporte que sigue rutas predefinidas en un almacén.

**Modos de guía:**
| Modo | Tecnología | Flexibilidad |
|------|-----------|-------------|
| Magnético | Cinta magnética en suelo | Baja |
| QR/ArUco | Marcadores en suelo | Media |
| SLAM 2D | LiDAR + mapa dinámico | Alta |
| SLAM 3D | LiDAR 3D + cámara | Muy alta |

**Gestión de flota (Python):**
```python
# Servidor de flota: asigna tareas a robots disponibles
class FleetManager:
    def assign_task(self, task: Task) -> Robot:
        available = [r for r in self.robots if r.idle]
        return min(available, key=lambda r: r.distance_to(task.origin))
```
- Cola de prioridad de misiones
- Resolución de conflictos: prioridad + reserva de celdas del mapa
- API REST (FastAPI) para integración con WMS/ERP
- Dashboard web: posición en tiempo real de cada robot

#### AMR — Autonomous Mobile Robot (Navegación libre)
- SLAM dinámico: remapea en tiempo real si cambia el entorno
- Detección y evitación dinámica de personas (DWA + costmap social)
- Reconocimiento de estaciones de carga: acoplamiento autónomo
- Integración ROS2 Nav2 + micro-ROS en ESP32

#### Sistema de Picking con Visión
- **Bin picking**: objeto desordenado en caja → visión 3D → cinemática inversa → agarre
- Detección de pose 6D: FoundPose / MegaPose para objetos sin textura
- Control de fuerza adaptativo: agarrar sin aplastar objetos frágiles
- Cambio rápido de end-effector: gripper paralelo / ventosa / pinza angular
- Métricas: picks/hora, tasa de error, tiempo de ciclo

#### Clasificación y Sorteo
- Visión artificial: leer QR/código de barras + detectar defectos
- Actuadores de desvío: servo o solenoide en punto de decisión
- Throughput objetivo: diseño inverso desde piezas/hora → velocidad de cinta
- Rechazo inteligente: guardar imagen + metadata de piezas rechazadas para auditoría

---

### 9.4 Automatización Repetitiva (Cobot)

#### Programación por Demostración (Lead-Through)
El operario mueve el brazo a mano; el robot graba y repite la trayectoria.

```python
class LeadThroughRecorder:
    def record(self, duration_s: float) -> Trajectory:
        poses = []
        for _ in range(int(duration_s * self.hz)):
            poses.append(self.robot.get_end_effector_pose())
            time.sleep(1 / self.hz)
        return Trajectory(poses).smooth(window=5)

    def replay(self, traj: Trajectory, speed: float = 1.0):
        for pose in traj.resample(speed):
            self.robot.move_to(pose)
```

#### Tareas de Automatización Repetitiva
| Tarea | Técnica | Sensores |
|-------|---------|---------|
| Pick & place | Cinemática inversa + visión | Cámara 2D / 3D |
| Atornillado | Fuerza constante en eje Z | Celda de carga |
| Soldadura MIG/TIG | Trayectoria CartSpace | Sensor de junta laser |
| Pintura / recubrimiento | Velocidad constante del TCP | Encoders + IMU |
| Control de calidad | Cámara + CNN | Cámara de alta resolución |
| Paletizado | Patrones predefinidos + adaptativo | Sensor de peso |

#### Seguridad Colaborativa (ISO/TS 15066)
- Detección de presencia humana: LiDAR 2D perimetral, cámara con pose estimation
- Zonas dinámicas: velocidad reducida al 25% si hay persona a < 1 m
- Modo stop: parada inmediata si distancia < 30 cm
- Límites de fuerza configurables desde Python en tiempo de ejecución
- Log de todos los eventos de parada de seguridad (trazabilidad)

#### GUI de Programación sin Código
- **Waypoint editor** en Python (PyQt6 / tkinter): drag & drop de puntos
- Visualización 3D del brazo con `robotics-toolbox-python`
- Exportar programa a Rust para ejecución determinista en ESP32
- Simulación previa en Gazebo antes de ejecutar en el robot real

---

### 9.5 Robots Relacionales con LLM

#### Arquitectura de un Robot Conversacional

```
Usuario (voz / texto)
    ↓ Whisper (STT) / texto directo
LLM (Claude / GPT-4o / Llama local)
    ↓ Tool calling → skills del robot
Planificador de tareas (Python)
    ↓ ROS2 / micro-ROS
ESP32 → actuadores
    ↓ estado / feedback
LLM → respuesta verbal (TTS / pantalla OLED)
```

#### Capa de Skills del Robot
```python
# Cada skill es una herramienta que el LLM puede llamar
ROBOT_SKILLS = [
    {
        "name": "move_to_location",
        "description": "Mueve el robot a una ubicación nombrada del mapa",
        "parameters": {"location": {"type": "string", "enum": ["cocina", "sala", "entrada"]}}
    },
    {
        "name": "pick_object",
        "description": "Toma un objeto identificado por nombre o descripción",
        "parameters": {"object_description": {"type": "string"}}
    },
    {
        "name": "say",
        "description": "El robot habla una frase en voz alta",
        "parameters": {"text": {"type": "string"}}
    },
]

# El LLM recibe la instrucción y devuelve qué skill llamar y con qué parámetros
```

#### Comprensión de Lenguaje Natural → Acción
- Instrucción ambigua: "trae eso de allá" → VLM identifica objeto señalado en imagen
- Instrucción compuesta: "ve a la cocina, agarra la botella azul y tráela aquí"
  → LLM descompone en secuencia de skills → ejecuta una a una con confirmación de éxito
- Corrección en tiempo real: "espera", "para", "mejor la verde" → interrupción y replanning
- Negociación: si el robot no puede ejecutar, explica por qué y propone alternativa

#### Memoria Persistente del Robot
- **Memoria episódica**: log de interacciones pasadas → RAG con embeddings locales
- **Mapa semántico**: "donde está la botella" → base de datos de posiciones de objetos
- **Preferencias del usuario**: aprender horarios, rutas favoritas, objetos frecuentes
- Stack: `chromadb` (embeddings locales) + `sentence-transformers` + SQLite

#### Interacción Multimodal
| Canal entrada | Tecnología | Latencia típica |
|--------------|-----------|----------------|
| Voz | Whisper-tiny en local | 300–800 ms |
| Texto | directo al LLM | < 100 ms |
| Gestos | MediaPipe Hands/Pose | 50 ms (en GPU) |
| Mirada | Tobii / cámara + gaze | 100 ms |

| Canal salida | Tecnología |
|-------------|-----------|
| Voz | pyttsx3 / Coqui TTS (local) |
| Pantalla OLED | `luma.oled` desde ESP32 |
| LEDs expresivos | NeoPixel + FSM de emociones |
| Movimiento expresivo | gestos predefinidos según estado emocional |

#### Seguridad y Ética en Robots con LLM
- **Grounding**: el LLM solo puede llamar skills aprobadas — no ejecutar código arbitrario
- **Confirmación humana** para acciones de alto impacto (abrir puertas, manipular objetos críticos)
- **Rate limiting**: máximo N acciones por minuto para evitar comportamiento errático
- **Transparencia**: el robot siempre puede explicar qué está a punto de hacer y por qué
- **Modo privacidad**: desactivar grabación de audio/video bajo demanda

---

### 9.6 Robótica de Servicio

#### Robot de Servicio en Espacios Públicos
- Navegación en entornos dinámicos: personas, sillas, puertas — costmap social
- Detección e identificación de personas: age/gender (opcional), reconocimiento facial (con consentimiento)
- Interacción proactiva: acercarse si detecta persona buscando algo (pose de duda)
- Integración con sistema de pedidos: pantalla táctil → cocina → robot recoge y entrega

#### Robot Asistencial / Compañero
- Monitoreo de signos vitales no invasivos: frecuencia respiratoria por cámara, temperatura IR
- Alerta de caídas: detección por cámara + IMU → notificación a familiar/cuidador
- Recordatorios de medicación con confirmación vocal
- Entretenimiento: juegos adaptativos por voz con LLM personalizado

#### Robot Educativo Programable
- Modo "aprende con el robot": Python básico → el estudiante programa al robot en vivo
- Interfaz Blockly/Scratch → genera código Python → ejecuta en robot real
- Retos progresivos: seguir línea → navegar laberinto → reconocer objetos
- Feedback inmediato: el robot celebra o pide corrección con expresiones LED/voz

---

### 9.7 Automatización de Invernadero y Agricultura
- **Nodo sensor IoT**: ESP32 + sensores (temperatura, humedad, CO2, luz) → MQTT → servidor Python
- **Robot de monitoreo**: navega entre filas, captura imágenes, detecta plagas con YOLOv8
- **Brazo de cosecha**: detección de madurez por color/forma, fuerza controlada para no dañar fruto
- **Riego autónomo**: modelo ML decide cantidad y zona de riego según datos de sensores + historial
- **Drone de mapeo NDVI**: vuelo programado, procesamiento con Python + OpenDroneMap, alertas automáticas

---

### 9.8 Integración de Sistemas Robóticos

#### API REST del Robot (FastAPI)
```python
from fastapi import FastAPI
app = FastAPI()

@app.post("/robot/move")
async def move(x: float, y: float, theta: float):
    await robot.navigate_to(x, y, theta)
    return {"status": "ok", "pose": robot.current_pose}

@app.get("/robot/telemetry")
async def telemetry():
    return robot.get_telemetry_snapshot()
```

#### Stack de Integración Empresarial
| Sistema externo | Integración | Protocolo |
|----------------|-------------|-----------|
| WMS / ERP | API REST bidireccional | HTTP/JSON |
| PLC industrial | OPC-UA con `asyncua` | OPC-UA |
| SCADA | Publicar telemetría | MQTT / Modbus |
| Base de datos | Logs y trazabilidad | PostgreSQL + TimescaleDB |
| Dashboard | Monitoreo en tiempo real | Grafana + MQTT |

#### Multi-Robot Orchestration
- Un servidor Python coordina N robots con colas de prioridad
- Algoritmo de asignación de tareas: Hungarian method para minimizar distancia total
- Resolución de colisiones: reserva de celdas + comunicación entre robots vía MQTT
- Failover: si un robot falla, reasignar su tarea al siguiente disponible

---

## Módulo 12 — Electrónica para Robótica: Componentes, Circuitos y Diseño de PCB
*De la protoboard a la placa propia lista para producción*

> Este módulo cierra la brecha entre "comprar módulos en breakout" y
> "diseñar la electrónica propia del robot": esquemático → layout → fabricación → soldadura → prueba.
> Resultado: PCB custom del robot lista para integrarse en el kit.

---

### 12.1 Fundamentos de Electrónica para Robótica

#### Componentes pasivos
| Componente | Rol en robótica | Regla práctica |
|-----------|----------------|---------------|
| Resistencias | Pull-up/pull-down, limitadores de corriente | E24, 1%, 0603 para SMD |
| Condensadores cerámicos | Desacoplo de alimentación (100nF por IC) | Colocar junto al pin VCC |
| Condensadores electrolíticos | Filtrado de fuente y motores | 470µF–1000µF cerca de drivers |
| Inductores | Filtros DC-DC, EMI | Elegir por corriente de saturación |
| Cristales / osciladores | Reloj preciso (si el MCU lo requiere) | ESP32 tiene oscilador interno |

#### Componentes activos
- **Transistores BJT**: interruptores para cargas pequeñas (LEDs, relés)
- **MOSFETs N/P**: conmutación de motores, LEDs de potencia, carga de batería
- **Optoacopladores**: aislamiento galvánico entre lógica 3.3V y potencia 12V/24V
- **Reguladores LDO**: AMS1117, AP2112 — 3.3V para lógica desde 5V o 12V
- **Convertidores DC-DC switching**: buck (MP2307), boost (MT3608) — eficiencia > 90%

#### Herramientas de medición fundamentales
- **Multímetro**: tensión, corriente (en serie), resistencia, continuidad, diodos
- **Osciloscopio** (2 canales mínimo): señales PWM, I2C, SPI, UART — verificar tiempos
- **Analizador lógico** (Saleae / clone): decodificar I2C, SPI, SBUS, UART en software
- **Fuente de banco regulable**: probar circuitos antes de soldar batería real
- **Soldador de temperatura controlada** (Hakko / TS100): 350°C para SMD, 380°C para through-hole

---

### 12.2 Sistemas de Alimentación del Robot

#### Arquitecturas de alimentación típicas

```
LiPo 2S (7.4V) ─── fusible ─── switch ──┬── Motor driver (7.4V directo)
                                         ├── Buck 5V 3A ──── Raspberry Pi
                                         └── LDO 3.3V ─────── ESP32 + sensores

LiPo 3S (11.1V) ── fusible ─── switch ──┬── ESCs dron (11.1V directo)
                                         └── BEC 5V ───────── FC + receptor RC
```

#### Cálculo de presupuesto de corriente (Power Budget)

```python
# Herramienta Python para calcular presupuesto de potencia del robot
components = {
    "ESP32":              {"v": 3.3, "i_ma": 240},
    "RPi 4 (carga media)":{"v": 5.0, "i_ma": 1200},
    "Motor DC x2":        {"v": 7.4, "i_ma": 800},  # pico: 2000mA
    "LiDAR RPLidar A1":   {"v": 5.0, "i_ma": 500},
    "Cámara OV2640":      {"v": 3.3, "i_ma": 100},
    "Servos x4":          {"v": 5.0, "i_ma": 1600}, # 400mA cada uno bajo carga
    "NeoPixel x16":       {"v": 5.0, "i_ma": 960},  # 60mA por LED
}

for name, c in components.items():
    print(f"{name}: {c['v']*c['i_ma']/1000:.2f} W")
```

#### Protección del circuito
- **Fusible reseteable (PPTC)**: protección de sobreintensidad sin abrir el robot
- **Diodo Schottky inverso**: protección contra inversión de polaridad de batería
- **TVS diode**: protección contra picos de tensión de motores (flyback)
- **Condensadores de bulk** en drivers de motor: absorben picos de corriente

---

### 12.3 Drivers de Motores — Teoría y Selección

#### Motores DC con encoder

| Driver | Corriente cont. | Tensión | Interfaz | Uso |
|--------|----------------|---------|----------|-----|
| DRV8833 | 1.5A | 2.7–10.8V | PWM | Robots pequeños |
| TB6612FNG | 1.2A | 4.5–13.5V | PWM | Zumo, seguidor |
| BTS7960 | 43A | 6–27V | PWM | AGV, robots grandes |
| Cytron MDD10A | 10A | 7–30V | PWM / UART | Ruedas grandes |

#### Motores paso a paso

| Driver | Microstepping | Corriente | Uso |
|--------|--------------|----------|-----|
| A4988 | 1/16 | 2A | Impresoras, brazos ligeros |
| DRV8825 | 1/32 | 2.5A | Mayor precisión |
| TMC2209 | 1/256 | 2A | Silencioso (StealthChop), 3D printing robótica |

#### ESC para BLDC (drones y robots rápidos)
- Protocolos de señal: PWM clásico → DSHOT300/600 (digital, sin calibración)
- BLHeli32: configurable por PC, telemetría de vuelta al FC
- Fórmula de selección: `corriente_ESC > corriente_pico_motor × 1.5`

---

### 12.4 Sensores — Integración y Circuitería

#### Interfaces de comunicación: cuándo usar cada una

| Bus | Velocidad | Pines | Dispositivos | Ideal para |
|-----|----------|-------|-------------|-----------|
| I2C | 400kHz / 1MHz | SDA + SCL | 127 | Sensores lentos (IMU, OLED, ToF) |
| SPI | 10–80 MHz | MOSI+MISO+CLK+CS | ilimitado (1 CS/dev) | LiDAR, ADC rápido, pantallas |
| UART | 115200–3Mbps | TX + RX | 1 a 1 | GPS, LiDAR serial, Bluetooth |
| ADC | — | 1 pin analógico | — | Batería, potenciómetros, FSR |
| CAN bus | 1 Mbps | CANH + CANL | 112 | Actuadores industriales, UAVCAN |

#### Circuitería de acondicionamiento
- **Divisor resistivo** para medir batería LiPo con ADC de 3.3V ESP32
- **Level shifter** bidireccional I2C: conectar módulo 5V a ESP32 3.3V sin quemar
- **Filtro RC** en líneas de encoder: eliminar rebotes eléctricos de alta frecuencia
- **Ferrite bead** en líneas de alimentación de sensores: aislar ruido de motores

---

### 12.5 Diseño de Esquemáticos con KiCad

> **KiCad 8** — herramienta profesional open-source para diseño de PCB.
> Es el estándar de facto en robótica y hardware open-source (Framework, Arduino, etc.)

#### Flujo de trabajo en KiCad

```
1. Esquemático (Eeschema)
   └── Colocar símbolos, conectar nets, asignar valores y footprints

2. Revisión (ERC — Electrical Rules Check)
   └── Detectar pines sin conectar, conflictos de drivers

3. Netlist → PCB Editor (Pcbnew)
   └── Importar netlist, colocar componentes, enrutar pistas

4. Revisión (DRC — Design Rules Check)
   └── Verificar clearances, anchos de pista, cortocircuitos

5. Generación de Gerbers
   └── Enviar a JLCPCB / PCBWay para fabricación
```

#### Buenas prácticas de esquemático
- Una hoja por subsistema: alimentación, MCU, motores, sensores, comunicación
- Net labels en lugar de cables largos: legibilidad
- Bloques de potencia explícitos: mostrar cómo llega VCC y GND a cada sección
- Símbolo de tierra único: GND analógico separado de GND de potencia (unir en un punto)
- DNP (Do Not Place): componentes opcionales marcados claramente

#### Bibliotecas de componentes para robótica
- ESP32, ESP32-S3 — símbolos y footprints oficiales Espressif
- Footprints de conectores: JST-PH, JST-XH, XT30, XT60
- Módulos comunes: MPU-6050, VL53L0X, DRV8833, BTS7960
- Crear símbolo custom en Python con `kicad-skip` (scripting API de KiCad)

---

### 12.6 Layout y Enrutado de PCB

#### Reglas de diseño (DRC) configuradas para fabricación económica (JLCPCB)

| Parámetro | Mínimo estándar | Recomendado |
|-----------|----------------|-------------|
| Ancho de pista señal | 0.127 mm | 0.2 mm |
| Ancho de pista potencia | 0.5 mm | 1.0 mm (calcular por corriente) |
| Clearance pista-pista | 0.127 mm | 0.2 mm |
| Diámetro via | 0.3 mm | 0.4 mm |
| Taladro via | 0.2 mm | 0.3 mm |
| Capas | 2 | 4 para alta densidad |

#### Cálculo de ancho de pista por corriente

```python
# IPC-2221 simplificado: ancho mínimo para temperatura de 10°C sobre ambiente
def trace_width_mm(current_a: float, oz_copper: float = 1.0) -> float:
    # Fórmula para pistas externas, ΔT = 10°C
    area_mil2 = (current_a / 0.048) ** (1 / 0.44)
    area_mm2 = area_mil2 * 0.000645
    thickness_mm = oz_copper * 0.035
    return area_mm2 / thickness_mm

print(f"2A → {trace_width_mm(2):.2f} mm")   # ≈ 0.5 mm
print(f"5A → {trace_width_mm(5):.2f} mm")   # ≈ 1.5 mm
print(f"10A → {trace_width_mm(10):.2f} mm") # ≈ 3.2 mm
```

#### Estrategia de layout por función

- **MCU (ESP32)**: en el centro, todos los periféricos a su alrededor
- **Plano de tierra**: cobre de relleno en capa bottom como retorno de corriente
- **Desacoplo**: condensador 100nF a < 1 mm de cada pin VCC del IC
- **Separación analógico/digital**: sensores IMU alejados de drivers de motor
- **Conectores en el borde**: JST-XH orientados hacia afuera para fácil conexión
- **Fiduciales**: 3 marcas para pick-and-place automático (fabricación en serie)

#### Diseño para fabricación económica en JLCPCB / PCBWay
- Tamaño ≤ 100×100 mm para precio base mínimo
- 2 capas para la mayoría de diseños del curso
- ENIG (gold finish) para conectores y pads de soldadura manual
- Serigrafía: identificar todos los componentes, polaridad de condensadores y LEDs
- Pasta de soldadura (stencil): pedir junto con la PCB para SMD

---

### 12.7 Componentes SMD — Soldadura y Ensamble

#### Tamaños de componentes pasivos SMD

| Tamaño | Dimensiones | Soldabilidad a mano | Uso recomendado |
|--------|------------|---------------------|----------------|
| 0402 | 1.0 × 0.5 mm | Difícil | Producción industrial |
| 0603 | 1.6 × 0.8 mm | Moderada | Proyectos avanzados |
| 0805 | 2.0 × 1.25 mm | Fácil | **Estándar del curso** |
| 1206 | 3.2 × 1.6 mm | Muy fácil | Componentes de alta potencia |

#### Proceso de soldadura SMD a mano (sin horno)
1. Aplicar flux a los pads
2. Estañar uno de los dos pads
3. Posicionar el componente con pinzas, soldar el pad estañado
4. Soldar el segundo pad con poco estaño
5. Limpiar flux residual con IPA 99%
6. Verificar con lupa: sin puentes, sin frías

#### Proceso con pasta y hot plate / heat gun
1. Aplicar pasta de soldadura con stencil metálico
2. Colocar componentes con pinzas o pick-and-place manual
3. Calentar en hot plate 60s a 150°C (precalentamiento) + 30s a 230°C (reflow)
4. Inspección visual + test eléctrico

#### ICs con pad térmico expuesto (ESP32, drivers de motor)
- Soldar pad térmico con vías térmicas al plano de cobre opuesto
- Hot air: boquilla estrecha, 300°C, movimiento circular
- Verificar con multímetro: continuidad de pines críticos

---

### 12.8 PCBs del Curso — Diseño por Proyecto

#### PCB 1 — Shield ESP32 Universal (base de todos los kits)
- Dimensiones: 60 × 50 mm (2 capas, ENIG)
- Incluye: regulador 3.3V, conectores JST-XH para I2C×2 / SPI×1 / UART×2 / GPIO×8
- Conectores motor: XT30 para batería, 2× JST-XH 2P para motores
- LED de estado + botón de boot + botón de reset accesibles
- Header 2.54mm compatible con ESP32 DevKit V1

#### PCB 2 — Driver de Motores Doble (Kit 1, 2 y 6)
- DRV8833 × 2 (hasta 1.5A por canal)
- Condensadores de bulk 1000µF + diodos TVS
- Ventilación térmica: pad térmico + vías al plano
- Conector entrada: XT30 batería, salida: JST-XH 2P × 4 motores

#### PCB 3 — Placa de Control Brazo 4-DOF (Kit 3)
- ESP32 + PCA9685 (16 canales servo PWM)
- Separación de alimentación: 3.3V lógica / 5V servos (cada una con capacidad 3A)
- LED por canal servo para diagnóstico visual
- Conector I2C para IMU, OLED, ToF

#### PCB 4 — Flight Controller Custom (Kit 4 — Dron)
- ESP32-S3 + MPU-6000 (SPI)
- 4× conectores DShot para ESC
- Receptor RC: UART con inversión de nivel para SBUS
- Buzzer, LED arcoíris, conector JTAG para debug
- Dimensiones: 36 × 36 mm (standard FC mounting M3)

#### PCB 5 — Placa Base AGV (Kit 6)
- ESP32 + BTS7960 × 2 (hasta 43A por canal)
- INA226 (medición de corriente y potencia en tiempo real)
- Conector para RPi via UART/I2C
- Fusibles reseteables PPTC 5A en cada canal de motor

---

### 12.9 Pedido y Fabricación

#### JLCPCB — flujo estándar del curso

```
KiCad → Archivo → Plot → Gerbers (zip)
KiCad → Archivo → Fabrication Outputs → BOM + Pick and Place

Subir a jlcpcb.com:
  - Gerbers: capas Cu, Mask, Silk, Edge.Cuts
  - Cantidad: 5 piezas (mínimo, precio base ~$2 USD)
  - Tiempo: 24h fabricación + 5–7 días envío
  - PCBA (ensamble): subir BOM + CPL → JLCPCB solda los SMD
```

#### Cuándo usar PCBA (ensamble en fábrica) vs. soldadura manual
- **Soldadura manual** (curso): aprender el proceso, cantidades < 20 unidades
- **PCBA JLCPCB**: kits en producción > 50 unidades, componentes 0402, ICs de paso fino

#### Control de calidad post-fabricación
- **Visual**: lupa × 10, verificar todos los pads soldados
- **Eléctrico básico**: multímetro — continuidad VCC/GND, sin cortocircuitos
- **Funcional**: cargar firmware mínimo → LED parpadea → sensores responden
- **Test jig**: pogo pins contra la PCB para producción en serie (script Python verifica automáticamente)

---

### 12.10 Fabricación de PCB — Manual · Láser · Diseño de Rutas · JLCPCB

> Tres métodos para pasar de un diseño KiCad a una PCB física:
> manual con química, con nuestra cortadora láser 400×400 mm,
> y el flujo profesional para enviar a fabricar a JLCPCB.

---

#### 12.10.1 Método Manual — Transferencia de Tóner + Grabado Químico

El método más accesible para probar un diseño en horas, sin esperar envíos.
Solo una capa, resolución limitada a ~0.5 mm — ideal como primer prototipo.

**Materiales necesarios:**

| Material | Dónde conseguir | Precio aprox. |
|----------|----------------|--------------|
| Placa de cobre FR1 (papel fenólico) o FR4 | Tienda electrónica | $1–3 USD / pieza |
| Cloruro férrico FeCl₃ (solución 40%) | Tienda química / electrónica | $3–5 USD |
| Papel fotográfico brillante | Papelería | $0.10 / hoja |
| Plancha de ropa (150°C) | Doméstica | — |
| Acetona | Ferretería | $2 USD |
| Taladro con brocas PCB 0.6–1.0 mm | Tienda hobby / electrónica | $5–15 USD |
| Guantes de nitrilo + bandeja plástica | — | $2 USD |

**Proceso paso a paso:**

```
1. PREPARAR EL DISEÑO EN KICAD
   Plot → SVG de F.Cu → Espejar horizontalmente (la imagen va invertida)
   Imprimir en papel fotográfico brillante, tóner lleno (no economía)

2. LIMPIAR LA PLACA
   Lijar suavemente con lija 400 → limpiar con acetona
   No tocar el cobre con los dedos después de limpiar

3. TRANSFERIR EL DISEÑO
   Colocar papel impreso boca abajo sobre el cobre
   Planchar a 150–160°C con presión firme, movimiento circular, 4–6 minutos
   Enfriar 2 minutos → remojar en agua tibia 5 minutos
   Retirar papel con cuidado — el tóner queda sobre el cobre

4. CORREGIR DEFECTOS
   Revisar con lupa — pistas cortadas o unidas
   Corregir con marcador permanente impermeable (Sharpie negro)

5. GRABAR (ETCHING)
   Sumergir en FeCl₃ a temperatura ambiente o tibio (acelera el proceso)
   Agitar suavemente, 10–20 minutos según concentración
   El cobre sin proteger desaparece — solo quedan las pistas cubiertas por tóner
   ⚠️ FeCl₃ mancha permanentemente — usar guantes y bandeja plástica

   Alternativa más casera (HCl + H₂O₂):
   Mezcla 1 parte HCl (ácido muriático) + 2 partes H₂O₂ 3%
   Más rápido (2–5 min), pero más agresivo — ventilación obligatoria

6. LIMPIAR Y FINALIZAR
   Enjuagar con agua abundante
   Limpiar tóner con acetona → quedan las pistas de cobre brillantes
   Opcional: estañar pistas con flux + soldadura para proteger de oxidación

7. TALADRAR
   Brocas PCB 0.6mm (SMD vías) y 0.8–1.0mm (through-hole)
   Mini taladro a baja velocidad para no romper las brocas finas
```

**Limitaciones del método manual:**
- Solo 1 capa (single-sided)
- Resolución mínima: ~0.5 mm de pista y separación
- Sin máscara de soldadura ni serigrafía
- Tiempo total: 1.5–2.5 horas por placa
- Uso recomendado: verificar diseño antes de ordenar en JLCPCB

---

#### 12.10.2 Método Láser — Ablación de Cobre con nuestra Cortadora (400×400 mm)

La cortadora láser del curso elimina directamente la capa de cobre
donde no debe existir pista, dejando el diseño grabado con precisión.

**¿Por qué láser sobre manual?**
- Reproducible: la segunda placa es idéntica a la primera
- Sin químicos de grabado (puede omitirse o reducirse)
- Nuestra máquina CO₂ 40W cabe en el área de trabajo 400×400 mm ✅
- Resolución: ~0.3 mm con buen enfoque (mejor que tóner)

**Dos técnicas con láser:**

**Técnica A — Ablación directa de cobre (más rápida):**
```
El láser quema y evapora el cobre (Cu) donde NO debe haber pista.
Las pistas se definen como zonas NO grabadas.
```

```
PARÁMETROS TÍPICOS para láser CO₂ 40W sobre placa FR1:
  Potencia:   65–75%
  Velocidad:  80–120 mm/s
  Pasadas:    2–3 (para eliminar el cobre completamente)
  Enfoque:    exacto sobre la superficie de cobre
  Nota: FR4 (fibra de vidrio) emite gases tóxicos — usar FR1 (papel fenólico)
        o extraer vapores con ventilación forzada
```

**Técnica B — Corte de máscara + grabado químico breve:**
```
1. Cortar vinyl autoadhesivo (0.08 mm grosor) con el láser
   siguiendo las pistas del diseño
2. Aplicar vinyl como máscara sobre la placa de cobre
3. Grabar químico breve (30–60 seg con HCl+H₂O₂ tibio)
4. Retirar máscara → pistas perfectas

Ventaja: grabado más limpio en pistas muy finas (< 0.3 mm)
```

**Flujo KiCad → LightBurn → Láser:**

```python
# En KiCad: File → Plot → SVG
# Seleccionar: F.Cu únicamente
# ✅ Espejar (Mirror): NO (el láser no necesita imagen espejada)
# ✅ Escala 1:1 exacta
# ✅ Color: negro sobre blanco (negro = zona a grabar)

# En LightBurn:
# 1. Importar SVG
# 2. Invertir si usas ablación (grabar TODO excepto las pistas)
# 3. Asignar capa de corte con los parámetros de tu máquina
# 4. Frame → verificar que la placa está bien posicionada
# 5. Start
```

**Perfiles de LightBurn recomendados (guardar como preset):**

| Preset | Potencia | Velocidad | Pasadas | Material |
|--------|---------|-----------|---------|---------|
| `pcb_fr1_ablation` | 70% | 100 mm/s | 3 | FR1 cobre 35µm |
| `pcb_vinyl_mask` | 15% | 300 mm/s | 1 | Vinyl autoadhesivo |
| `pcb_fr1_edge` | 85% | 50 mm/s | 5 | Corte del contorno |

**Post-proceso:**
- Limpiar residuos con cepillo + IPA 99%
- Estañar pistas para proteger de oxidación
- Taladrar through-holes con mini taladro

---

#### 12.10.3 Diseño de Rutas para PCB — Filosofía y Técnica

El enrutado es el arte de conectar los pads del esquemático
respetando física, señal e interferencia electromagnética.
Un mal enrutado puede hacer que un circuito perfecto en papel falle en hardware.

**Enrutado manual vs. auto-router:**

| Método | Calidad | Velocidad | Recomendación |
|--------|---------|----------|--------------|
| Manual total | ✅✅✅ | Lento | Siempre para señales críticas |
| Freerouting (plugin KiCad) | ✅✅ | Rápido | Punto de partida para señales simples |
| Auto-router KiCad nativo | ✅ | Muy rápido | Solo para referencias, revisar siempre |

**Orden de enrutado — siempre este orden:**

```
1. PLANO DE TIERRA (copper fill en B.Cu o capa dedicada)
   → Primero lo más importante: retorno de corriente limpio

2. PISTAS DE POTENCIA (VCC, VBAT, 5V, 3.3V)
   → Anchas y directas, calculadas con la fórmula de IPC-2221

3. SEÑALES DE ALTA FRECUENCIA (SPI CLK, UART, I2S, RF)
   → Cortas, sin vías si es posible, alejadas de pistas de motor

4. SEÑALES DIFERENCIALES (USB D+/D-, Ethernet TX/RX)
   → Length matching: misma longitud exacta los dos pares
   → Separación constante entre ellas (impedancia diferencial 90Ω)

5. SEÑALES DE CONTROL (GPIO, PWM, I2C)
   → Las más flexibles, pueden rodear obstáculos

6. RELLENO DE COBRE (copper pour GND en F.Cu)
   → Al final, rellenar zonas sin pista con tierra
```

**Reglas de enrutado profesional:**

```
ANCHO DE PISTA:
  Señal 50mA  → 0.2 mm mínimo (recomendado 0.3 mm)
  Señal 200mA → 0.5 mm
  Potencia 1A → 1.0 mm
  Potencia 3A → 2.0 mm
  Motor 5A    → 3.0–4.0 mm (calcular con IPC-2221)

CLEARANCE (separación):
  Señal-señal:    0.2 mm (mínimo JLCPCB)
  Potencia-señal: 0.5 mm
  HV-señal:       1.0 mm+ (> 48V)

VÍAS:
  Diámetro: 0.4 mm exterior, 0.2 mm taladro (mínimo JLCPCB)
  Via stitching GND: cada 5–8 mm para reducir inductancia de retorno
  Teardrops: habilitar en KiCad para refuerzo mecánico pad-pista

REGLAS EMC:
  No cruzar pistas de señal bajo pistas de motor/potencia
  Loop de retorno: cada pista de señal debe tener retorno GND próximo
  Condensadores de desacoplo: a < 1mm del pin VCC del IC (no > 2mm)
  Punto estrella de tierra: unir GND analógico y GND digital en un solo punto
```

**Script Python para verificar reglas con la API de KiCad:**

```python
import pcbnew

def check_custom_rules(kicad_pcb_path: str):
    board = pcbnew.LoadBoard(kicad_pcb_path)
    issues = []

    for track in board.GetTracks():
        width_mm = pcbnew.ToMM(track.GetWidth())

        # Verificar pistas de señal demasiado delgadas
        if width_mm < 0.15:
            pos = track.GetStart()
            issues.append(
                f"Pista muy delgada {width_mm:.2f}mm en "
                f"({pcbnew.ToMM(pos.x):.1f}, {pcbnew.ToMM(pos.y):.1f})"
            )

    # Verificar condensadores de desacoplo cerca de ICs
    for fp in board.GetFootprints():
        if fp.GetValue().startswith("C") and "100n" in fp.GetValue():
            # Verificar que hay un IC en radio < 2mm
            nearby_ics = [
                f for f in board.GetFootprints()
                if f != fp and
                pcbnew.ToMM(
                    (fp.GetPosition() - f.GetPosition()).EuclideanNorm()
                ) < 2.0
                and f.GetValue().startswith("U")
            ]
            if not nearby_ics:
                issues.append(
                    f"Condensador {fp.GetReference()} "
                    f"posiblemente lejos de su IC"
                )

    for issue in issues:
        print(f"⚠️  {issue}")
    print(f"\n{'✅ Sin problemas' if not issues else f'{len(issues)} problema(s) encontrado(s)'}")

check_custom_rules("firmware/hardware/robot_shield.kicad_pcb")
```

**Diseño de vías de impedancia controlada (para RF y USB):**
```
Línea de transmisión 50Ω en FR4 (εr = 4.4):
  Capa externa (microstrip):
    w = ancho de pista, h = grosor del dieléctrico
    Fórmula simplificada: w ≈ 1.9 × h para 50Ω en 1.6mm FR4
    → Con h = 1.6mm: w ≈ 3 mm (demasiado ancha para señales rápidas)
    → Con h = 0.2mm (4 capas): w ≈ 0.38 mm ← manejable

  Calculadora integrada en KiCad: Board Setup → Impedance Control
```

---

#### 12.10.4 Flujo Completo para JLCPCB — De KiCad al Pedido

JLCPCB es el fabricante que usamos para todos los kits del curso.
Precio base: 5 piezas de 100×100mm por ~$2 USD + envío.

**Paso 1 — Verificación final en KiCad (DRC completo)**

```
KiCad → Inspect → Design Rules Checker → Run DRC

Configuración mínima antes del DRC:
  Board Setup → Constraints:
    Minimum track width:    0.127 mm
    Minimum clearance:      0.127 mm
    Minimum via diameter:   0.4 mm
    Minimum via drill:      0.2 mm
    Minimum hole to hole:   0.5 mm
    Minimum courtyard gap:  0.1 mm

Resultado esperado: 0 errores, 0 warnings
```

**Paso 2 — Generación de Gerbers**

```
KiCad → File → Fabrication Outputs → Gerbers

Capas a exportar (todas necesarias para JLCPCB):
  ✅ F.Cu          (capa superior de cobre)
  ✅ B.Cu          (capa inferior de cobre)
  ✅ F.Mask        (máscara de soldadura superior)
  ✅ B.Mask        (máscara de soldadura inferior)
  ✅ F.Silkscreen  (serigrafía superior)
  ✅ B.Silkscreen  (serigrafía inferior — opcional)
  ✅ Edge.Cuts     (contorno de la placa — OBLIGATORIO)
  ✅ F.Courtyard   (solo para verificación, no obligatorio)

Configuración de plot:
  ✅ Use Protel filename extensions
  ✅ Generate Gerber job file
  Coordinate format: 4.6 (recomendado por JLCPCB)

Archivo de taladros (Drill File):
  Format: Excellon
  ✅ PTH y NPTH en archivos separados
  ✅ Metric
  Map file format: Gerber (para visualización)
```

**Paso 3 — Verificar Gerbers antes de ordenar**

```python
# Verificar Gerbers con el visor online antes de pedir
# URL: https://gerber-viewer.jlcpcb.com  (no compartir datos sensibles)

# También verificar localmente con gerbv o el visor de KiCad:
# KiCad → File → Open Gerber Viewer → cargar los archivos generados

# Script para verificar que existen todos los archivos necesarios:
from pathlib import Path

REQUIRED_GERBERS = [
    "*.GTL",   # Top copper
    "*.GBL",   # Bottom copper
    "*.GTS",   # Top soldermask
    "*.GBS",   # Bottom soldermask
    "*.GTO",   # Top silkscreen
    "*.GKO",   # Board outline (Edge.Cuts)
    "*.DRL",   # Drill file
]

gerber_dir = Path("hardware/gerbers/")
for pattern in REQUIRED_GERBERS:
    files = list(gerber_dir.glob(pattern))
    status = "✅" if files else "❌ FALTA"
    print(f"{status}  {pattern}: {files[0].name if files else '---'}")
```

**Paso 4 — Formulario de pedido en JLCPCB**

```
jlcpcb.com → Order → Upload Gerber ZIP

OPCIONES ESTÁNDAR DEL CURSO:
  Base Material:     FR-4
  Layers:            2
  Dimensions:        (automático desde Edge.Cuts)
  PCB Qty:           5  (mínimo, precio base)
  PCB Thickness:     1.6 mm
  PCB Color:         Verde (más económico) / Negro (mejor para kits)
  Silkscreen:        White
  Surface Finish:    HASL(with lead) — económico para prototipo
                     ENIG — para conectores, pads de contacto, BGA
  Copper Weight:     1 oz (35µm) estándar
                     2 oz si hay pistas de alta corriente (> 3A)
  Via Covering:      Tented (vías cubiertas — recomendado)
  Min Hole Size:     0.3 mm
  Board Outline:     ± 0.2 mm tolerancia

OPCIONES AVANZADAS (cuando aplica):
  Impedance Control: Sí → especificar capas y valor (50Ω, 90Ω diferencial)
  Gold Fingers:      Para conectores de borde (PCIe-style)
  Castellated Holes: Para módulos que se sueldan sobre otra PCB
```

**Paso 5 — PCBA (Ensamble por JLCPCB)**

Para los kits del curso en producción (> 20 unidades), JLCPCB solda los SMD:

```
Archivos adicionales necesarios para PCBA:
  1. BOM (Bill of Materials): Excel/CSV
     Columnas: Comment, Designator, Footprint, LCSC Part#

  2. CPL (Component Placement List): CSV
     Columnas: Designator, Val, Package, Mid X, Mid Y, Rotation, Layer
     ⚠️ Rotar ICs: KiCad vs. JLCPCB tienen convenciones distintas
                   Verificar cada IC manualmente en el preview

Exportar desde KiCad:
  File → Fabrication Outputs → Component Placement (CPL)
  File → Fabrication Outputs → BOM
```

```python
# Script para generar BOM con LCSC part numbers desde campos del esquemático
import csv
from pathlib import Path
import xml.etree.ElementTree as ET

def kicad_bom_to_jlcpcb(kicad_bom_xml: Path, output_csv: Path):
    tree = ET.parse(kicad_bom_xml)
    root = tree.getroot()

    components = []
    for comp in root.iter("comp"):
        value     = comp.find("value").text or ""
        footprint = comp.find("footprint").text or ""
        ref       = comp.get("ref", "")
        lcsc      = ""

        for field in comp.iter("field"):
            if field.get("name") == "LCSC":
                lcsc = field.text or ""

        if "DNP" in value.upper():   # omitir Do Not Place
            continue

        components.append({
            "Comment":    value,
            "Designator": ref,
            "Footprint":  footprint.split(":")[-1],  # solo nombre, sin librería
            "LCSC Part#": lcsc,
        })

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["Comment","Designator","Footprint","LCSC Part#"])
        w.writeheader()
        w.writerows(components)

    print(f"✅ BOM exportado: {len(components)} componentes → {output_csv}")
    sin_lcsc = [c for c in components if not c["LCSC Part#"]]
    if sin_lcsc:
        print(f"⚠️  {len(sin_lcsc)} componentes sin LCSC Part#:")
        for c in sin_lcsc:
            print(f"    {c['Designator']} ({c['Comment']})")

kicad_bom_to_jlcpcb(
    Path("hardware/robot_shield.xml"),
    Path("hardware/gerbers/bom_jlcpcb.csv")
)
```

**Paso 6 — Checklist final antes de confirmar el pedido**

```
DISEÑO
  ✅ DRC sin errores en KiCad
  ✅ Gerbers revisados en visor (gerber-viewer.jlcpcb.com)
  ✅ Dimensiones correctas (medir Edge.Cuts en el visor)
  ✅ Orientación del texto de serigrafía legible
  ✅ Polaridad de condensadores electrolíticos marcada en silk

FABRICACIÓN
  ✅ Surface finish elegida según uso (HASL vs ENIG)
  ✅ Copper weight correcto (1oz vs 2oz)
  ✅ Número de piezas confirmado

PCBA (si aplica)
  ✅ LCSC Part# para todos los componentes a ensamblar
  ✅ Stock suficiente en LCSC (verificar en la web)
  ✅ Rotación de ICs verificada en preview de JLCPCB
  ✅ Componentes DNP marcados correctamente
  ✅ Fiduciales en el layout (mínimo 3, en esquinas)

ENTREGA
  ✅ Dirección de envío correcta
  ✅ Método de envío seleccionado (DHL/FedEx para urgente, EMS económico)
  ✅ Revisado precio final antes de pagar
```

---

#### 12.10.3b Método CNC — Fresado Mecánico de PCB

La CNC mecánica usa una fresa de punta V-bit o end-mill para cortar
físicamente el cobre entre las pistas, sin químicos ni láser.
Es el método con mejor resolución sin depender de fábrica.

**¿Cuándo elegir CNC sobre láser o manual?**
- Resolución de 0.1–0.2 mm (mejor que láser en pistas finas)
- Sin vapores tóxicos (vs. láser sobre FR4 con fibra de vidrio)
- Resultado profesional: pistas limpias, bordes definidos
- Repetible y automático una vez configurado

**Herramientas necesarias:**

| Herramienta | Especificación | Notas |
|------------|---------------|-------|
| CNC de escritorio | 3 ejes, área mínima 200×200 mm | Estilo 3018, Sainsmart, Carbide3D |
| Fresa V-bit | 10–30° punta, Ø3.175 mm (1/8") | Para trazar pistas aisladas |
| End-mill recto | Ø0.8–1.0 mm | Para cortar el contorno (Edge.Cuts) |
| Placa FR4 / FR1 | Cobre 35µm (1oz) | Sujetar firmemente con cinta de doble cara |
| Software CAM | FlatCAM (gratis) o KiCad-to-GCode | Genera G-code desde Gerbers |

**Flujo KiCad → FlatCAM → CNC:**

```
1. EXPORTAR GERBERS desde KiCad
   (igual que para JLCPCB: F.Cu, Edge.Cuts, Drill)

2. IMPORTAR EN FLATCAM
   File → Open Gerber → seleccionar F.Cu.gbr
   File → Open Excellon → seleccionar archivo de taladros

3. CONFIGURAR OPERACIÓN DE AISLAMIENTO (pistas)
   Seleccionar F.Cu → Selected → Isolation Routing
     Tool dia:    0.1–0.2 mm (ancho de corte)
     Pass overlap: 0.15  (solapamiento entre pasadas)
     Passes:       2     (una pasada por cada lado de la pista)
   → Genera CNCJob con las rutas de fresado

4. CONFIGURAR CORTE DE CONTORNO (Edge.Cuts)
   Seleccionar Edge.Cuts → Selected → Cutout Tool
     Tool dia:  0.8–1.0 mm end-mill
     Multi-depth: ✅  (profundidad en varias pasadas para no romper fresa)
     Depth/pass:  0.3 mm

5. CONFIGURAR TALADRADO
   Seleccionar Excellon → Selected → Drilling
     Fresa: Ø0.8 mm (vías), Ø1.0 mm (through-hole)

6. EXPORTAR G-CODE
   CNCJob → Export G-code → robot_shield_traces.gcode
   CNCJob → Export G-code → robot_shield_cutout.gcode
   Drilling → Export G-code → robot_shield_drill.gcode

7. ENVIAR A LA CNC
   Software: Candle, bCNC, UGS (Universal Gcode Sender)
   Procedimiento:
     a. Fijar placa con cinta de doble cara sobre sacrificial
     b. Home y zero en la esquina de referencia
     c. Probing de superficie (Z auto-level) si la placa no es perfectamente plana
     d. Ejecutar: primero trazado → luego taladros → luego contorno
```

**Z auto-leveling — clave para resultados precisos:**

La placa de cobre nunca es perfectamente plana sobre la mesa CNC.
El auto-leveling mide la altura en una grilla de puntos y ajusta el Z
en tiempo real durante el fresado → profundidad de corte constante.

```python
# Script Python para generar G-code con Z compensation (bCNC / Candle)
# Basado en un mapa de altura medido en grilla 5×5

import numpy as np
from scipy.interpolate import RectBivariateSpline

def apply_z_compensation(gcode_path: str, height_map: np.ndarray,
                          x_range: tuple, y_range: tuple) -> str:
    xs = np.linspace(x_range[0], x_range[1], height_map.shape[1])
    ys = np.linspace(y_range[0], y_range[1], height_map.shape[0])
    interp = RectBivariateSpline(ys, xs, height_map)

    compensated = []
    with open(gcode_path) as f:
        for line in f:
            if line.startswith("G1") and "Z" in line:
                # Parsear X, Y, Z del comando
                parts = {p[0]: float(p[1:]) for p in line.split()
                         if p[0] in "XYZ"}
                x, y = parts.get("X", 0), parts.get("Y", 0)
                z_original = parts.get("Z", 0)
                z_offset = float(interp(y, x))
                z_new = z_original + z_offset
                line = f"G1 X{x:.3f} Y{y:.3f} Z{z_new:.3f}\n"
            compensated.append(line)
    return "".join(compensated)
```

**Parámetros de corte recomendados (CNC 3018, fresa V-bit 20°):**

| Parámetro | Valor | Notas |
|-----------|-------|-------|
| Velocidad de husillo | 8000–12000 RPM | Mínimo para cortes limpios |
| Feed rate (XY) | 100–150 mm/min | Lento = mejor acabado |
| Plunge rate (Z) | 50 mm/min | Bajar con cuidado |
| Profundidad de corte | 0.05–0.1 mm | Solo quitar el cobre (35µm) |
| Passes de aislamiento | 2 | Una a cada lado de la pista |
| Overlap | 40% del diámetro de fresa | Para no dejar cobre entre pasadas |

**Post-proceso:**
- Limpiar viruta con cepillo suave + aspiradora
- Revisar continuidad de pistas con multímetro
- Lijar suavemente con lija 600 para eliminar rebabas
- Estañar pistas o aplicar barniz protector (conformal coating)

---

#### Comparativa de los cuatro métodos

| Criterio | Manual (tóner) | Láser (ablación) | CNC (fresado) | JLCPCB |
|----------|---------------|-----------------|--------------|--------|
| **Costo por placa** | ~$1–2 | ~$0.5–1 | ~$0.5–2 | ~$0.40 (lote 5) |
| **Tiempo hasta tener la placa** | 2 horas | 1 hora | 1–2 horas | 5–10 días |
| **Resolución mínima** | 0.5 mm | 0.3 mm | **0.1–0.2 mm** | **0.127 mm** |
| **Capas** | 1 | 1 | 1 | 2–32 |
| **Máscara de soldadura** | No | No | No | ✅ Sí |
| **Serigrafía** | No | Parcial | No | ✅ Sí |
| **SMD fine-pitch** | No | No | Parcial | ✅ Sí |
| **Sin químicos** | No (FeCl₃) | ✅ Sí | ✅ Sí | N/A |
| **Sin vapores tóxicos** | Parcial | ⚠️ FR4 tóxico | ✅ Sí | N/A |
| **Reproducibilidad** | Baja | Alta | Alta | ✅ Perfecta |
| **Inversión inicial** | Baja | Máquina láser | CNC de escritorio | Ninguna |
| **Cuándo usarlo** | Urgente / sin máquinas | Prototipos repetibles | Fine-pitch sin químicos | Diseño final / kits |

---

**Flujo recomendado del curso:**
```
Diseño en KiCad
    │
    ├── ¿Necesitas validar el circuito HOY sin máquinas?
    │       └── Manual (tóner + FeCl₃) → 2 horas
    │
    ├── ¿Prototipo repetible con nuestra cortadora láser?
    │       └── Láser → LightBurn → 1 hora por placa (FR1 únicamente)
    │
    ├── ¿Pistas finas (< 0.3 mm) sin químicos?
    │       └── CNC → FlatCAM → G-code → 1–2 horas por placa
    │
    └── ¿Diseño finalizado para el kit o producción?
            └── JLCPCB → Gerbers + BOM + CPL → 5–10 días → profesional
```

---

### 12.11 Electrónica Miniaturizada para Robótica — Componentes SMD: Catálogo Completo

> Los robots modernos se construyen con componentes de superficie (SMD/SMT). Esta sección es la
> enciclopedia de referencia del curso: cada tipo de componente, su familia de encapsulados,
> valores típicos para robótica y cómo trabajarlo con nuestras herramientas.

---

#### 12.11.1 Pasivos SMD — El 80 % del área de la PCB

##### Resistencias

| Encapsulado | Tamaño (mm) | Potencia típica | Uso en robótica |
|-------------|------------|-----------------|-----------------|
| **0201** | 0.6 × 0.3 | 50 mW | RF, wearables (rework muy difícil) |
| **0402** | 1.0 × 0.5 | 63 mW | Señal: pull-up/down, divisores, filtros |
| **0603** | 1.6 × 0.8 | 100 mW | Uso general — mínimo recomendado para mano |
| **0805** | 2.0 × 1.2 | 125 mW | Potencia media, LED series |
| **1206** | 3.2 × 1.6 | 250 mW | Alta potencia, shunts de medición |
| **2512** | 6.3 × 3.2 | 1 W | Shunts de corriente de precisión |

Valores críticos para robótica:
- **Pull-up I²C:** 2.2 kΩ / 4.7 kΩ (0402/0603, depende de velocidad del bus)
- **Pull-down MOSFET gate:** 10 kΩ (seguridad en cold boot)
- **Shunt de corriente:** 10 mΩ / 100 mΩ 2512 (para INA226)
- **Terminación CAN:** 120 Ω 0805 (en cada extremo del bus)

Resistencias especiales:
- **Thermistor NTC** (10 kΩ @ 25°C, β = 3950) — temperatura de motor/batería
- **Varistor (MOV)** — supresión de ESD en entradas de usuario
- **Resistencia de fusión (fusible resettable)** — protección de cortocircuito soft

---

##### Condensadores

| Tipo | Serie | Rango de valor | Ventaja | Limitación |
|------|-------|---------------|---------|------------|
| **MLCC C0G/NP0** | Murata GRM | 1 pF – 100 nF | Temperatura estable, RF | Valor bajo |
| **MLCC X5R** | Samsung CL | 100 nF – 10 µF | Compacto, decoupling | Coef. tensión alto |
| **MLCC X7R** | Kemet | 1 nF – 10 µF | Balance temperatura/valor | Microfónico |
| **Tantalio (Tant)** | AVX TAJB | 1 µF – 470 µF | Alta capacidad, bajo ESR | Polarizado, caro |
| **Electrolítico SMD** | Panasonic FC | 10 µF – 1000 µF | Bulk supply, económico | ESR alto, tamaño |
| **Polímero SMD** | Panasonic SP | 10 µF – 560 µF | ESR muy bajo | Precio |

Distribución típica en placa de robot:

```
Por cada regulador de potencia:
  ├── 100 nF X7R 0402   (decoupling HF — junto al pin VCC del IC)
  ├── 10 µF X5R 0805    (decoupling MF — ≤ 5 mm del IC)
  └── 100 µF tantalo/electrolítico (bulk — cerca del conector de alimentación)

Entrada ADC:
  └── 100 nF C0G 0402 + 1 µF X5R 0402 (filtro RC, sin microfónico)

Bus I²C / SPI (pull-up):
  └── 100 pF C0G 0402 en cada línea (absorbe glitches de bus largo)
```

---

##### Inductores y Ferrites

| Componente | Encapsulado | Corriente nominal | Uso |
|------------|------------|-------------------|-----|
| **Inductor de potencia** | 4×4 mm, 6×6 mm | 1 A – 10 A | Buck/boost, filtros de salida |
| **Inductor de señal** | 0402, 0603 | 100 mA | Filtros RF, EMI supresión |
| **Ferrite bead** | 0402, 0603, 0805 | 500 mA – 3 A | Separación de dominio analógico/digital |
| **Common mode choke** | 2012, 3216 | 100 mA | USB, CAN diferencial |

Regla para ferrites: elige impedancia ≥ 100 Ω @ 100 MHz; corriente ≥ 2× corriente de carga.

---

##### Cristales y Resonadores

| Componente | Encapsulado | Exactitud | Uso en robótica |
|------------|------------|-----------|-----------------|
| **Cristal 32.768 kHz** | 3215, 1610 | ±20 ppm | RTC (hora en SLAM, logs) |
| **Cristal 8–25 MHz** | HC-49/SMD, 5032 | ±30 ppm | Clock principal MCU |
| **Resonador cerámico** | 3225 | ±0.5 % | Bajo costo, menos exacto |
| **TCXO 10 MHz** | 5032 | ±0.5 ppm | GPS, LoRa timing crítico |

---

#### 12.11.2 Discretos Activos SMD

##### Diodos

| Tipo | Ejemplo | Encapsulado | Parámetro clave | Uso robótico |
|------|---------|-------------|-----------------|--------------|
| **Schottky** | BAT54S | SOT-23 | Vf = 0.2–0.4 V | Protección polaridad inversa, OR-ing de alimentación |
| **Schottky potencia** | SS54 / SK54 | SMA/SMB | 5 A, 40 V | Diodo catch en drivers de motor |
| **Zener** | BZX84 | SOT-23 | 2.4 V – 39 V | Regulación de referencia, clamp de gate |
| **TVS unidireccional** | SMBJ5.0A | SMB | Ipp = 26 A | ESD en señales externas, pines RF |
| **TVS bidireccional** | SMBJ12CA | SMB | ±clamp | Señales AC, bus diferencial |
| **Diodo rápido** | 1N4148W | SOD-123 | trr = 4 ns | Señales rápidas, circuitos de control |
| **LED** | VLMK33 | 0603 | 20 mA | Indicadores de estado del robot |

---

##### Transistores BJT

| Tipo | Ejemplo | Encapsulado | Ic max | hFE | Uso |
|------|---------|-------------|--------|-----|-----|
| **NPN general** | BC817 | SOT-23 | 500 mA | 100–600 | Switch de LED, relay, buzzer |
| **PNP general** | BC807 | SOT-23 | 500 mA | 100–600 | High-side switch (con NPN driver) |
| **NPN potencia** | FMMT617 | SOT-23 | 1 A | 200 | Driver de solenoide/ventilador |
| **Par Darlington** | MMBT6427 | SOT-23 | 500 mA | 10 000 | Alta ganancia, señal muy débil |

Aplicación típica — driver de relay con protección:

```
MCU GPIO ──[R 1kΩ 0402]──┬── Base BC817
                          └── [R 10kΩ 0402] ── GND
BC817 Colector ── Relay Coil ── VCC
BC817 Emisor ── GND
En paralelo con Relay Coil: SS14 (Schottky catch, cátodo a VCC)
```

---

##### MOSFETs

| Tipo | Ejemplo | Encapsulado | Vds | Id | Rds(on) | Uso |
|------|---------|-------------|-----|----|---------|-----|
| **N-Ch pequeño** | 2N7002 | SOT-23 | 60 V | 115 mA | Switch lógico, level shifter |
| **N-Ch medio** | AO3400 | SOT-23 | 30 V | 5.7 A | Motor pequeño, switch 3–5 A |
| **N-Ch potencia** | IRLR2905 | DPAK/D2PAK | 55 V | 42 A | Half-bridge, power stage |
| **P-Ch pequeño** | AO3401 | SOT-23 | −30 V | −4 A | Load switch high-side |
| **P-Ch potencia** | IRF9540N | DPAK | −100 V | −19 A | High-side potencia |
| **Dual N+P** | Si1904EDH | SC-70-6 | 20 V | 1.5 A | Load switch compacto (1 chip) |

Regla de selección:
- **Vds ≥ 1.5× Vsistema** (margen de seguridad ante transitorios)
- **Rds(on) @ Vgs = 4.5 V** (para sistemas 5 V) o **@ 2.5 V** (para sistemas 3.3 V)
- **Id ≥ 3× corriente nominal** (calentamiento, seguridad)

---

#### 12.11.3 ICs de Gestión de Potencia

| Categoría | Parte | Encapsulado | Especificación | Uso en robot |
|-----------|-------|-------------|----------------|--------------|
| **LDO 3.3 V** | AMS1117-3.3 | SOT-223 | 1 A, Vdo = 1.2 V | Alimenta lógica desde 5 V |
| **LDO 3.3 V bajo quiescente** | MCP1700 | SOT-23 | 250 mA, Iq = 1.6 µA | Modo sleep, batería |
| **LDO ajustable** | TLV1117 | SOT-223 | 800 mA, adj. | Tensión variable a sensor |
| **Buck 5 V** | MP2307 | SOIC-8 | 3 A, 23 V in | VCC principal desde LiPo |
| **Buck sincrónico** | TPS5430 | SOIC-8 | 3 A, η > 90 % | Eficiencia alta, calentamiento mínimo |
| **Boost 5 V** | MT3608 | SOT-23-6 | 2 A, 2–24 V in | USB desde 3.7 V LiPo |
| **Buck-boost** | TPS63020 | QFN-20 | 1.8 A | Batería descargándose (2–5.5 V → 3.3 V) |
| **Cargador LiPo** | TP4056 | SOP-8 | 1 A, CC/CV | Carga USB de una celda |
| **Cargador + protección** | CN3791 | SOP-8 | MPPT solar | Robot solar |
| **Protección de batería** | DW01A + FS8205 | SOT-23 | OVP, UVP, OCP | Protege celda LiPo |
| **Monitor de corriente** | INA226 | SOIC-8 | 16-bit, I²C, shunt | Telemetría de consumo |
| **Monitor de batería** | MAX17048 | SOT-23-6 | SOC por coulomb counting | % batería en app |
| **Power mux** | TPS2113A | SOT-23-8 | 2A, auto-switch | USB o batería, sin glitch |

---

#### 12.11.4 ICs de Comunicación

| Protocolo | Parte | Encapsulado | Características | Aplicación robot |
|-----------|-------|-------------|-----------------|------------------|
| **UART ↔ USB** | CH340G / CP2102 | SOIC-16 / QFN-28 | 1 Mbps | Programar ESP32, debug |
| **I²C / SPI bridge** | SC18IS602B | SOIC-16 | I²C↔SPI x4 | Ampliar puertos SPI |
| **CAN transceiver** | SN65HVD230 | SOIC-8 | 1 Mbps, 3.3 V | Bus CAN en robot industrial |
| **CAN transceiver HV** | TJA1050 | SOIC-8 | 5 V, ISO 11898 | CAN con MCUs 5 V |
| **RS-485 / UART** | SP3485 | SOIC-8 | Half-duplex, 10 Mbps | Servos Dynamixel, Modbus |
| **RS-485 full-duplex** | MAX485 | SOIC-8 | Robusto, −7/+12 V | Entornos industriales |
| **LIN transceiver** | TLE7259 | SOIC-8 | 20 kbps | Actuadores auto-grade |
| **USB 2.0 PHY** | USB3300 | QFN-32 | HS 480 Mbps | Robot con USB host |
| **Optoacoplador** | PC817 / TLP291 | SOP-4 | 80 mA, CTR > 100 % | Aislamiento galvánico señales |
| **Isolador digital** | ISO7421 | SOIC-8 | 2 canales, 100 Mbps | I²C/SPI aislado, high voltage |
| **Ethernet PHY** | LAN8720A | QFN-24 | RMII, 100BASE-T | Robot en red cableada |

---

#### 12.11.5 ICs Sensores en Encapsulado SMD

| Magnitud | Parte | Encapsulado | Interfaz | Precisión | Uso |
|----------|-------|-------------|----------|-----------|-----|
| **IMU 6-DOF** | MPU-6050 | QFN-24 | I²C/SPI | ±0.05°/s | Estabilización drones, AHRS |
| **IMU 6-DOF alta perf.** | ICM-42688-P | QFN-14 | SPI | ±0.007°/s | Drones de competición |
| **IMU 9-DOF** | ICM-20948 | QFN-24 | I²C/SPI | + magnetómetro | AHRS completo |
| **Magnetómetro** | QMC5883L / HMC5883L | QFN-16 | I²C | 0.2 µT | Brújula, fusión sensorial |
| **Temperatura** | LM75B | SOIC-8 | I²C | ±2°C | Monitor térmico de motor |
| **Temperatura precisa** | MCP9808 | MSOP-8 | I²C | ±0.25°C | Gestión térmica batería |
| **Presión barométrica** | BMP388 | LGA-12 | I²C/SPI | ±0.08 hPa | Altímetro para drone |
| **Presión + humedad** | BME280 | LGA-8 | I²C/SPI | ±1 hPa / ±3 % | Estación meteorológica robot |
| **Efecto Hall (corriente)** | ACS712 | SOIC-8 | Analógico | ±1.5 % | Medición corriente motor |
| **Efecto Hall (posición)** | DRV5056 | SOT-23 | Analógico | lineal 0–VCC | Encoder magnético sin contacto |
| **Encoder magnético** | AS5048A | TSSOP-14 | SPI | 14-bit, 0.022° | Posición absoluta de eje |
| **Luz ambiente + proximidad** | APDS-9960 | LCC-6 | I²C | RGB + gest. | Detección obstáculos |
| **ToF distancia** | VL53L1X | LCC-12 | I²C | ±1 mm, 4 m | Sensor de obstáculos SMD |
| **Sensor de corriente** | INA219 | SOIC-8 | I²C | 12-bit | Telemetría de potencia |
| **ADC externo** | ADS1115 | SOIC-8 | I²C | 16-bit, 4 ch | Leer sensores analógicos lentos |
| **ADC rápido** | MCP3204 | PDIP/SOIC | SPI | 12-bit, 100 ksps | Señales de audio/fuerza |

---

#### 12.11.6 Memoria SMD

| Tipo | Parte | Encapsulado | Capacidad | Interfaz | Retención | Uso |
|------|-------|-------------|-----------|----------|-----------|-----|
| **SPI Flash** | W25Q128JV | SOIC-8 / WSON-8 | 128 Mb (16 MB) | SPI / QSPI | > 20 años | Firmware OTA, SPIFFS, mapas |
| **SPI Flash pequeña** | W25Q32JV | SOIC-8 | 32 Mb (4 MB) | SPI | > 20 años | Config, logs, certificados |
| **I²C EEPROM** | AT24C256 | SOIC-8 | 256 Kb (32 KB) | I²C | > 40 años | Calibración, odometría acumulada |
| **FRAM** | MB85RC256V | SOIC-8 | 256 Kb | I²C | > 10¹² ciclos | Contadores de ciclo, estado crítico |
| **microSD conector** | DM3AT-SF | Push-push | — | SPI / SDIO | — | Logs, mapas SLAM, datasets |
| **PSRAM** | ESP-PSRAM64H | QFN-8 | 64 Mb | SPI | RAM volátil | Buffer de imagen, SLAM |

Uso en ESP32 con Rust:

```rust
// W25Q128 con embedded-hal SPI
use w25q::W25Q;
let flash = W25Q::new(spi, cs).unwrap();
let mut buf = [0u8; 256];
flash.read(0x0000, &mut buf).unwrap();   // leer sector 0
flash.write_page(0x1000, &data).unwrap(); // escribir página
flash.erase_sector(0x1000).unwrap();      // borrar antes de escribir
```

---

#### 12.11.7 Conectores SMD

| Conector | Paso | Corriente | Uso en robótica |
|----------|------|-----------|-----------------|
| **JST-SH** | 1.0 mm | 1 A | Señal: sensores, I²C, SPI (tipo DF13 en drones) |
| **JST-GH** | 1.25 mm | 1 A | Pixhawk / Ardupilot estándar |
| **JST-PH** | 2.0 mm | 2 A | UART, small servo, baterías 1S |
| **JST-XH** | 2.54 mm | 3 A | Baterías LiPo balanceo, cargas medias |
| **JST-VH** | 3.96 mm | 10 A | Motores, fuentes de alto voltaje |
| **USB-C** | — | 3–5 A | Carga de batería, datos, programación |
| **Micro-USB** | — | 1.8 A | Programación / carga (legacy) |
| **FFC / FPC** | 0.5 / 1.0 mm | 0.5 A por pista | Pantallas, cámaras, flex flat |
| **U.FL / IPEX** | — | RF | Antena WiFi / BT / LoRa / GNSS |
| **SMA / RP-SMA** | — | RF | Antena externa, ELRS, LoRa |
| **SIM card (nano)** | — | — | Módulos 4G/LTE para robot remoto |

---

#### 12.11.8 Componentes RF SMD

| Componente | Parte | Función | Uso |
|------------|-------|---------|-----|
| **Balun WiFi/BT** | BALF-NRG-01D3 | Equilibra señal diferencial | ESP32 sin antena integrada |
| **Filtro SAW 2.4 GHz** | SF2093E | Rechaza fuera de banda | Receptor BLE, ZigBee |
| **Filtro SAW 868/915 MHz** | B39871B3726U410 | LoRa band filter | Módulos LoRa DIY |
| **Chip antenna** | 2450AT18A100 | 2.4 GHz cerámica | Cuando no hay hueco para PCB antenna |
| **PCB antenna (trace)** | — | Meander / IFA / F invertida | ESP32, nRF52: gratuita si hay espacio |
| **LNA** | SKY67151 | Amplificador bajo ruido | Aumentar rango GNSS/LoRa |
| **PA** | RFX2401C | Power amplifier 2.4 GHz | Aumentar potencia TX WiFi |

---

#### 12.11.9 Encapsulados SMD — Tabla de Referencia

| Familia | Nombre | Pines | Paso (mm) | Soldadura a mano | Soldadura hot-air | Notas |
|---------|--------|-------|-----------|-----------------|------------------|-------|
| **SOT-23** | Small Outline Transistor | 3–6 | 0.95 | ✅ Fácil | ✅ | Discretos, reguladores pequeños |
| **SOT-89** | — | 3+tab | 1.5 | ✅ | ✅ | TO-92 SMD equiv., disipación media |
| **SOIC-8** | Small Outline IC 8 | 8 | 1.27 | ✅ | ✅ | Muy común: drivers, memoria, sensores |
| **SOIC-16** | — | 16 | 1.27 | ✅ con práctica | ✅ | Bus ICs, bridges |
| **TSSOP-16/20** | Thin Shrink SOP | 16–20 | 0.65 | ⚠️ Difícil | ✅ | MCUs pequeños, FPGAs pequeños |
| **QFP-32/44/64** | Quad Flat Pack | 32–64 | 0.8 | ⚠️ Difícil | ✅ con stencil | MCUs medios |
| **QFN-16/24/32** | Quad Flat No-lead | 16–32 | 0.5 | ❌ Requiere stencil | ✅ | Sensores IMU, power ICs modernos |
| **LGA-12/16** | Land Grid Array | 12–16 | 0.65 | ❌ Solo hot-air/reflow | ✅ | Sensores presión, BME280 |
| **DFN-6/8** | Dual Flat No-lead | 6–8 | 0.5 | ❌ | ✅ | Reguladores modernos, MOSFETs |
| **TO-252 / DPAK** | Transistor Outline | 3+tab | 2.28 | ✅ | ✅ | MOSFETs potencia, LDOs > 1 A |
| **D2PAK / TO-263** | — | 3+tab | 2.28 | ✅ | ✅ | MOSFETs alta potencia |
| **BGA** | Ball Grid Array | 16–1000+ | 0.4–1.0 | ❌ Imposible a mano | ❌ Horno reflow | CPUs, SoCs, FPGAs avanzados |
| **SC-70 / SOT-363** | — | 6 | 0.65 | ⚠️ Lupa necesaria | ✅ | MOSFETs duales, comparadores |
| **SOD-123** | Diode | 2 | 1.0 | ✅ | ✅ | Diodos 1N4148W, Schottky |
| **SMA/SMB/SMC** | Diode power | 2 | — | ✅ | ✅ | TVS, diodos de potencia |

---

#### 12.11.10 Herramientas para Trabajo SMD

##### Soldadura

| Herramienta | Especificación recomendada | Para qué |
|-------------|---------------------------|----------|
| **Estación de soldadura** | ≥ 65 W, control digital (JBC, Hakko FX-951) | Todo — base obligatoria |
| **Punta tipo D (bisel)** | 0.5–1.6 mm | SOIC, pasivos 0402+ |
| **Punta tipo I (cónica fina)** | 0.2 mm | Rework de pines individuales |
| **Estación hot-air** | 858D / Quick 861DW, 100 l/min ajustable | QFN, LGA, retirar chips |
| **Boquilla hot-air** | 4–8 mm diámetro | Ajustar según tamaño IC |
| **Placa precalentadora** | 200 W, hasta 180°C | Previene warping en PCBs gruesas |

##### Consumibles

| Consumible | Especificación | Función |
|------------|----------------|---------|
| **Estaño sin plomo** | SAC305, 0.5 mm ⌀ | ROHS, buenas propiedades |
| **Flux de limpieza** | No-clean RMA (MG Chemicals 835) | Facilita flujo, reduce bridges |
| **Flux gel/pasta** | MG Chemicals 8341 | Hot-air, rework de QFN |
| **Malla de desoldar** | 1.5–2.0 mm | Retirar estaño en exceso |
| **Bomba de desoldar** | Anti-estática | Quick removal |
| **Alcohol isopropílico ≥ 99 %** | IPA 99 % | Limpiar flux post-soldadura |
| **Pasta de soldar** | SAC305 T4 / T5 en jeringa | Stencil printing |
| **Stencil** | Acero inoxidable 0.12 mm | Aplicar pasta uniformemente |

##### Inspección

| Herramienta | Especificación | Función |
|-------------|----------------|---------|
| **Microscopio digital** | 10–60×, USB (Andonstar AD207) | Inspección de bridges, cold joints |
| **Lupa con luz** | 10× con LED anular | Inspección rápida |
| **Multímetro** | Autorango, diodo, continuidad | Verificar soldaduras, nets |
| **Osciloscopio** | ≥ 100 MHz, 2 ch (DS1054Z) | Debug de señales digitales |
| **Analizador lógico** | 8 ch, 24 MHz (Saleae / clone) | I²C, SPI, UART, CAN |
| **Termómetro IR** | −50–380°C | Punto caliente en placa |

##### Proceso de soldadura SMD con stencil

```
1. Alinear stencil sobre PCB desnuda con pines guía o cinta Kapton
2. Extender pasta de soldar (SAC305 T4) con espátula en ángulo 45°
3. Retirar stencil verticalmente en un solo movimiento
4. Inspeccionar con lupa: pasta en todos los pads, sin bridges
5. Colocar componentes con pinzas SMD (comenzar por los más pequeños)
6. Reflow:
   a. Horno: perfil SAC305 — rampa 2°C/s → 150°C, soak 60 s,
              pico 245°C 10 s, enfriamiento 4°C/s
   b. Hot-air (prototipo): 350°C, caudal bajo, circular, ≈15 s por zona
7. Inspeccionar con microscopio — buscar: cold joints, bridges, tombstoning
8. Limpiar con IPA 99 % y pincel ESD
9. Test funcional antes de añadir la siguiente capa
```

---

#### 12.11.11 Reglas de Diseño KiCad para SMD

##### Courtyard (zona de exclusión)

```
Encapsulado        | Courtyard extra
0402               | 0.25 mm
0603               | 0.25 mm
0805               | 0.25 mm
SOT-23             | 0.25 mm
SOIC-8             | 0.25 mm
QFN-24             | 0.50 mm
BGA                | 0.50 mm
Conectores JST     | 1.00 mm (movimiento de cable)
Conector USB       | 1.00 mm
```

##### Thermal reliefs (conexión a plano GND)

```python
# En KiCad 8, vía pcbnew scripting:
import pcbnew

board = pcbnew.LoadBoard("robot_control.kicad_pcb")
settings = board.GetDesignSettings()

# Thermal relief: spoke width mínimo para SMD en plano GND
settings.m_ThermalReliefMinHoleDiameter = pcbnew.FromMM(0.3)
settings.m_ThermalReliefSpokeWidth = pcbnew.FromMM(0.25)

# Para QFN (pad térmico central), usar conexión sólida al plano:
# en el footprint, pad de calor → zona de cobre sólida, sin spokes
```

##### Pasta de soldar (F.Paste / B.Paste)

```
Componente         | Reducción de paste mask
Resistencias 0402  | −10 % (paste = 90 % del pad)
Condensadores 0402 | −10 %
SOIC-8             | −10 %
QFN pad térmico    | Cuadrícula 0.5 mm, 25 % de cobre expuesto
BGA                | −20 %, usar stencil profesional
```

##### Via-in-pad para QFN

```
Si el layout requiere via bajo el pad térmico del QFN:
1. Via tapada con resina (tented via) — pedir a JLCPCB "via plugging"
2. Tamaño via: drill 0.3 mm, anular 0.15 mm (JLCPCB mínimo)
3. Cantidad: 4–9 vias en cuadrícula para QFN 5×5 mm
4. Plano GND en cara inferior del PCB para disipación máxima
```

---

#### 12.11.12 Laboratorio Práctico SMD — Secuencia de Aprendizaje

```
Semana 1 — Pasivos grandes
  └── Soldar resistencias 0805, condensadores 0805 en placa de práctica
      Herramienta: soldador + punta D + flux
      Verificar: multímetro continuidad, inspección visual

Semana 2 — Pasivos pequeños
  └── Resistencias 0402, condensadores 0402
      Técnica: "drag soldering" con flux generoso
      Verificar: microscopio digital

Semana 3 — Discretos y SOIC
  └── SOT-23 (transistores, LDOs), SOIC-8 (TP4056, INA226)
      Técnica: una pata → alinear → resto de patas → flux + arrastre
      Verificar: test funcional del circuito

Semana 4 — ICs con pad térmico
  └── QFN-24 (sensores IMU), DFN-8 (reguladores)
      Técnica: stencil + pasta + hot-air reflow
      Verificar: microscopio + test I²C (i2cdetect)

Semana 5 — PCB completa de robot
  └── Placa custom diseñada en M12 (KiCad) con mix de encapsulados
      Proceso completo: stencil → pick & place → hot-air → inspección → test
      Entregable: robot funcional con PCB propia soldada por el estudiante
```

---

#### Checklist SMD de producción

- [ ] DRC en KiCad: cero errores de courtyard, soldermask, paste
- [ ] Revisión de footprints: todos los pads coinciden con datasheet
- [ ] Pasta aplicada con stencil: cobertura ≥ 90 % sin bridges
- [ ] Todos los componentes alineados antes de reflow
- [ ] Inspección post-reflow con microscopio (× 40 mínimo)
- [ ] Prueba de continuidad GND → VCC: > 1 MΩ (sin cortocircuito)
- [ ] Test funcional: encendido, comunicación I²C/SPI, medición de tensiones
- [ ] Limpieza IPA 99 % y secado 24 h antes de conformal coating

---

### 12.12 Energía para Robótica — LiPo · Li-ion · Voltajes · Instrumentación

> Entender la energía es entender los límites reales del robot. Esta sección cubre
> desde la química de las celdas hasta las técnicas de medición con multímetro y osciloscopio.

---

#### 12.12.1 Tipos de baterías usadas en robótica

| Química | Tensión nominal | Tensión cargada | Tensión mínima | Densidad energética | Descarga máxima |
|---------|----------------|-----------------|----------------|--------------------|-----------------| 
| **LiPo** | 3.7 V/celda | 4.2 V/celda | 3.0 V/celda | 150–200 Wh/kg | 20–100 C |
| **Li-ion 18650** | 3.6 V/celda | 4.2 V/celda | 2.5 V/celda | 200–265 Wh/kg | 2–10 C |
| **LiFePO4** | 3.2 V/celda | 3.65 V/celda | 2.5 V/celda | 90–120 Wh/kg | 3–5 C |
| **NiMH** | 1.2 V/celda | 1.4 V/celda | 1.0 V/celda | 60–120 Wh/kg | 10–20 C |
| **NiCd** | 1.2 V/celda | 1.4 V/celda | 1.0 V/celda | 40–60 Wh/kg | 20–40 C |

**Notación de paquetes LiPo:**
```
3S2P — 3 celdas en Serie × 2 grupos en Paralelo
  3S → tensión nominal: 3 × 3.7 = 11.1 V
  2P → capacidad doble: 2× Ah de una celda
  C rating → 1000 mAh × 25C = 25 A máx. descarga continua
```

**Cuándo usar cada tipo:**
- **LiPo alta C** → drones FPV, combat robots (potencia instantánea)
- **Li-ion 18650** → robot móvil autónomo (autonomía, densidad energética)
- **LiFePO4** → robot industrial (seguridad, > 2000 ciclos)

---

#### 12.12.2 Sistemas de tensión en robots

```
Robot diferencial típico (2S LiPo = 7.4 V nominal):

7.4 V (bat) ┬── Buck 5 V ──┬── ESP32 (3.3 V LDO interno)
            │              ├── Servo (5 V)
            │              └── LED strips (5 V)
            │
            ├── Buck 12 V (boost) ── Actuadores lineales
            │
            └── Directo 7.4 V ──── Motores DC (via H-bridge)

Drone quadrotor (4S LiPo = 14.8 V nominal):

14.8 V (bat) ┬── Directo ──── ESCs → Motores BLDC
             └── BEC 5 V ──── FC (Flight Controller), servos, Rx
```

**Niveles de tensión y su uso:**

| Tensión | Fuente | Uso típico |
|---------|--------|-----------|
| 3.3 V | LDO desde 5 V | MCU, sensores I²C/SPI, nivel lógico |
| 5 V | Buck desde batería | Servos, USB, nivel lógico 5 V, cámaras |
| 7.4 V | 2S LiPo directo | Motores DC pequeños/medianos |
| 11.1 V | 3S LiPo directo | Motores BLDC, drivers potencia |
| 14.8 V | 4S LiPo directo | Drones de competición, motores potentes |
| 24 V | Pack Li-ion / PSU | AGV industrial, brazos pesados |

---

#### 12.12.3 Gestión y protección de baterías

**BMS — Battery Management System:**
```
Funciones del BMS:
  ├── OVP — Overvoltage Protection  (corta carga si V > 4.25 V/celda)
  ├── UVP — Undervoltage Protection (corta descarga si V < 2.9 V/celda)
  ├── OCP — Overcurrent Protection  (corta si I > I_max)
  ├── SCP — Short Circuit Protection (respuesta < 1 µs)
  ├── Balanceo pasivo / activo entre celdas
  └── Comunicación: SMBus / I²C / CAN → datos al MCU
```

ICs de BMS para el curso:
- **DW01A + FS8205**: 1S, protección básica, SOIC pequeño
- **S8254AA**: 2S, OVP + UVP + OCP
- **BQ29700**: multi-celda, comunicación I²C
- **BQ76920**: 3–5S, integrado, para robot de producción

**Monitor de consumo en tiempo real:**

```rust
// INA226 via I²C — monitorea V y I del robot completo
use ina226::INA226;

let mut monitor = INA226::new(i2c, 0x40);
monitor.set_calibration(0.01, 5.0)?;  // shunt 10mΩ, max 5A

loop {
    let bus_v  = monitor.bus_voltage_mv()?;
    let current = monitor.current_ma()?;
    let power   = monitor.power_mw()?;
    let soc_pct = (bus_v as f32 - 3000.0) / (4200.0 - 3000.0) * 100.0;

    mqtt_publish("telemetry", json!({
        "voltage_mv": bus_v,
        "current_ma": current,
        "power_mw":   power,
        "soc_pct":    soc_pct.clamp(0.0, 100.0),
    }));

    FreeRtos::delay_ms(500);
}
```

---

#### 12.12.4 Carga segura de baterías

**Perfil de carga CC/CV (Constant Current / Constant Voltage):**
```
Fase 1 — CC (corriente constante):
  I = C/2 típicamente (ej. 500 mA para batería de 1000 mAh)
  Tensión sube de 3.0 V → 4.2 V/celda

Fase 2 — CV (tensión constante):
  V = 4.20 V/celda (±0.05 V)
  Corriente cae hasta C/10 → fin de carga

Reglas de oro:
  ✅ Nunca cargar LiPo < 0°C o > 45°C
  ✅ Tensión de carga máxima: 4.20 V/celda (no 4.25 V)
  ✅ Nunca descargar bajo 3.0 V/celda bajo carga
  ✅ Almacenamiento: 3.7–3.8 V/celda (50–60 % SOC)
  ❌ Nunca dejar cargando sin supervisión con cargador desconocido
```

**Cargadores recomendados por aplicación:**

| Cargador | Tipo | Uso en el curso |
|----------|------|----------------|
| TP4056 (SMD) | 1S LiPo, 1 A | PCB propia, robot small |
| IMAX B6 AC | Multi-química, 6 A | Laboratorio, LiPo de drones |
| Isdt Q6 Pro | 14 A, balance | Producción, ciclos rápidos |
| CN3791 (SMD) | Solar MPPT | Robot exterior autónomo |

---

#### 12.12.5 Instrumentación — Multímetro y Osciloscopio

##### Multímetro digital — guía de uso en robótica

| Función | Cómo usarlo | Precaución |
|---------|-------------|-----------|
| **Voltaje DC** | Paralelo al componente. Rojo → + , Negro → GND | Rango > tensión medida |
| **Voltaje AC** | Igual que DC | No medir VBAT directo en rango AC |
| **Continuidad** | Mide resistencia, beep < 30 Ω | Circuito SIN energizar |
| **Diodo** | Ánodo → Rojo, Cátodo → Negro; Vf 0.2–0.7 V = OK | — |
| **Resistencia** | Paralelo al componente | Sin energizar; desconectar un extremo |
| **Corriente DC** | EN SERIE con la carga | Nunca paralelo: funde el fusible |
| **Temperatura** | Con termopar tipo K | Rango −50 a 400°C |

**Diagnóstico rápido de circuito de robot con multímetro:**
```
1. Verificar tensión de batería: V_batt ≥ 3.0 V/celda × N celdas
2. Continuidad GND entre módulos: beep = bus GND correcto
3. Tensión en reguladores: 3.3 V ± 0.1 V en VCC del ESP32
4. Corriente total en boot: mide en serie entre bat y PCB (< 500 mA en idle)
5. Buscar cortocircuito: resistencia batería+ → GND debe ser > 1 MΩ
```

##### Osciloscopio — señales digitales de robots

```
Señales típicas a medir en un robot:

PWM de motor:
  Freq: 20 kHz, Duty 0–100 %
  Canal 1 → señal PWM, trigger → flanco ascendente
  Medir: Ton, Toff, frecuencia, Vhigh (debe ser 3.3 V)

UART (debug ESP32):
  115200 baud → 1 bit ≈ 8.68 µs
  Trigger: flanco descendente (bit de inicio)
  Decodificar: modo Serial del osciloscopio

I²C:
  Dos canales: SDA + SCL
  Trigger: SDA flanco descendente mientras SCL alto (START)
  Medir: frecuencia (100 kHz o 400 kHz), ACK bits

PWM de servo:
  Freq: 50 Hz, pulso 1.0 ms (0°) → 2.0 ms (180°)
  Zoom en el pulso: medir ancho con cursores

SBUS (radio control):
  Freq: 100 Hz, UART invertido 100000 baud
  Trigger: nivel invertido; decodificar en modo Serial inv.
```

**Osciloscopio recomendado para el curso:**
- **Rigol DS1054Z** — 4 ch, 50 MHz, decodificación UART/I²C/SPI integrada (~$350)
- **DSO138 (kit)** — 1 ch, 200 kHz, para aprender conceptos básicos (~$20)
- **Saleae Logic 8** — analizador lógico 8 ch, 100 MHz, excelente para I²C/SPI (~$150)

---

#### 12.12.6 Cálculo de autonomía del robot

```python
def robot_autonomy(
    battery_mah: float,
    battery_cells: int,
    motors_current_a: float,
    electronics_ma: float = 300,
    efficiency: float = 0.85
) -> dict:
    """
    battery_mah: capacidad total de la batería en mAh
    motors_current_a: corriente promedio de los motores en operación normal
    efficiency: eficiencia del sistema de potencia (reguladores, cableado)
    """
    total_current_ma = (motors_current_a * 1000) + electronics_ma
    effective_capacity = battery_mah * efficiency
    autonomy_h = effective_capacity / total_current_ma
    autonomy_min = autonomy_h * 60

    cell_v_nominal = 3.7
    energy_wh = (battery_mah / 1000) * battery_cells * cell_v_nominal

    return {
        "autonomy_min":    round(autonomy_min, 1),
        "total_current_ma": round(total_current_ma, 1),
        "energy_wh":       round(energy_wh, 2),
        "motors_pct":      round(motors_current_a * 1000 / total_current_ma * 100, 1),
    }

# Ejemplo: robot diferencial con 2S 2200 mAh, motores 0.8 A promedio
result = robot_autonomy(2200, 2, 0.8)
# → {'autonomy_min': 87.5, 'total_current_ma': 1100, 'energy_wh': 16.28, 'motors_pct': 72.7}
```

---

### 12.13 Pantallas LCD y OLED para Robótica

> Las pantallas convierten un robot ciego en un sistema con retroalimentación local:
> estado de la batería, modo de operación, posición y diagnósticos sin necesidad
> de conectar un computador.

---

#### 12.13.1 Tipos de pantallas usados en robótica

| Tipo | Controlador | Interfaz | Tamaño típico | Resolución | Ventaja | Uso robótico |
|------|-------------|----------|---------------|-----------|---------|--------------|
| **LCD alfanumérico** | HD44780 | Paralelo / I²C (PCF8574) | 16×2, 20×4 | Caracteres | Económico, sin código especial | Menú de configuración, estado |
| **LCD gráfico** | ST7920 / KS0108 | SPI / paralelo | 128×64 px | 128×64 | Sin PWM backlight | Gráficas simples de sensores |
| **OLED monocromático** | SSD1306 | I²C / SPI | 0.96" / 1.3" | 128×64 px | Contraste infinito, muy pequeño | Robot pequeño, wearable, telemetría |
| **OLED color** | SSD1351 / SSD1331 | SPI | 1.5" / 0.95" | 128×128 px | Color RGB | Iconos de estado, cámara thumbnail |
| **TFT color** | ILI9341 / ST7789 | SPI | 2.0"–3.5" | 240×320 px | Color + táctil opcional | Dashboard local completo |
| **TFT + táctil** | ILI9488 + XPT2046 | SPI dual | 3.5" | 480×320 px | Interfaz táctil | Panel de operador |
| **E-Ink** | UC8151 / SSD1681 | SPI | 1.54"–4.2" | 200×200 px | Cero consumo en standby | Etiqueta de estado, AGV |
| **Nextion** | Propietario | UART | 2.4"–7" | Variable | GUI drag-and-drop | HMI sin programar UI en MCU |

---

#### 12.13.2 LCD 16×2 con I²C (PCF8574)

El adaptador I²C permite controlar el LCD con sólo 2 pines (SDA + SCL).

```rust
// LCD HD44780 via PCF8574 I²C en Rust
use lcd_lcm1602_i2c::Lcd;
use esp_idf_hal::i2c::I2cDriver;

let mut lcd = Lcd::new(i2c, 0x27).unwrap();  // dirección por defecto PCF8574
lcd.init().unwrap();
lcd.set_cursor(0, 0).unwrap();
lcd.print("Robot v1.0").unwrap();
lcd.set_cursor(1, 0).unwrap();
lcd.print(&format!("Bat: {:.1}V", bat_v)).unwrap();
```

**Caracteres personalizados (ícono de batería):**
```rust
// Definir carácter custom en CGRAM
const BATT_FULL: [u8; 8] = [
    0b01110,
    0b11111,
    0b11111,
    0b11111,
    0b11111,
    0b11111,
    0b11111,
    0b11111,
];
lcd.create_char(0, &BATT_FULL).unwrap();
lcd.print_char(0).unwrap();  // imprime el ícono de batería llena
```

---

#### 12.13.3 OLED SSD1306 — La pantalla estándar del maker

La más usada en robots pequeños: 0.96" I²C, 128×64 px, consume < 20 mA.

```rust
// OLED SSD1306 en Rust con embedded-graphics
use ssd1306::{prelude::*, I2CDisplayInterface, Ssd1306};
use embedded_graphics::{
    mono_font::{ascii::FONT_6X10, MonoTextStyleBuilder},
    pixelcolor::BinaryColor,
    prelude::*,
    text::Text,
    primitives::{Rectangle, PrimitiveStyle},
};

let interface = I2CDisplayInterface::new(i2c);
let mut display = Ssd1306::new(interface, DisplaySize128x64, DisplayRotation::Rotate0)
    .into_buffered_graphics_mode();
display.init().unwrap();

// Panel de telemetría
let style = MonoTextStyleBuilder::new()
    .font(&FONT_6X10)
    .text_color(BinaryColor::On)
    .build();

loop {
    display.clear(BinaryColor::Off).unwrap();

    Text::new(&format!("Bat: {:.1}%", soc), Point::new(0, 10), style)
        .draw(&mut display).unwrap();
    Text::new(&format!("V:  {:.2} m/s", vel), Point::new(0, 22), style)
        .draw(&mut display).unwrap();
    Text::new(&format!("Yaw:{:.1} deg", yaw), Point::new(0, 34), style)
        .draw(&mut display).unwrap();

    // Barra de batería gráfica
    let bar_w = (soc / 100.0 * 80.0) as u32;
    Rectangle::new(Point::new(0, 50), Size::new(bar_w, 10))
        .into_styled(PrimitiveStyle::with_fill(BinaryColor::On))
        .draw(&mut display).unwrap();

    display.flush().unwrap();
    FreeRtos::delay_ms(100);
}
```

---

#### 12.13.4 TFT ILI9341 — Panel de control a color

```rust
// TFT ILI9341 SPI en Rust
use ili9341::{DisplaySize240x320, Ili9341, Orientation};
use embedded_graphics::pixelcolor::Rgb565;

let mut tft = Ili9341::new(spi, cs, dc, &mut rst, &mut delay,
                            DisplaySize240x320).unwrap();
tft.set_orientation(Orientation::Landscape).unwrap();

// Fondo negro
tft.clear(Rgb565::BLACK).unwrap();

// Texto de estado
let style = MonoTextStyleBuilder::new()
    .font(&FONT_10X20)
    .text_color(Rgb565::GREEN)
    .background_color(Rgb565::BLACK)
    .build();

Text::new("ROBOT ONLINE", Point::new(10, 30), style)
    .draw(&mut tft).unwrap();

// Gauge de batería con color semafórico
let color = match soc as u32 {
    75..=100 => Rgb565::GREEN,
    30..=74  => Rgb565::YELLOW,
    _        => Rgb565::RED,
};
Rectangle::new(Point::new(10, 60), Size::new((soc * 2.0) as u32, 20))
    .into_styled(PrimitiveStyle::with_fill(color))
    .draw(&mut tft).unwrap();
```

---

#### 12.13.5 Python — Pantalla desde el PC (supervisión remota)

```python
# Dashboard en terminal con rich (sin pantalla física)
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
import time

def make_dashboard(state: dict) -> Panel:
    table = Table(show_header=False)
    table.add_row("🔋 Batería",  f"[green]{state['soc']:.1f}%[/] — {state['voltage_mv']/1000:.2f} V")
    table.add_row("⚡ Velocidad", f"{state['linear_vel']:.2f} m/s")
    table.add_row("🧭 Rumbo",    f"{state['yaw_deg']:.1f}°")
    table.add_row("🌡️ Temp",     f"{state['motor_temp']}°C")
    table.add_row("⏱ Uptime",   f"{state['uptime_s']} s")
    return Panel(table, title="[bold cyan]Robot-01 — Telemetría en vivo[/]")

with Live(refresh_per_second=10) as live:
    while True:
        state = robot.state.get("telemetry", {})
        live.update(make_dashboard(state))
        time.sleep(0.1)
```

---

#### 12.13.6 Selección de pantalla por proyecto

| Proyecto del curso | Pantalla recomendada | Justificación |
|--------------------|---------------------|---------------|
| Robot diferencial (Kit 1) | SSD1306 0.96" OLED | Pequeño, bajo consumo, telemetría básica |
| Brazo manipulador (Kit 3) | ILI9341 2.4" TFT | Posición de cada eje en tiempo real |
| Combat robot | Sin pantalla | Peso y espacio críticos |
| AGV logístico (Kit 6) | Nextion 3.5" | Menú de misiones sin programar |
| Robot relacional (Kit 5) | SSD1306 + LEDs RGB | Expresión facial simple |
| Dron (Kit 4) | Sin pantalla / OSD (Max7456) | OSD integrado en la señal de vídeo FPV |

---

## Módulo 10 — Control: Radio · Web · Mobile
*El robot se controla desde cualquier dispositivo, en cualquier distancia*

> Este módulo convierte cada robot del curso en un sistema teleoperado completo:
> control por radiofrecuencia, programación por secuencias desde el navegador
> y app móvil nativa (iOS) o multiplataforma (Android + iOS).

---

### 10.1 Radio Control — Frecuencia y Protocolos

#### Espectro y protocolos disponibles

| Protocolo | Frecuencia | Alcance | Latencia | Uso ideal |
|-----------|-----------|---------|----------|-----------|
| ExpressLRS (ELRS) | 2.4 GHz | 500m–2km | < 5 ms | FPV racing, robots ágiles |
| ELRS largo alcance | 868/915 MHz | 5–30 km | 10–20 ms | Drones de trabajo, AGV exterior |
| LoRa (SX1278) | 433/868 MHz | 2–15 km | 50–200 ms | Telemetría, comandos lentos |
| NRF24L01+ | 2.4 GHz | 100m | 2 ms | Robots de competición |
| Bluetooth BLE | 2.4 GHz | 10–50 m | 10–30 ms | Control cercano, indoor |

#### Lectura de señales RC en ESP32 con Rust

```rust
// Parser de trama SBUS (100 kbps, UART invertido)
struct SbusParser { buf: [u8; 25], idx: usize }

impl SbusParser {
    fn feed(&mut self, byte: u8) -> Option<[u16; 16]> {
        if byte == 0x0F { self.idx = 0; }
        self.buf[self.idx] = byte;
        self.idx += 1;
        if self.idx == 25 { Some(self.decode()) } else { None }
    }

    fn decode(&self) -> [u16; 16] {
        // Desempaqueta 11 bits por canal desde 25 bytes SBUS
        let b = &self.buf;
        let mut ch = [0u16; 16];
        ch[0] = ((b[1] as u16) | ((b[2] as u16) << 8)) & 0x07FF;
        ch[1] = ((b[2] as u16 >> 3) | ((b[3] as u16) << 5)) & 0x07FF;
        // ... canales 2–15
        ch
    }
}
```

#### Configuración de canales y mezclas

- **Robot diferencial**: CH1 = velocidad lineal, CH2 = giro (mezcla por software en Rust)
- **Brazo 4-DOF**: CH1–CH4 = articulaciones, CH5 = abrir/cerrar gripper, CH6 = speed
- **Dron**: throttle, pitch, roll, yaw + modos de vuelo en CH5
- **Failsafe**: comportamiento predefinido en Rust si no llegan tramas en 100 ms
  - Diferencial: parar motores
  - Dron: activar return-to-home
  - Brazo: mantener posición actual

#### LoRa para largo alcance (SX1278 + ESP32 + Rust)

```rust
// Envío de telemetría cada 500 ms por LoRa
let payload = TelemetryPacket {
    robot_id: ROBOT_ID,
    battery_mv: adc.read_battery(),
    position: odometry.get_pose(),
    status: state_machine.current(),
};
lora.transmit(&payload.to_bytes()).await?;
```

- Uplink (control): comandos de movimiento compactos (< 20 bytes)
- Downlink (telemetría): batería, posición GPS, estado de misión
- Spread factor adaptativo: SF7 para velocidad, SF12 para máximo alcance

---

### 10.2 Control Web Punto a Punto

#### Arquitectura según hardware disponible

```
Opción A — ESP32 standalone (sin Raspberry Pi):
  Navegador → WebSocket → ESP32 HTTP server → motores/servos

Opción B — ESP32 + Raspberry Pi (más capacidad):
  Navegador → WebSocket → FastAPI (RPi) → MQTT → ESP32 → actuadores
```

#### ESP32 como servidor web embebido (Rust + embassy-net)

- HTML/CSS/JS servido desde flash (SPIFFS / LittleFS)
- WebSocket bidireccional: comandos en JSON → ESP32, estado del robot → UI
- mDNS: el robot aparece como `robot-01.local` en la red local sin IP fija
- Autenticación básica: PIN de 6 dígitos para evitar control no autorizado

#### Interface de secuencias — "Programación paso a paso"

El concepto central: el usuario define una secuencia de movimientos en el navegador
y el robot la ejecuta en orden, con confirmación visual de cada paso. Puede repetirla N veces.

```
┌──────────────────────────────────────────────────┐
│  SECUENCIA: Ciclo de ensamble #3          [▶ Run] │
│  Repeticiones: [5]            Estado: En espera   │
├──────────────────────────────────────────────────┤
│  ① Mover a posición HOME              ✅ OK      │
│  ② Adelante 30 cm                     ⏳ ...     │
│  ③ Girar derecha 90°                  ⬜         │
│  ④ Bajar brazo → posición recogida    ⬜         │
│  ⑤ Cerrar gripper                     ⬜         │
│  ⑥ Subir brazo → posición transporte  ⬜         │
│  ⑦ Girar izquierda 90°                ⬜         │
│  ⑧ Adelante 15 cm                     ⬜         │
│  ⑨ Abrir gripper                      ⬜         │
│  ⑩ Volver a HOME                      ⬜         │
├──────────────────────────────────────────────────┤
│  [+ Agregar paso]  [Guardar]  [Exportar JSON]    │
└──────────────────────────────────────────────────┘
```

**Modos de ejecución:**
- **Manual paso a paso**: el usuario presiona "Siguiente" después de cada paso
- **Automático**: el robot avanza al siguiente paso cuando confirma éxito del anterior
- **Repetición**: ejecuta la secuencia completa N veces (útil para automatización repetitiva)
- **Pausa / emergencia**: detención inmediata en cualquier momento

**Backend Python (FastAPI) para secuencias complejas:**

```python
from fastapi import FastAPI, WebSocket
from asyncio import Queue

app = FastAPI()
command_queue: Queue[RobotCommand] = Queue()

@app.websocket("/ws/control")
async def control_ws(ws: WebSocket):
    await ws.accept()
    async for msg in ws.iter_json():
        if msg["type"] == "sequence":
            for step in msg["steps"]:
                await command_queue.put(RobotCommand(**step))
                result = await wait_for_ack(timeout=30)
                await ws.send_json({"step": step["id"], "status": result})

@app.post("/sequence/{name}/run")
async def run_sequence(name: str, repeat: int = 1):
    seq = await db.get_sequence(name)
    for _ in range(repeat):
        for step in seq.steps:
            await command_queue.put(step)
    return {"queued": len(seq.steps) * repeat}
```

**Almacenamiento de secuencias:**
- JSON en el filesystem del ESP32 (SPIFFS) para secuencias simples
- SQLite en Raspberry Pi para historial completo con timestamps y logs de ejecución
- Exportar/importar secuencias en formato JSON desde la UI

---

### 10.3 App iOS con Swift

#### Stack tecnológico

| Tecnología | Uso |
|-----------|-----|
| SwiftUI | Interfaz declarativa: joystick, telemetría, secuencias |
| CoreBluetooth | Conexión BLE directa al ESP32 |
| CocoaMQTT | Cliente MQTT para control por red WiFi |
| AVFoundation | Streaming de video MJPEG desde la cámara del robot |
| Combine | Reactividad: estado del robot → UI en tiempo real |
| SwiftData | Persistencia local de secuencias y configuraciones |

#### Componentes principales de la app

```swift
// Joystick virtual con SwiftUI + haptic feedback
struct JoystickView: View {
    @State private var offset = CGSize.zero
    let onMove: (Float, Float) -> Void  // (linear, angular)

    var body: some View {
        Circle()
            .fill(.blue.opacity(0.3))
            .frame(width: 150, height: 150)
            .overlay(
                Circle().fill(.blue).frame(width: 50, height: 50)
                    .offset(offset)
            )
            .gesture(DragGesture()
                .onChanged { v in
                    offset = clamp(v.translation, radius: 50)
                    let lin = Float(-offset.height / 50)
                    let ang = Float(-offset.width  / 50)
                    UIImpactFeedbackGenerator(style: .light).impactOccurred()
                    onMove(lin, ang)
                }
                .onEnded { _ in
                    offset = .zero
                    onMove(0, 0)
                }
            )
    }
}
```

**Pantallas de la app:**
1. **Scanner**: descubre robots en la red BLE/WiFi
2. **Dashboard**: telemetría en tiempo real (batería, velocidad, pose)
3. **Joystick**: control manual con joystick virtual + botones de acción
4. **Secuencias**: editor y reproductor de secuencias paso a paso
5. **Cámara**: streaming de video + overlay de datos de visión
6. **Configuración**: parámetros PID, velocidad máxima, failsafe

**Conexión dual BLE + MQTT:**
```swift
class RobotConnection: ObservableObject {
    private let ble = BLEManager()
    private let mqtt = MQTTClient(host: "robot.local", port: 1883)

    func send(_ cmd: RobotCommand) {
        // BLE si está cerca y conectado; MQTT como fallback por WiFi
        if ble.isConnected {
            ble.send(cmd.toBytes())
        } else {
            mqtt.publish("robots/\(robotId)/commands", cmd.toJSON())
        }
    }
}
```

---

### 10.4 App Multiplataforma con React Native (Android + iOS)

#### Stack tecnológico

| Librería | Función |
|---------|---------|
| React Native + Expo | Base multiplataforma |
| `mqtt.js` + WebSocket | Control por MQTT en red WiFi |
| `react-native-ble-manager` | Control por BLE |
| `react-native-joystick` | Joystick táctil |
| Zustand | Estado global en tiempo real |
| React Navigation | Navegación entre pantallas |
| Expo Notifications | Alertas push del robot |
| Expo Camera | Preview de cámara del robot vía MJPEG |

#### Arquitectura de la app

```
┌─────────────────────────────────────────┐
│           React Native App              │
│  ┌──────────┐  ┌──────────────────────┐ │
│  │ Joystick │  │ Sequence Editor      │ │
│  │ Screen   │  │ (drag & drop steps)  │ │
│  └──────────┘  └──────────────────────┘ │
│  ┌──────────┐  ┌──────────────────────┐ │
│  │ Camera   │  │ Fleet Dashboard      │ │
│  │ Stream   │  │ (múltiples robots)   │ │
│  └──────────┘  └──────────────────────┘ │
└──────────────────────────────────────────┘
         │ MQTT/WebSocket    │ BLE
    ┌────┴────┐         ┌────┴────┐
    │ Broker  │         │ ESP32   │
    │ MQTT    │         │ directo │
    └─────────┘         └─────────┘
```

**Publicación de comandos en tiempo real:**

```javascript
// Hook personalizado para control del robot vía MQTT
function useRobotControl(robotId) {
  const client = useMQTTClient();

  const sendVelocity = useCallback(
    throttle((linear, angular) => {
      client.publish(
        `robots/${robotId}/cmd_vel`,
        JSON.stringify({ linear, angular }),
        { qos: 0 }  // QoS 0 para mínima latencia
      );
    }, 50),  // máximo 20 Hz
    [robotId]
  );

  const runSequence = (steps, repeat = 1) =>
    client.publish(
      `robots/${robotId}/sequence`,
      JSON.stringify({ steps, repeat }),
      { qos: 1 }  // QoS 1 garantiza entrega de secuencias
    );

  return { sendVelocity, runSequence };
}
```

**Editor de secuencias en React Native:**
- Lista drag & drop de pasos (reordenar con `react-native-draggable-flatlist`)
- Selector de tipo de paso: mover, girar, esperar, acción del brazo, etc.
- Parámetros numéricos con slider o teclado numérico
- Guardar en AsyncStorage + sincronizar al broker MQTT
- Compartir secuencia como QR code (otro teléfono la importa y ejecuta)

**Notificaciones push:**
```javascript
// El robot publica alertas → servidor las reenvía como push notification
const ALERTS = {
  'battery_low':    { title: 'Batería baja',    body: 'Robot al 15% — cargar pronto' },
  'mission_done':   { title: 'Misión completa', body: 'Secuencia ejecutada 5/5 veces' },
  'obstacle':       { title: 'Obstáculo',       body: 'Robot detenido — revisar ruta' },
  'connection_lost':{ title: 'Sin conexión',    body: 'Robot no responde desde hace 30s' },
};
```

---

### 10.5 MQTT como columna vertebral del control

#### Estructura de tópicos

```
robots/
  {id}/
    cmd_vel          ← velocidad lineal/angular (QoS 0, retain: false)
    cmd_sequence     ← secuencia de pasos (QoS 1, retain: false)
    telemetry        ← batería, pose, velocidad (QoS 0, retain: true)
    status           ← connected | idle | running | error (QoS 1, retain: true)
    camera/stream    ← frames JPEG en base64 (QoS 0, baja frecuencia)
    alerts           ← eventos críticos (QoS 2, retain: false)
fleet/
  all/cmd_vel        ← comando broadcast a toda la flota
  all/stop           ← parada de emergencia global
```

**QoS según tipo de mensaje:**
- QoS 0 — velocidad, joystick: sin garantía, mínima latencia
- QoS 1 — secuencias, configuración: garantía de entrega al menos una vez
- QoS 2 — parada de emergencia, alertas críticas: exactamente una vez

**Broker MQTT recomendado:** Mosquitto en Raspberry Pi o VPS propio
**Seguridad:** TLS + autenticación por usuario/password o certificado por robot

---

## Módulo 11 — DevOps para Robótica
*De "funciona en mi PC" a flota en producción con actualizaciones automáticas*

---

### 11.1 Control de Versiones para el Stack Completo

**Estructura de monorepo del curso:**
```
robot-project/
  firmware/          # Rust — ESP32
  algorithms/        # Python — cinemática, IA, navegación
  mobile/
    ios/             # Swift — app iOS
    app/             # React Native — app Android + iOS
  web/               # FastAPI + React — control web
  models/            # archivos 3D: STL, DXF, STEP (→ git LFS)
  ml_models/         # pesos de redes entrenadas (→ DVC)
  simulation/        # mundos Gazebo, URDFs
  infra/             # Docker Compose, configs, scripts CI
```

**Herramientas de versionado:**
- `git` + `git LFS`: código + modelos 3D pesados
- `DVC` (Data Version Control): datasets de IA y pesos de modelos ML
- Conventional commits: `feat(firmware): add LoRa telemetry` → changelog automático
- Semantic versioning: firmware `v1.3.2` sincronizado con app `v1.3.x`

---

### 11.2 CI/CD para Firmware Rust (ESP32)

```yaml
# .github/workflows/firmware.yml
name: Firmware CI
on: [push, pull_request]
jobs:
  build-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: esp-rs/xtensa-toolchain@v1
      - name: Build firmware
        run: cargo build --release --target xtensa-esp32-espidf
      - name: Lint
        run: cargo clippy -- -D warnings
      - name: Unit tests (no hardware)
        run: cargo test --lib
      - name: Upload artifact
        uses: actions/upload-artifact@v4
        with:
          name: firmware-${{ github.sha }}.bin
          path: target/xtensa-esp32-espidf/release/robot_firmware.bin
```

**Hardware-in-the-loop (HIL) en CI:**
- Runner self-hosted con ESP32 conectado por USB
- `probe-rs` flashea el firmware y ejecuta tests de integración reales
- Tests verifican: arranque correcto, comunicación MQTT, respuesta de sensores

---

### 11.3 CI/CD para Python, Web y Mobile

**Python (algoritmos + FastAPI):**
```yaml
- name: Lint + format
  run: ruff check . && black --check .
- name: Tests + coverage
  run: pytest --cov=src --cov-report=xml
- name: Build Docker image
  run: docker build -t robot-api:${{ github.sha }} .
- name: Push to registry
  run: docker push ghcr.io/org/robot-api:${{ github.sha }}
```

**React Native:**
```yaml
- name: Install deps
  run: cd mobile/app && npm ci
- name: Type check
  run: npx tsc --noEmit
- name: Unit tests
  run: npx jest --coverage
- name: Build APK (Android)
  run: eas build --platform android --profile preview
```

**Swift (iOS):**
```yaml
- name: Build & test
  run: xcodebuild test -scheme RobotApp -destination 'platform=iOS Simulator'
- name: Deploy to TestFlight
  run: fastlane beta
  if: github.ref == 'refs/heads/main'
```

---

### 11.4 OTA Updates (Over-The-Air) para ESP32

**Particiones A/B en ESP32:**
```
Flash ESP32:
┌──────────────┬──────────────┬────────────┬──────────┐
│  Bootloader  │ Partition A  │ Partition B │  SPIFFS  │
│   (16 KB)   │  (firmware   │  (firmware  │  (datos) │
│              │  activo)     │  standby)   │          │
└──────────────┴──────────────┴────────────┴──────────┘
```

**Flujo de actualización:**
```
Servidor OTA (FastAPI)
    │ robot consulta /ota/check con versión actual
    │ servidor responde: { "update": true, "version": "1.4.0", "url": "..." }
    ▼
ESP32 descarga firmware nuevo a Partition B
    │ verifica hash SHA256 + firma ECDSA
    ▼
Reinicia → Bootloader activa Partition B
    │ si arranca OK en 30s → confirma nueva partición
    │ si falla → rollback automático a Partition A
```

**Rollout gradual en Python:**
```python
class OTAManager:
    def release(self, version: str, rollout_pct: int = 10):
        robots = db.get_all_robots()
        target = random.sample(robots, k=int(len(robots) * rollout_pct / 100))
        for robot in target:
            mqtt.publish(f"robots/{robot.id}/ota", {"version": version})

    def promote(self, version: str):
        # Si el 10% inicial está OK → promover al 100%
        if self.success_rate(version) > 0.95:
            self.release(version, rollout_pct=100)
```

---

### 11.5 Monitoreo y Observabilidad

**Stack de observabilidad:**
```
ESP32/RPi → MQTT → Telegraf → InfluxDB → Grafana
                             ↘ Loki (logs) → Grafana
```

**Métricas por robot en Grafana:**

| Panel | Métrica | Alerta |
|-------|---------|--------|
| Batería | `battery_mv` | < 3500 mV → warning |
| CPU Raspberry | `cpu_usage_pct` | > 85% → warning |
| Temperatura ESP32 | `chip_temp_c` | > 70°C → critical |
| Latencia MQTT | `mqtt_rtt_ms` | > 500 ms → warning |
| Misiones/hora | `missions_completed` | < KPI → info |
| Errores firmware | `fault_count` | > 0 → critical |

**Docker Compose del stack de monitoreo:**
```yaml
services:
  mosquitto:
    image: eclipse-mosquitto:2
    ports: ["1883:1883", "8883:8883"]  # plain + TLS

  telegraf:
    image: telegraf:1.30
    volumes: ["./telegraf.conf:/etc/telegraf/telegraf.conf"]

  influxdb:
    image: influxdb:2.7
    volumes: ["influx-data:/var/lib/influxdb2"]

  grafana:
    image: grafana/grafana:10
    ports: ["3000:3000"]
    volumes: ["./dashboards:/etc/grafana/provisioning/dashboards"]

  loki:
    image: grafana/loki:2.9
```

---

### 11.6 Containerización del Stack Robótico

**Multi-arch Docker para Raspberry Pi + PC:**
```dockerfile
# Base ROS2 + dependencias Python del robot
FROM ros:jazzy-ros-base AS base
RUN apt-get install -y python3-pip && pip install fastapi uvicorn paho-mqtt

FROM base AS robot-api
COPY algorithms/ /app/
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0"]

FROM base AS robot-nav
COPY navigation/ /app/
CMD ["ros2", "launch", "nav2_bringup", "navigation_launch.py"]
```

```bash
# Build para arm64 (Raspberry Pi) desde tu Mac/PC
docker buildx build --platform linux/arm64 -t robot-api:latest --push .
```

**Docker Compose en Raspberry Pi del robot:**
```yaml
services:
  robot-api:    image: ghcr.io/org/robot-api:latest
  robot-nav:    image: ghcr.io/org/robot-nav:latest
  mqtt-client:  image: ghcr.io/org/mqtt-bridge:latest
  # Todo orquestado: un solo `docker compose up` levanta el robot
```

---

### 11.7 Seguridad del Sistema Robótico

**TLS + autenticación en MQTT:**
```
Cada robot tiene:
  - Certificado de cliente único (CA propia del proyecto)
  - Se autentica con mTLS al broker (nadie puede suplantar un robot)
  - Tópicos ACL: robot-01 solo puede publicar en robots/01/* y leer fleet/*
```

**Secure Boot en ESP32:**
- Clave RSA-3072 quemada en eFuses en producción
- El bootloader verifica firma del firmware antes de ejecutarlo
- Un firmware no firmado = boot bloqueado

**Gestión de secretos:**
```bash
# .env.encrypted en git (cifrado con age)
MQTT_PASSWORD=...
OPENAI_API_KEY=...
OTA_SIGNING_KEY=...

# Descifrar en el runner de CI con clave en GitHub Secrets
age --decrypt -i $PRIVATE_KEY .env.encrypted > .env
```

**Rate limiting en API de control:**
```python
from slowapi import Limiter
limiter = Limiter(key_func=get_robot_id)

@app.post("/robots/{id}/command")
@limiter.limit("20/second")  # máximo 20 comandos/s por robot
async def send_command(id: str, cmd: RobotCommand):
    ...
```

---

## Módulo 13 — Web API con ESP32 y MQTT en Tiempo Real
*El ESP32 como servidor de datos y la web como panel de control vivo*

> En este módulo el ESP32 deja de ser un esclavo pasivo: expone su propia API REST,
> publica telemetría en MQTT y alimenta dashboards en tiempo real con WebSocket.
> El stack completo funciona sin servidor externo — el robot ES el servidor.

---

### 13.1 Arquitectura: ESP32 como servidor autónomo

```
[Sensores] ──► [ESP32 - Firmware Rust]
                    │
                    ├── HTTP REST   ───► /api/status · /api/cmd · /api/telemetry
                    ├── WebSocket   ───► /ws  (push bidireccional)
                    ├── MQTT client ───► broker.local (topics del robot)
                    └── mDNS        ───► robot-01.local (sin IP fija)
```

**Por qué este enfoque:**
- Cero dependencia de la nube para operación básica
- La app web se sirve desde SPIFFS del propio ESP32
- MQTT conecta múltiples robots al mismo broker
- WebSocket elimina el polling: el robot empuja datos cuando cambian

---

### 13.2 Servidor HTTP + REST en Rust con ESP-IDF

```rust
use esp_idf_svc::http::server::{Configuration, EspHttpServer};
use esp_idf_svc::http::Method;

fn start_http_server() -> anyhow::Result<EspHttpServer<'static>> {
    let mut server = EspHttpServer::new(&Configuration {
        stack_size: 8192,
        ..Default::default()
    })?;

    // GET /api/status → JSON con estado actual del robot
    server.fn_handler("/api/status", Method::Get, |req| {
        let state = ROBOT_STATE.lock().unwrap();
        let json = serde_json::json!({
            "linear_vel":  state.linear_vel,
            "angular_vel": state.angular_vel,
            "battery_pct": state.battery_pct,
            "mode":        state.mode.as_str(),
            "uptime_s":    state.uptime_s,
        });
        req.into_ok_response()?.write_all(json.to_string().as_bytes())?;
        Ok(())
    })?;

    // POST /api/cmd → recibe JSON con velocidad
    server.fn_handler("/api/cmd", Method::Post, |mut req| {
        let mut body = Vec::new();
        req.read_to_end(&mut body)?;
        let cmd: RobotCmd = serde_json::from_slice(&body)?;
        COMMAND_TX.send(cmd).ok();
        req.into_ok_response()?.write_all(b"{\"ok\":true}")?;
        Ok(())
    })?;

    Ok(server)
}
```

**Rutas de la API REST del robot:**

| Método | Ruta | Body | Respuesta |
|--------|------|------|-----------|
| GET | `/api/status` | — | JSON: vel, batería, modo, uptime |
| POST | `/api/cmd` | `{linear, angular}` | `{ok: true}` |
| GET | `/api/sensors` | — | JSON: IMU, encoders, sonar |
| POST | `/api/mode` | `{mode: "auto"/"manual"/"stop"}` | `{ok, mode}` |
| GET | `/api/config` | — | Parámetros PID y motion |
| PUT | `/api/config` | `{kp, ki, kd, max_vel}` | `{ok, applied}` |
| GET | `/` | — | HTML de la app web (desde SPIFFS) |

---

### 13.3 WebSocket bidireccional — telemetría push en tiempo real

```rust
// Servidor WebSocket en ESP-IDF (Rust)
use esp_idf_svc::ws::server::EspHttpWsDetachedSender;

static WS_SENDER: Mutex<Option<EspHttpWsDetachedSender>> = Mutex::new(None);

fn ws_handler(ws: EspHttpWsConnection) -> anyhow::Result<()> {
    let (sender, mut receiver) = ws.split();
    *WS_SENDER.lock().unwrap() = Some(sender);

    // Recibir comandos del browser
    loop {
        match receiver.recv()? {
            Frame::Text(text) => {
                let cmd: RobotCmd = serde_json::from_str(&text)?;
                COMMAND_TX.send(cmd).ok();
            }
            Frame::Close(_) => break,
            _ => {}
        }
    }
    Ok(())
}

// Task periódica que empuja telemetría cada 50 ms
fn telemetry_push_task() {
    loop {
        let state = ROBOT_STATE.lock().unwrap().clone();
        let payload = serde_json::json!({
            "t":     state.timestamp_ms,
            "lv":    state.linear_vel,
            "av":    state.angular_vel,
            "bat":   state.battery_pct,
            "imu":   { "roll": state.roll, "pitch": state.pitch, "yaw": state.yaw },
            "enc":   { "left": state.enc_left, "right": state.enc_right },
        }).to_string();

        if let Some(sender) = WS_SENDER.lock().unwrap().as_mut() {
            sender.send(FrameType::Text(false), payload.as_bytes()).ok();
        }
        FreeRtos::delay_ms(50);
    }
}
```

**Cliente JavaScript — dashboard en tiempo real:**

```javascript
// Conecta al WebSocket del robot (sin servidor externo)
const ws = new WebSocket(`ws://${robotIp}/ws`);

// Actualiza la UI con cada frame de telemetría
ws.onmessage = ({ data }) => {
  const d = JSON.parse(data);
  document.getElementById('bat').textContent  = `${d.bat.toFixed(1)} %`;
  document.getElementById('vel').textContent  = `${d.lv.toFixed(2)} m/s`;
  document.getElementById('yaw').textContent  = `${d.imu.yaw.toFixed(1)}°`;
  updateChart(d.t, d.lv);          // gráfica deslizante
};

// Envía comando de velocidad
function sendVelocity(linear, angular) {
  if (ws.readyState === WebSocket.OPEN)
    ws.send(JSON.stringify({ linear, angular }));
}
```

---

### 13.4 MQTT — publicación y suscripción desde el ESP32

```rust
use esp_idf_svc::mqtt::client::{EspMqttClient, MqttClientConfiguration, QoS};

fn start_mqtt(client_id: &str, broker: &str) -> anyhow::Result<EspMqttClient<'static>> {
    let cfg = MqttClientConfiguration {
        client_id: Some(client_id),
        username:  Some("robot"),
        password:  Some("secret"),
        keep_alive_interval: Some(Duration::from_secs(15)),
        ..Default::default()
    };

    let (mut client, mut conn) = EspMqttClient::new(broker, &cfg)?;

    // Suscribirse a comandos entrantes
    client.subscribe("robots/+/cmd", QoS::AtLeastOnce)?;

    // Thread que procesa mensajes entrantes
    std::thread::spawn(move || {
        for msg in conn.iter() {
            if let Ok(event) = msg {
                handle_mqtt_event(event);
            }
        }
    });

    Ok(client)
}

fn publish_telemetry(client: &mut EspMqttClient, robot_id: &str, state: &RobotState) {
    let payload = serde_json::json!({
        "linear_vel":  state.linear_vel,
        "battery_pct": state.battery_pct,
        "yaw_deg":     state.yaw,
        "ts":          state.timestamp_ms,
    }).to_string();

    client.publish(
        &format!("robots/{robot_id}/telemetry"),
        QoS::AtMostOnce,   // QoS 0: máxima frecuencia, mínima latencia
        false,
        payload.as_bytes(),
    ).ok();
}
```

**Jerarquía de tópicos MQTT del curso:**

```
robots/
  {robot_id}/
    cmd_vel      ← Python/app envía  {linear, angular}          QoS 0
    telemetry    ← ESP32 publica     {vel, bat, imu, ts}        QoS 0
    status       ← ESP32 publica     {mode, wifi_rssi, errors}  QoS 1
    config/set   ← Python envía      {kp, ki, kd, max_vel}      QoS 1
    config/ack   ← ESP32 responde    {ok, applied_config}       QoS 1
    alerts       ← ESP32 publica     {type, value, threshold}   QoS 2
    ota/start    ← CI envía          {url, sha256, version}     QoS 2
    ota/progress ← ESP32 reporta     {pct, state}               QoS 1

fleet/
  broadcast/cmd  ← comando a toda la flota                      QoS 1
  status         ← agregado de todos los robots                 QoS 0
```

---

### 13.5 Dashboard web servido desde SPIFFS del ESP32

La app web completa vive en el propio ESP32, sin CDN ni servidor externo.

**Estructura en SPIFFS:**
```
/spiffs/
  index.html       ← layout del dashboard
  app.js           ← lógica WebSocket + charts
  style.css        ← UI responsive
  config.json      ← parámetros editables en campo
```

**Subir archivos a SPIFFS desde el PC:**
```bash
# Con esptool + mkspiffs
mkspiffs -c web/ -s 0x100000 spiffs.bin
esptool.py --port /dev/ttyUSB0 write_flash 0x300000 spiffs.bin

# Con cargo-espflash (integrado en el build de Rust)
# En partitions.csv:
# spiffs, data, spiffs, 0x300000, 1M
```

**Panel de control HTML minimalista:**
```html
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Robot Control</title>
  <style>
    body { font-family: monospace; background: #111; color: #0f0; }
    .gauge { font-size: 2rem; margin: 1rem; }
    .joystick-zone { width: 200px; height: 200px; border: 2px solid #0f0; margin: auto; }
    canvas { display: block; margin: auto; }
  </style>
</head>
<body>
  <div class="gauge">🔋 <span id="bat">--</span>%</div>
  <div class="gauge">⚡ <span id="vel">--</span> m/s</div>
  <div class="gauge">🧭 <span id="yaw">--</span>°</div>
  <canvas id="chart" width="400" height="150"></canvas>
  <div class="joystick-zone" id="joy"></div>
  <script src="app.js"></script>
</body>
</html>
```

---

### 13.6 Suscriptor Python — consumir MQTT del robot

```python
import paho.mqtt.client as mqtt
import json, time

class RobotSubscriber:
    def __init__(self, broker: str, robot_id: str):
        self.robot_id = robot_id
        self.state = {}
        self.client = mqtt.Client(f"python-monitor-{robot_id}")
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.connect(broker, 1883)
        self.client.loop_start()

    def _on_connect(self, client, *_):
        client.subscribe(f"robots/{self.robot_id}/telemetry", qos=0)
        client.subscribe(f"robots/{self.robot_id}/status",    qos=1)
        client.subscribe(f"robots/{self.robot_id}/alerts",    qos=2)

    def _on_message(self, client, userdata, msg):
        payload = json.loads(msg.payload)
        topic   = msg.topic.split("/")[-1]
        self.state[topic] = payload

        if topic == "alerts":
            print(f"⚠️ ALERTA {self.robot_id}: {payload['type']} = {payload['value']}")

    def send_velocity(self, linear: float, angular: float):
        cmd = json.dumps({"linear": linear, "angular": angular})
        self.client.publish(f"robots/{self.robot_id}/cmd_vel", cmd, qos=0)

# Uso
robot = RobotSubscriber("broker.local", "robot-01")
time.sleep(1)
robot.send_velocity(0.3, 0.0)   # adelante 0.3 m/s
print(robot.state.get("telemetry", {}))
```

---

### 13.7 Broker MQTT local — Mosquitto en Docker

```yaml
# docker-compose.yml — broker + dashboard
services:
  mosquitto:
    image: eclipse-mosquitto:2
    ports: ["1883:1883", "9001:9001"]   # TCP + WebSocket
    volumes:
      - ./mosquitto/config:/mosquitto/config
      - ./mosquitto/data:/mosquitto/data
    restart: unless-stopped

  # Monitor MQTT vía web
  mqtt-explorer:
    image: smeagolworms4/mqtt-explorer
    ports: ["4000:4000"]
    restart: unless-stopped
```

```ini
# mosquitto/config/mosquitto.conf
listener 1883
listener 9001
protocol websockets
allow_anonymous false
password_file /mosquitto/config/passwords.txt
persistence true
log_dest file /mosquitto/data/mosquitto.log
```

```bash
# Crear usuario para el robot
docker exec mosquitto mosquitto_passwd -c /mosquitto/config/passwords.txt robot
docker exec mosquitto mosquitto_passwd /mosquitto/config/passwords.txt python-client
```

---

### 13.8 mDNS — El robot se anuncia sin IP fija

```rust
use esp_idf_svc::mdns::EspMdns;

fn start_mdns(hostname: &str, robot_id: &str) -> anyhow::Result<EspMdns> {
    let mut mdns = EspMdns::take()?;
    mdns.set_hostname(hostname)?;    // ej: "robot-01"
    mdns.set_instance_name(robot_id)?;
    mdns.add_service(
        Some(robot_id),
        "_robot",        // tipo de servicio custom
        "_tcp",
        8080,
        &[("version", "1.0"), ("api", "rest+ws")],
    )?;
    Ok(mdns)
}
```

```python
# Descubrir todos los robots en la red local
from zeroconf import ServiceBrowser, Zeroconf

class RobotListener:
    def add_service(self, zc, type_, name):
        info = zc.get_service_info(type_, name)
        ip = info.parsed_addresses()[0]
        print(f"Robot encontrado: {name} → http://{ip}:{info.port}")

zc = Zeroconf()
browser = ServiceBrowser(zc, "_robot._tcp.local.", RobotListener())
```

---

### 13.9 Seguridad de la API del robot

| Amenaza | Mitigación |
|---------|-----------|
| Acceso no autorizado a `/api/cmd` | Token Bearer en header `X-Robot-Token` |
| Replay attacks | Timestamp en payload, rechazo si `|now - ts| > 5 s` |
| MQTT sin auth | Usuario + contraseña por robot, TLS con `mqtts://` |
| Flood de comandos | Rate limiting en Rust: máx. 20 cmd/s por conexión |
| Sniffing WiFi | WPA2-AES en la red, HTTPS con cert auto-firmado |

```rust
// Rate limiting simple en Rust
struct RateLimiter { last: Instant, count: u32, max: u32 }
impl RateLimiter {
    fn allow(&mut self) -> bool {
        if self.last.elapsed() > Duration::from_secs(1) {
            self.last = Instant::now();
            self.count = 0;
        }
        self.count += 1;
        self.count <= self.max
    }
}
```

---

### 13.10 Proyecto integrador: Robot con API completa en 48 horas

**Objetivo:** Construir el backend completo de un robot diferencial con:
- REST API para comandos y configuración
- WebSocket para telemetría a 20 Hz
- MQTT con broker local
- Dashboard HTML servido desde SPIFFS
- mDNS para descubrimiento en LAN

**Checklist de entrega:**

```
Firmware (Rust / ESP32)
  ├── [ ] HTTP server con 6 rutas REST
  ├── [ ] WebSocket handler bidireccional
  ├── [ ] MQTT client: pub telemetry / sub cmd_vel
  ├── [ ] SPIFFS: servir index.html + app.js
  ├── [ ] mDNS: robot-{id}.local
  └── [ ] Rate limiter + token auth

Dashboard (HTML / JS en SPIFFS)
  ├── [ ] Conecta WebSocket al ESP32
  ├── [ ] Muestra batería, velocidad, yaw en tiempo real
  ├── [ ] Joystick táctil envía cmd_vel por WebSocket
  └── [ ] Gráfica deslizante de velocidad (Chart.js ligero)

Python (PC / Raspberry)
  ├── [ ] RobotSubscriber con paho-mqtt
  ├── [ ] Descubrimiento mDNS con zeroconf
  ├── [ ] Logger: guarda telemetría en CSV/InfluxDB
  └── [ ] Envía secuencias de movimiento preprogramadas

Infraestructura
  ├── [ ] Mosquitto en Docker con autenticación
  ├── [ ] MQTT Explorer corriendo en http://localhost:4000
  └── [ ] Grafana con datasource InfluxDB mostrando batería y velocidad
```

---

## Proyectos integradores

### Proyecto A — Robot Móvil Autónomo
**Stack:** ESP32 (Rust) + Python + ROS2 + IA

- Hardware: robot diferencial con LiDAR y cámara
- Firmware Rust: control PID de motores, lectura de encoders e IMU
- Python: EKF, A*, SLAM 2D, Nav2
- IA: detección de personas con YOLOv8, evitación inteligente
- Dashboard: telemetría en tiempo real vía MQTT → Grafana

### Proyecto B — Brazo Manipulador Inteligente
**Stack:** ESP32 (Rust) + Python + PyTorch + OpenCV

- Hardware: brazo 4-DOF con servos y gripper
- Firmware Rust: trayectorias en tiempo real, control par calculado
- Python: cinemática DH, planificación de movimiento
- IA: detección de objetos YOLOv8, estimación de pose 6D, pick & place autónomo
- Interfaz: control por voz con LLM → planificador → ejecución

### Proyecto C — Enjambre de Robots (Swarm)
**Stack:** múltiples ESP32 (Rust) + Python + RL

- Múltiples robots moviéndose de forma coordinada vía WiFi mesh
- Algoritmo de formación distribuido en Rust en cada nodo
- Entrenamiento RL multi-agente con `pettingzoo`
- Coordinación centralizada opcional vía servidor Python

### Proyecto D — Robot Fabricado desde Cero (Diseño → Física)
**Stack:** CadQuery + Blender + Impresión 3D + Corte Láser + ESP32 (Rust) + Python

- **Diseño paramétrico**: dimensiones derivadas del modelo cinemático Python → CadQuery genera STL y DXF
- **Validación digital**: URDF desde Blender + Phobos → Gazebo confirma cinemática antes de fabricar
- **Fabricación híbrida**:
  - Chasis en MDF 4mm cortado a láser (finger joints, sin tornillos)
  - Eslabones y brackets en PETG (calidad estructural)
  - Ruedas y gripper en TPU (grip y flexibilidad)
  - Paneles de cubierta en acrílico 3mm cortado a láser
- **Electrónica integrada en el diseño**: ESP32, motores, encoders, LiPo, gestión de cables
- **Resultado entregable**: robot físico funcional + gemelo digital en Gazebo + firmware Rust + código Python de control + BOM completo

### Proyecto E — Robot de Competición Completo
**Stack:** ESP32 (Rust) + Python + CadQuery + impresión 3D

- Zumo robot: chasis PETG bajo perfil, firmware Rust a 500 Hz, estrategia de combate por FSM
- Seguidor de línea: PID adaptativo, ganancia variable por zona, < 5 s en circuito estándar
- FPV racer: frame impreso + firmware Rust para FC, tune de PID desde logs analizados en Python
- Proceso completo: diseño CAD → impresión → ensamble → tune → competición

### Proyecto F — Robot Relacional con LLM
**Stack:** Python + LLM (Claude API) + ESP32 (Rust) + ROS2 + OpenCV + TTS/STT

- Robot móvil con pantalla, altavoz y micrófono
- Pipeline: voz → Whisper → LLM → skill → acción → respuesta TTS
- Memoria episódica con ChromaDB: recuerda objetos, preferencias, historial de tareas
- VLM: "¿qué ves?" → el robot describe la escena en lenguaje natural
- Integración: navega de forma autónoma y conversa mientras trabaja

### Proyecto G — Sistema Logístico Multi-Robot
**Stack:** Python (FastAPI + ROS2) + múltiples ESP32 (Rust) + MQTT + Grafana

- 3 AMRs con SLAM, coordinados por servidor Python central
- API REST para recibir órdenes de un WMS simulado
- Dashboard Grafana: posición, batería, tarea actual de cada robot en tiempo real
- Algoritmo de asignación Hungarian para minimizar distancia total recorrida

---

## Herramientas y librerías clave

### Python
```
robotics-toolbox-python   # cinemática y dinámica
numpy / scipy / sympy     # matemáticas
opencv-python             # visión
ultralytics               # YOLOv8
torch / onnxruntime       # IA / inferencia
stable-baselines3         # Reinforcement Learning
filterpy                  # Kalman, partículas
open3d                    # nube de puntos
pyserial / asyncio        # comunicación
rclpy                     # ROS2 en Python
do-mpc                    # MPC
```

### Rust
```
esp-idf-hal               # HAL oficial ESP32
embassy                   # async embebido
nalgebra                  # álgebra lineal
serde / postcard          # serialización binaria
defmt + probe-rs          # logging y debugging
esp32-nimble              # BLE en Rust
pyo3                      # FFI Rust → Python
rclrs                     # ROS2 en Rust
```

### Diseño 3D y Fabricación
```
cadquery / build123d      # CAD paramétrico en Python
solidpython2              # OpenSCAD desde Python
blender + phobos          # modelado 3D + exportar URDF
freecad (Python API)      # CAD mecánico + FEM básico
inkscape                  # vectores SVG/DXF para corte láser
prusaslicer               # slicing para impresión FDM
lightburn / laserweb      # control de cortadora láser
```

### Aplicaciones y Dominio
```
mavsdk-python             # control de drones PX4/ArduPilot
dronekit                  # alternativa MAVLink en Python
opendronemap              # fotogrametría y mapeo aéreo
chromadb                  # memoria vectorial para robots LLM
sentence-transformers     # embeddings para RAG robótico
whisper                   # STT local (instrucciones por voz)
coqui-tts / pyttsx3       # TTS local para respuesta del robot
pyqt6                     # GUI de programación de robots
asyncua                   # OPC-UA para integración industrial
timescaledb               # series temporales de telemetría
betaflight                # firmware de referencia para FPV
```

### Control Remoto y Mobile
```
# Rust / ESP32
embassy-net               # servidor web + WebSocket embebido
esp-idf-svc (httpd)       # HTTP server en SPIFFS
sx127x / lora-rs          # driver LoRa en Rust

# iOS (Swift)
CocoaMQTT                 # cliente MQTT nativo Swift
CoreBluetooth             # BLE nativo iOS

# React Native
mqtt.js                   # cliente MQTT sobre WebSocket
react-native-ble-manager  # BLE para Android + iOS
react-native-joystick     # joystick táctil
zustand                   # estado global reactivo
expo-notifications        # push notifications
eas (Expo Application Services) # build + deploy iOS/Android
fastlane                  # automatización deploy iOS TestFlight
```

### DevOps
```
probe-rs                  # flash + debug ESP32 en CI
cargo-embed               # runner de tests embebidos
dvc                       # versión de datasets y modelos ML
age / sops                # cifrado de secretos en git
telegraf                  # ingesta de métricas MQTT → InfluxDB
influxdb                  # series temporales de telemetría
loki                      # logs centralizados de flota
grafana                   # dashboards + alertas
mosquitto / emqx          # broker MQTT de producción
docker buildx             # builds multi-arch (arm64 + amd64)
```

### Electrónica y Diseño de PCB
```
kicad 8                   # diseño de esquemáticos y PCB (estándar del curso)
kicad-skip (Python API)   # generación de símbolos y footprints por script
lcsc / jlcpcb             # componentes SMD + fabricación de PCBs
freecad (A2plus)          # verificar encaje mecánico PCB en chasis
ltspice / kicad-sim       # simulación de circuitos antes de fabricar
openocd + probe-rs        # flash y debug vía JTAG/SWD en PCB custom
sigrok / pulseview        # análisis lógico open-source
```

### Infraestructura
```
ROS2 Jazzy               # middleware robótico
Gazebo Harmonic           # simulación
Docker + buildx           # builds multi-arch arm64 + amd64
GitHub Actions + git LFS  # CI/CD firmware + modelos 3D + Gerbers
Grafana + MQTT            # monitoreo en campo
```

---

## Kits del curso — BOM por proyecto

> Cada kit llega con piezas 3D impresas y/o cortadas a láser listas para ensamblar.
> El alumno solo necesita destornillador, soldador de punta fina y PC.

### Kit 0 — Fundamentos ESP32 (Módulos 0 y 1)
*Introducción a Rust + ESP32, sin robot completo aún*

| Componente | Cant. | Notas |
|-----------|-------|-------|
| ESP32 DevKit V1 (38 pines) | 1 | Placa principal de todo el curso |
| Breadboard 830 puntos | 1 | |
| LED RGB + resistencias 220Ω | 5 | |
| Pulsadores + potenciómetro 10kΩ | 3 | |
| HC-SR04 ultrasonido | 1 | Primer sensor |
| MPU-6050 IMU (I2C) | 1 | Orientación y aceleración |
| Jumpers M-M / M-H / H-H | 60 | |
| Cable USB-A a micro-USB | 1 | |
| Carcasa ESP32 impresa PETG | 1 | 80×55×25 mm — 3D impresa |

---

### Kit 1 — Robot Diferencial Base (Proyectos A y E-línea)
*Seguidor de línea · Navegación autónoma*

| Componente | Cant. | Fabricación | Dimensión |
|-----------|-------|------------|-----------|
| ESP32 DevKit V1 | 1 | — | — |
| Motor DC 6V con encoder cuadrático | 2 | — | — |
| Driver DRV8833 | 1 | — | — |
| Ruedas TPU Ø70mm | 2 | 3D impresa TPU | Ø70 × 20 mm |
| Rueda loca delantera | 1 | 3D impresa PETG | Ø25 mm |
| Chasis diferencial | 1 | Láser MDF 4mm | 220 × 170 mm |
| Bandeja electrónica | 1 | Láser MDF 4mm | 200 × 150 mm |
| Tapa acrílico | 1 | Láser acrílico 3mm | 220 × 170 mm |
| Array sensor QTR-8RC | 1 | — | Línea |
| Batería LiPo 7.4V 1200mAh + BMS | 1 | — | — |
| Soporte batería + cables XT30 | 1 | 3D impresa PETG | 80×45×25 mm |
| Tornillos M3 × 10 mm + tuercas | 20 | — | — |
| Heat inserts M3 | 10 | — | Incluidos en piezas |

---

### Kit 2 — Robot Zumo (Proyecto E-zumo)
*Competición sumo · Detección de oponente · Estrategia de combate*

| Componente | Cant. | Fabricación | Dimensión |
|-----------|-------|------------|-----------|
| ESP32 DevKit V1 | 1 | — | — |
| Motor DC 6V 300 RPM con encoder | 2 | — | Alta tracción |
| Driver DRV8883 | 1 | — | — |
| Chasis bajo perfil (cuña frontal) | 1 | 3D impresa PETG | 180×145×35 mm |
| Ruedas goma tracción Ø50mm | 2 | 3D impresa TPU | Ø50 × 22 mm |
| Sensor borde IR TCRT5000 × 4 | 4 | — | Esquinas |
| Sensor ToF VL53L0X (oponente) | 2 | — | Frontal + lateral |
| LiPo 7.4V 850mAh compacta | 1 | — | Perfil bajo |
| Soporte sensores frontales | 1 | 3D impresa PETG | 60×20×15 mm |

---

### Kit 3 — Brazo Manipulador 4-DOF (Proyecto B)
*Pick & place · Cinemática inversa · Control visual*

| Componente | Cant. | Fabricación | Dimensión |
|-----------|-------|------------|-----------|
| ESP32 DevKit V1 | 1 | — | — |
| Servo MG996R (articulaciones) | 4 | — | 180° |
| Servo SG90 (gripper) | 1 | — | — |
| PCA9685 (controlador 16 servos I2C) | 1 | — | — |
| Base del brazo | 1 | 3D impresa PETG | 120×120×40 mm |
| Eslabón 1 (hombro) | 1 | 3D impresa PETG | 140×50×30 mm |
| Eslabón 2 (codo) | 1 | 3D impresa PETG | 120×40×25 mm |
| Eslabón 3 (muñeca) | 1 | 3D impresa PETG | 90×35×20 mm |
| Gripper paralelo (2 dedos TPU) | 1 | 3D impresa PETG+TPU | 80×50×30 mm |
| Plataforma base cortada | 1 | Láser MDF 4mm | 300×300 mm |
| Rodamientos 608ZZ | 4 | — | Articulaciones |
| Heat inserts M3 + tornillos | 30 | — | — |
| Fuente 5V 3A (servos) | 1 | — | — |

---

### Kit 4 — Dron Quadrotor (Proyecto E-FPV / 9.2)
*FPV racing · Dron de trabajo · Control PX4*

| Componente | Cant. | Fabricación | Dimensión |
|-----------|-------|------------|-----------|
| ESP32-S3 DevKit (FC custom) | 1 | — | — |
| MPU-6000 IMU (SPI) | 1 | — | Soldado en FC |
| Motor BLDC 2206 2300KV | 4 | — | — |
| ESC 30A BLHeli32 | 4 | — | — |
| Hélice 5" bipaleta CW/CCW | 4+4 | — | Repuesto incluido |
| Brazo del frame (×4 idénticos) | 4 | 3D impresa PETG | 160×30×15 mm |
| Placa central top | 1 | Láser acrílico 3mm | 120×120 mm |
| Placa central bottom | 1 | Láser acrílico 3mm | 120×120 mm |
| LiPo 4S 1500mAh 75C | 1 | — | — |
| Soporte cámara FPV | 1 | 3D impresa PETG | 40×30×25 mm |
| Receptor ELRS 2.4GHz | 1 | — | — |
| Conector XT60 + cables silicona 14AWG | — | — | — |
| Standoffs M3 × 6 mm aluminio | 16 | — | — |

---

### Kit 5 — Robot Relacional LLM (Proyecto F)
*Voz → LLM → acción · Memoria episódica · Interacción multimodal*

| Componente | Cant. | Fabricación | Dimensión |
|-----------|-------|------------|-----------|
| ESP32-S3 DevKit | 1 | — | Con cámara OV2640 |
| Raspberry Pi 4 (4GB) | 1 | — | Ejecuta LLM local / API |
| Micrófono INMP441 (I2S) | 1 | — | — |
| Altavoz 3W + amplificador PAM8403 | 1 | — | — |
| Pantalla OLED 128×64 (I2C) | 1 | — | Expresiones del robot |
| Ring NeoPixel 16 LED WS2812B | 1 | — | Estado emocional |
| Base diferencial del Kit 1 | 1 | — | Reutiliza plataforma |
| Carcasa "cabeza" del robot | 1 | 3D impresa PETG | 130×100×120 mm |
| Soporte pantalla + micrófono | 1 | 3D impresa PETG | 90×60×20 mm |
| Batería powerbank 20000mAh USB-C | 1 | — | Alimenta Raspberry Pi |

---

### Kit 6 — AGV Logístico (Proyecto G)
*SLAM · Navegación autónoma · Coordinación multi-robot*

| Componente | Cant. | Fabricación | Dimensión |
|-----------|-------|------------|-----------|
| ESP32 DevKit V1 | 1 | — | micro-ROS |
| RPLidar A1 (LiDAR 2D 360°) | 1 | — | SLAM |
| Motor DC 12V con encoder | 2 | — | Mayor torque |
| Driver BTS7960 43A | 1 | — | Para motores 12V |
| Chasis AGV reforzado | 1 | Láser MDF 4mm | 380×280 mm |
| Bandeja electrónica | 1 | Láser MDF 4mm | 350×240 mm |
| Tapa superior con soporte LiDAR | 1 | Láser acrílico 3mm | 380×280 mm |
| Soporte LiDAR central | 1 | 3D impresa PETG | 80×80×60 mm |
| Ruedas goma Ø100mm | 2 | 3D impresa TPU | Ø100 × 30 mm |
| Rueda loca × 2 | 2 | 3D impresa PETG | Ø40 mm |
| Batería LiPo 12V 5000mAh | 1 | — | Autonomía 3–4 h |
| Raspberry Pi 4 (SLAM + Nav2) | 1 | — | — |
| Soporte Raspberry en chasis | 1 | 3D impresa PETG | 90×60×20 mm |

---

### Kit 7 — Control: Radio + Web + Mobile (Módulo 10)
*Se combina con cualquier kit de robot; amplía su interfaz de control*

| Componente | Cant. | Notas |
|-----------|-------|-------|
| Receptor ExpressLRS (ELRS) 2.4GHz | 1 | Protocolo CRSF → ESP32 vía UART |
| Módulo LoRa SX1278 433MHz | 2 | Uno en robot, uno en "base station" PC |
| Antena 433MHz | 2 | Dipolo simple |
| Módulo NRF24L01+ con antena PA/LNA | 1 | Control alternativo de corto alcance |
| ESP32 DevKit V1 (base station) | 1 | Actúa como gateway RC → WiFi |
| Carcasa base station impresa PETG | 1 | 100×70×35 mm — 3D impresa |
| Cable USB para base station | 1 | |

> **Software incluido (descarga en campus):**
> - Firmware Rust ESP32: parsers SBUS/CRSF/LoRa listos para integrar
> - Backend FastAPI: servidor web de secuencias paso a paso
> - Frontend HTML/JS: control web servido desde ESP32 (SPIFFS)
> - App React Native: proyecto base con joystick + MQTT + BLE
> - Proyecto Swift: app iOS base con joystick + CoreBluetooth + CocoaMQTT

---

### Resumen de kits y proyectos

| Kit | Proyecto(s) | Precio referencia | 3D impreso | Láser |
|-----|------------|------------------|------------|-------|
| Kit 0 | M0 · M1 (fundamentos) | base | 1 pieza | — |
| Kit 1 | A · E-línea | — | 4 piezas | 3 piezas MDF/acrílico |
| Kit 2 | E-zumo | — | 3 piezas PETG | — |
| Kit 3 | B (manipulador) | — | 6 piezas PETG+TPU | 1 plataforma MDF |
| Kit 4 | E-FPV · 9.2 (dron) | — | 5 piezas PETG | 2 placas acrílico |
| Kit 5 | F (relacional LLM) | — | 2 piezas PETG | — |
| Kit 6 | G (AGV logístico) | — | 4 piezas PETG+TPU | 3 piezas MDF/acrílico |
| Kit 7 | M10 (control remoto) | — | 1 carcasa PETG | — |

---

## Mapa de aprendizaje

```
M0  Fundamentos (Python · Rust · ESP32)
    │
    ├── M1  Hardware y Actuadores (motores · encoders · sensores · ESP32)
    │       │
    │       ├── M12 Electrónica: Componentes · Circuitos · PCB (KiCad)
    │       │       ├── 12.1  Componentes pasivos/activos · alimentación
    │       │       ├── 12.5  Esquemáticos con KiCad
    │       │       ├── 12.6  Layout y enrutado de PCB
    │       │       ├── 12.7  Soldadura SMD
    │       │       └── 12.8  PCBs propias del curso (Shield · Driver · FC)
    │       │
    │       └── M3  Dinámica y Control en tiempo real (Rust · PID · MPC)
    │               ├── M9.1  Competición (Zumo · Línea · FPV)
    │               └── M10.1 Radio Control (ELRS · LoRa · NRF24)
    │
    ├── M2  Cinemática (Python · robotics-toolbox · cuaterniones duales)
    │       ├── M3  Dinámica
    │       ├── M4  Navegación + SLAM (ROS2 · Nav2 · EKF)
    │       │       ├── M9.3  Logística (AGV · AMR · Picking)
    │       │       └── M9.7  Agricultura / Invernadero
    │       └── M8  Diseño 3D (CadQuery · Blender · Láser 400×400 · 3D 200³)
    │
    ├── M5  Visión Artificial (OpenCV · PyTorch · YOLOv8 · Pose 6D)
    │       ├── M9.2  Drones de Trabajo (PX4 · MAVSDK · Fotogrametría)
    │       └── M9.4  Automatización Repetitiva (Cobot · Lead-Through · GUI)
    │
    ├── M6  IA (RL · LLM · VLM · Edge ONNX)
    │       ├── M9.5  Robots Relacionales (LLM + Skills + Voz + Memoria)
    │       └── M9.6  Robótica de Servicio / Asistencial
    │
    ├── M7  Integración y Producción (ROS2 · micro-ROS · Gazebo · HIL)
    │       └── M9.8  Orquestación Multi-Robot (Fleet · API REST · MQTT)
    │
    ├── M10 Control Remoto y Mobile
    │       ├── M10.2 Web Secuencial (FastAPI · WebSocket · SPIFFS · Secuencias)
    │       ├── M10.3 App iOS (Swift · SwiftUI · CoreBluetooth · CocoaMQTT)
    │       ├── M10.4 App Android+iOS (React Native · Expo · MQTT · BLE)
    │       └── M10.5 MQTT: tópicos · QoS · broker · ACL
    │
    └── M11 DevOps para Robótica
            ├── CI/CD Firmware (GitHub Actions · probe-rs · HIL)
            ├── CI/CD Python + Mobile (Docker · Fastlane · EAS · DVC)
            ├── OTA Updates (particiones A/B · firma · rollout gradual)
            ├── Monitoreo (Telegraf · InfluxDB · Grafana · Loki · Alertas)
            ├── Containerización (Docker multi-arch arm64 · Portainer)
            └── Seguridad (mTLS · Secure Boot · Secrets · Rate Limiting)

Proyectos integradores (cada uno usa kits físicos):
    ├── A — Robot Móvil Autónomo       (Kit 1 + app móvil + Grafana)
    ├── B — Brazo Manipulador IA       (Kit 3 + PCB propia + LLM)
    ├── C — Enjambre Multi-Robot       (3× Kit 1 + orquestación)
    ├── D — Robot Diseñado desde Cero  (CAD → PCB → láser → impresión)
    ├── E — Robot de Competición       (Kit 1/2/4 + radio RC)
    ├── F — Robot Relacional LLM       (Kit 5 + Swift + Whisper + TTS)
    └── G — Sistema Logístico          (Kit 6 + DevOps completo + flota)
```

---

## Perfil de salida

Al completar este curso el estudiante puede:

- Diseñar e implementar firmware robusto en **Rust** para ESP32
- Modelar cinemática y dinámica de robots en **Python** sin MATLAB
- Integrar **IA** (visión, RL, LLMs) en sistemas robóticos reales
- Construir pipelines completos desde sensor hasta decisión autónoma
- **Diseñar y fabricar piezas** con CadQuery, Blender, impresora 3D y cortadora láser
- Construir y tunear robots de **competición** (Zumo, seguidor de línea, FPV drone)
- Programar **drones de trabajo** autónomos con PX4 y MAVSDK-Python
- Implementar **robots relacionales** que conversan con LLM y ejecutan tareas físicas
- Desplegar sistemas de **automatización logística** multi-robot con orquestación central
- Controlar robots por **radio RC** (ELRS/LoRa), **web secuencial** y **app móvil** (Swift + React Native)
- Diseñar y fabricar **PCBs propias** con KiCad: esquemático → layout → Gerbers → JLCPCB → soldadura SMD
- Mantener una **flota de robots en producción** con CI/CD, OTA, monitoreo Grafana y seguridad TLS
- Cerrar el ciclo completo: modelo matemático → PCB propia → CAD → simulación → fabricación → robot físico en producción

---

*Currículo generado a partir del análisis de: "Robótica de la Cinemática al Control" (Udemy 4.7★),
serie Roboticoss Python (Udemy), Robotics Toolbox Python (Peter Corke),
y mejores prácticas de la industria robótica 2025–2026.*

*Restricciones de fabricación propias: impresora 3D 200×200×200 mm · cortadora láser 400×400 mm.
Todos los archivos STL, DXF y STEP están optimizados para estas dimensiones.*
