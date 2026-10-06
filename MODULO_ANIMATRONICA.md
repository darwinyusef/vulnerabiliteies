# Módulo — Animatrónica para Robótica
*Crear robots con rostro, expresión, piel y presencia desde cero*

> Este módulo transforma un robot funcional en un ser con presencia física:
> ojos que siguen, bocas que sincronizan con el habla, músculos faciales que expresan,
> piel sintética que se mueve naturalmente. IA integrada que hace todo converger.

---

## Mapa del módulo

```
AN.1  Anatomía facial robótica — estructura base
AN.2  Ojos robóticos — diseño, movimiento y expresión
AN.3  Boca y mandíbula — sincronización de labios (visemas)
AN.4  Músculos faciales — servos y tensores en malla
AN.5  Piel sintética — polímeros, silicona y mezclas
AN.6  Magnetismo en animatrónica — conexiones e interacción
AN.7  IA de interacción — LLM + visión + audio + expresión
AN.8  Visemas — sincronización labial con síntesis de voz
AN.9  Visión por computadora para la cara del robot
AN.10 Modelos de comunicación robot-humano
AN.11 Integración completa — pipeline de expresión en tiempo real
```

---

## AN.1 Anatomía Facial Robótica — Estructura Base

### AN.1.1 Referencia anatómica humana para el diseño

La cara humana tiene ~43 músculos. Para animatrónica de primer nivel se emulan los grupos más expresivos:

| Grupo muscular | Músculos clave | Expresión que controla | DOF mínimos |
|----------------|----------------|----------------------|-------------|
| **Frontalis** | Frontal | Sorpresa, ceño | 2 (cejas ind.) |
| **Orbicularis oculi** | Anillo ocular | Guiño, entrecejo | 4 (párpados ind.) |
| **Corrugator** | Entre cejas | Enojo, concentración | 2 |
| **Zygomaticus** | Mejilla → boca | Sonrisa | 2 |
| **Orbicularis oris** | Anillo labial | Articulación del habla | 4–8 |
| **Depressor anguli** | Comisura → abajo | Tristeza, desagrado | 2 |
| **Mentalis** | Mentón | Duda, disgusto | 1 |

**Mínimo de DOF para expresiones reconocibles:**
- 6 DOF: expresiones básicas (alegría, tristeza, enojo, sorpresa, miedo, asco) — FACS simplificado
- 12 DOF: expresiones sutiles + sincronización labial básica
- 24+ DOF: animatrónica de nivel cinematográfico

### AN.1.2 Diseño del cráneo en CadQuery

```python
import cadquery as cq
import math

def skull_base(
    width_mm: float = 160,
    height_mm: float = 220,
    depth_mm: float = 180,
    wall_mm: float = 4.0,
) -> cq.Workplane:
    """
    Cráneo robótico hueco: esfera achatada con cavidades para
    servos, cámara, altavoz y electrónica.
    Debe caber en impresora 200×200×200 → dividir en mitades.
    """
    # Medio cráneo frontal (parte A — se imprime boca abajo)
    skull = (
        cq.Workplane("XY")
        .ellipseArc(width_mm/2, height_mm/2, 0, 180, 0)
        .close()
        .revolve(360, (0,0,0), (0,1,0))
        .shell(-wall_mm)
    )
    # Cavidad para cámara (ojos): dos orificios frontales
    eye_l = cq.Workplane("XZ").center(-32, 80).circle(18).extrude(depth_mm)
    eye_r = cq.Workplane("XZ").center( 32, 80).circle(18).extrude(depth_mm)

    # Cavidad para micrófono en el área auricular
    mic_l = cq.Workplane("YZ").center(-70, 60).circle(8).extrude(50)
    mic_r = cq.Workplane("YZ").center( 70, 60).circle(8).extrude(50)

    skull = skull.cut(eye_l).cut(eye_r).cut(mic_l).cut(mic_r)
    return skull

# División en dos mitades para impresora 200×200×200 mm
skull_full = skull_base()
skull_front = skull_full.intersect(
    cq.Workplane("XY").box(200, 100, 200).translate((0, 50, 100))
)
skull_back = skull_full.intersect(
    cq.Workplane("XY").box(200, 100, 200).translate((0, -50, 100))
)
cq.exporters.export(skull_front, "skull_front.stl")  # imprime en posición A
cq.exporters.export(skull_back,  "skull_back.stl")   # imprime en posición B
# Ensamble: pegado con epoxy + pasadores M3 alineados
```

### AN.1.3 Materiales estructurales del cráneo

| Material | Uso en el cráneo | Por qué |
|----------|-----------------|---------|
| **PETG** | Estructura principal del cráneo | Resistente, no frágil, post-impresión fácil |
| **TPU 95A** | Zonas de flexión (párpados, labios) | Dobla sin romperse |
| **TPU 85A** | Mejillas, mentón | Movimiento más blando |
| **Resina ABS-like (SLA)** | Dientes, cornea falsa | Alta definición superficial |
| **Acero inoxidable** | Ejes de servo, articulaciones | Carga y durabilidad |
| **Aluminio 6061** | Base del cuello, bracket servo principal | Ligereza estructural |

---

## AN.2 Ojos Robóticos — Diseño, Movimiento y Expresión

### AN.2.1 Anatomía del ojo mecánico

```
Vista en explosión del ojo robótico:

[Párpado superior] ── servo SG90 empuja vástago
[Córnea] ──────────── semiesfera transparente impresa en resina
[Iris] ─────────────── disco pintado / impreso + LED detrás (iris glow)
[Globo ocular] ────── semiesfera PETG blanca Ø28 mm
[Músculo H/V] ─────── 2× servos MG90S en yoke (pan + tilt)
[Yoke socket] ─────── bracket impreso PETG que aloja el globo
[Párpado inferior] ── pasivo (spring) o servo adicional
```

### AN.2.2 Mecanismo de movimiento ocular (yoke)

```python
import cadquery as cq

def eye_yoke(
    globe_d: float = 28.0,   # diámetro del globo ocular
    yoke_w:  float = 35.0,   # ancho del yoke
    servo_w: float = 12.0,   # ancho del servo MG90S
):
    """
    Yoke de 2 DOF (pan + tilt) para un ojo robótico.
    El servo de pan mueve el yoke exterior; el de tilt mueve el globo dentro del yoke.
    """
    # Anillo exterior (pan — movimiento horizontal)
    outer = (
        cq.Workplane("XY")
        .circle(yoke_w/2 + 3)
        .circle(yoke_w/2)
        .extrude(8)
        .faces(">Y").workplane()
        .hole(3.0, 15)   # eje de servo pan
        .faces("<Y").workplane()
        .hole(3.0, 15)
    )
    # Anillo interior (tilt — movimiento vertical)
    inner = (
        cq.Workplane("XZ")
        .circle(globe_d/2 + 2)
        .circle(globe_d/2)
        .extrude(6)
        .faces(">X").workplane()
        .hole(3.0, 15)   # eje de servo tilt
    )
    # Socket del globo ocular
    socket = (
        cq.Workplane("XY")
        .sphere(globe_d/2 + 0.5)   # tolerancia 0.5 mm
        .shell(-2.0)
        .intersect(cq.Workplane("XY").box(globe_d+4, globe_d+4, globe_d/2+2)
                   .translate((0,0,-(globe_d/4+1))))
    )
    return outer, inner, socket

outer, inner, socket = eye_yoke()
cq.exporters.export(outer,  "eye_yoke_outer.stl")
cq.exporters.export(inner,  "eye_yoke_inner.stl")
cq.exporters.export(socket, "eye_socket.stl")
```

### AN.2.3 Control de los ojos en Rust (ESP32)

```rust
use esp_idf_hal::ledc::{config::TimerConfig, LedcDriver, LedcTimerDriver, Resolution};
use std::time::Duration;

struct EyeController {
    pan_l:  LedcDriver<'static>,   // ojo izq, horizontal
    tilt_l: LedcDriver<'static>,   // ojo izq, vertical
    pan_r:  LedcDriver<'static>,   // ojo der, horizontal
    tilt_r: LedcDriver<'static>,   // ojo der, vertical
    blink_l: LedcDriver<'static>,  // párpado izq
    blink_r: LedcDriver<'static>,  // párpado der
}

impl EyeController {
    // Convierte ángulo −90..+90° a pulso PWM de servo (500–2500 µs)
    fn angle_to_duty(&self, angle_deg: f32, timer_hz: u32) -> u32 {
        let pulse_us = 1500.0 + angle_deg * (1000.0 / 90.0);
        let period_us = 1_000_000.0 / timer_hz as f32;
        ((pulse_us / period_us) * self.max_duty() as f32) as u32
    }

    pub fn look_at(&mut self, pan_deg: f32, tilt_deg: f32) {
        let duty_pan  = self.angle_to_duty(pan_deg,  50);
        let duty_tilt = self.angle_to_duty(tilt_deg, 50);
        self.pan_l.set_duty(duty_pan).unwrap();
        self.pan_r.set_duty(-duty_pan + self.max_duty()).unwrap();  // espejo
        self.tilt_l.set_duty(duty_tilt).unwrap();
        self.tilt_r.set_duty(duty_tilt).unwrap();
    }

    pub fn blink(&mut self) {
        // Parpadeo natural: 150 ms cerrar, 100 ms abrir
        self.blink_l.set_duty(self.angle_to_duty(45.0, 50)).unwrap();
        self.blink_r.set_duty(self.angle_to_duty(45.0, 50)).unwrap();
        FreeRtos::delay_ms(150);
        self.blink_l.set_duty(self.angle_to_duty(0.0, 50)).unwrap();
        self.blink_r.set_duty(self.angle_to_duty(0.0, 50)).unwrap();
    }
}
```

### AN.2.4 Parpadeo natural con ruido Perlin

```python
import numpy as np

def natural_blink_schedule(duration_s: float = 60, avg_blinks_per_min: float = 15):
    """
    Genera secuencia de parpadeos con distribución natural (Poisson).
    Los humanos parpadean 10–20 veces/min con intervalos aleatorios.
    """
    rate = avg_blinks_per_min / 60   # parpadeos por segundo
    times = []
    t = 0
    while t < duration_s:
        interval = np.random.exponential(1 / rate)
        t += interval
        times.append(t)
    return times

def micro_saccade(current_pan: float, current_tilt: float) -> tuple[float, float]:
    """
    Micro-movimiento sacádico: los ojos humanos hacen pequeños movimientos
    involuntarios de ±2° cada 150–200 ms. Da vida al robot.
    """
    delta_pan  = np.random.normal(0, 0.8)
    delta_tilt = np.random.normal(0, 0.5)
    return (
        np.clip(current_pan  + delta_pan,  -30, 30),
        np.clip(current_tilt + delta_tilt, -20, 20),
    )
```

---

## AN.3 Boca y Mandíbula — Sincronización de Labios

### AN.3.1 Mecanismo de la mandíbula

```
Servo principal (1 DOF):
  Servo MG996R → palanca → mandíbula inferior → apertura máx 30°

Labios (4–8 DOF para visemas):
  Comisura izq/der:  2× SG90 (tracción por hilo de nylon 0.5 mm)
  Labio sup centro:  1× SG90 (levanta el filtrum)
  Labio inf centro:  1× SG90 (baja el mentón independiente del jaw)
  Pout / protrusión: 1× micro servo (empuja los labios hacia adelante)
```

### AN.3.2 Diseño de la mandíbula en CadQuery

```python
def jaw_mechanism(
    jaw_width: float = 90,
    jaw_depth: float = 55,
    jaw_height: float = 30,
    hinge_d:   float = 4.0,
    servo_arm: float = 20.0,
):
    """
    Mandíbula inferior articulada con eje en la articulación témporo-mandibular.
    El servo empuja la palanca desde la parte interior del cráneo.
    """
    jaw = (
        cq.Workplane("XY")
        .ellipse(jaw_width/2, jaw_depth/2)
        .extrude(jaw_height)
        .edges("|Z").fillet(5)
        .faces(">Z").shell(-3.5)
    )
    # Pivote (bisagra)
    pivot = (
        cq.Workplane("YZ").center(0, jaw_height)
        .circle(hinge_d/2).extrude(jaw_width + 10).translate((-jaw_width/2 - 5, 0, 0))
    )
    # Palanca del servo (arm interno)
    arm = (
        cq.Workplane("XY").center(0, -jaw_depth/2 + 10)
        .rect(6, servo_arm).extrude(6)
    )
    return jaw.union(pivot).union(arm)

jaw = jaw_mechanism()
cq.exporters.export(jaw, "jaw_lower.stl")
```

---

## AN.4 Músculos Faciales — Servos y Tensores en Malla

### AN.4.1 Dos tecnologías de "músculo"

| Tecnología | DOF | Fuerza | Velocidad | Sonido | Costo | Uso |
|-----------|-----|--------|----------|--------|-------|-----|
| **Servo micro (SG90/MG90S)** | 1 por servo | 1.8–2.5 kg·cm | Rápido | Audible | $1–3 | Cejas, comisuras |
| **Servo de alto par (DS3218)** | 1 por servo | 18 kg·cm | Medio | Audible | $8 | Mandíbula |
| **Hilo de nylon (tracción)** | 1 por hilo | 0.5–2 kg | Muy rápido | Silencioso | $0.1 | Labios, párpados |
| **Alambre SMA (Nitinol)** | 1 por hilo | 0.1–0.5 kg | Lento (1 Hz) | Silencioso | $5/m | Micro expresiones |
| **Actuador lineal soft** | 1 | Bajo | Lento | Silencioso | $15 | Robótica blanda |

### AN.4.2 Topología de músculos faciales para 12-DOF

```
ESP32 → PCA9685 (16 canales PWM I²C) → 12 servos

Canal 0:  Ceja izquierda (arriba/abajo)
Canal 1:  Ceja derecha (arriba/abajo)
Canal 2:  Corrugator (cejas juntas → enojo)
Canal 3:  Párpado superior izquierdo
Canal 4:  Párpado superior derecho
Canal 5:  Mandíbula (apertura)
Canal 6:  Comisura labial izquierda
Canal 7:  Comisura labial derecha
Canal 8:  Labio superior (levantar)
Canal 9:  Labio inferior (bajar)
Canal 10: Labio protrusión (uj, oo)
Canal 11: Mentón (mentalis)
```

### AN.4.3 Control de 12 servos con PCA9685 en Rust

```rust
use pwm_pca9685::{Address, Channel, Pca9685};

struct FaceController {
    pca: Pca9685<I2cDriver<'static>>,
}

#[derive(Clone)]
struct FaceState {
    brow_l:    f32,  // −1.0 baja .. +1.0 sube
    brow_r:    f32,
    corrugator: f32, // 0.0 relajado .. 1.0 fruncido
    lid_l:     f32,  // 0.0 abierto .. 1.0 cerrado
    lid_r:     f32,
    jaw:       f32,  // 0.0 cerrado .. 1.0 abierto
    corner_l:  f32,  // −1.0 abajo .. +1.0 sonrisa
    corner_r:  f32,
    lip_upper: f32,
    lip_lower: f32,
    lip_pout:  f32,
    chin:      f32,
}

impl FaceController {
    fn apply(&mut self, state: &FaceState) {
        let values: [f32; 12] = [
            state.brow_l, state.brow_r, state.corrugator,
            state.lid_l,  state.lid_r,  state.jaw,
            state.corner_l, state.corner_r,
            state.lip_upper, state.lip_lower, state.lip_pout, state.chin,
        ];
        for (ch, &v) in values.iter().enumerate() {
            let duty = ((v + 1.0) / 2.0 * 1000.0 + 500.0) as u16;  // 500–1500 µs
            self.pca.set_channel_on_off(Channel::from(ch as u8), 0, duty).unwrap();
        }
    }

    fn interpolate(&self, from: &FaceState, to: &FaceState, t: f32) -> FaceState {
        FaceState {
            brow_l:    from.brow_l    + (to.brow_l    - from.brow_l)    * t,
            brow_r:    from.brow_r    + (to.brow_r    - from.brow_r)    * t,
            corrugator: from.corrugator + (to.corrugator - from.corrugator) * t,
            lid_l:     from.lid_l     + (to.lid_l     - from.lid_l)     * t,
            lid_r:     from.lid_r     + (to.lid_r     - from.lid_r)     * t,
            jaw:       from.jaw       + (to.jaw       - from.jaw)       * t,
            corner_l:  from.corner_l  + (to.corner_l  - from.corner_l)  * t,
            corner_r:  from.corner_r  + (to.corner_r  - from.corner_r)  * t,
            lip_upper: from.lip_upper + (to.lip_upper - from.lip_upper) * t,
            lip_lower: from.lip_lower + (to.lip_lower - from.lip_lower) * t,
            lip_pout:  from.lip_pout  + (to.lip_pout  - from.lip_pout)  * t,
            chin:      from.chin      + (to.chin      - from.chin)      * t,
        }
    }
}

// Poses base de expresiones (FACS simplificado)
fn pose_happy() -> FaceState {
    FaceState { corner_l: 0.8, corner_r: 0.8, lid_l: 0.2, lid_r: 0.2,
                brow_l: 0.1, brow_r: 0.1, ..FaceState::default() }
}
fn pose_sad() -> FaceState {
    FaceState { corner_l: -0.7, corner_r: -0.7, brow_l: -0.3, brow_r: -0.3,
                corrugator: 0.5, ..FaceState::default() }
}
fn pose_angry() -> FaceState {
    FaceState { corrugator: 1.0, brow_l: -0.5, brow_r: -0.5,
                corner_l: -0.3, corner_r: -0.3, ..FaceState::default() }
}
fn pose_surprise() -> FaceState {
    FaceState { brow_l: 1.0, brow_r: 1.0, lid_l: -0.3, lid_r: -0.3,
                jaw: 0.4, ..FaceState::default() }
}
```

---

## AN.5 Piel Sintética — Polímeros, Silicona y Mezclas

### AN.5.1 Tipos de materiales de piel

| Material | Shore A | Elongación | Trasparencia | Costo | Uso |
|----------|---------|-----------|-------------|-------|-----|
| **Silicona Platin-30** | 30 | 700 % | Alta | Alto | Mejillas, cuello — piel premium |
| **Silicona Platin-10** | 10 | 800 % | Alta | Alto | Párpados, labios — ultra suave |
| **Dragonskin 10 NV** | 10 | 1000 % | Media | Alto | Piel de alta elongación |
| **Ecoflex 00-30** | ~0 | 900 % | Media | Medio | Zonas de alta deformación |
| **TPU 85A (impresión)** | ~85 | 450 % | Baja | Bajo | Orejas, frente rígida |
| **Latex natural** | 40–60 | 500 % | No | Bajo | Alternativa económica (alergia) |
| **Gelatina + glicerina** | Soft | 300 % | Alta | Muy bajo | Prototipo rápido |

### AN.5.2 Mezclas de polímeros para propiedades específicas

#### Mezcla base para piel de robot relacional (tono carne, flexible)

```
Formulación base (100 g de mezcla total):

Componente A: Silicona Platin-30 Part A     40 g
Componente B: Silicona Platin-30 Part B     40 g
Ecoflex 00-30 (ablandador)                  15 g  → reduce dureza final
Pigmento piel (óxido de hierro + titanio)    3 g  → tono natural
Agente fibra de algodón micro               2 g  → textura porosa
─────────────────────────────────────────────────
Total                                       100 g

Mezcla: A + B → mezclar 3 min → añadir Ecoflex → pigmento → fibra
Desgasificación: 5 min en cámara de vacío (10 mbar)
Vertido: en molde del negativo facial
Curado: 4–6 h a 25°C o 1 h a 60°C (horno convencional)
Desmolde: esperar 12 h para resistencia completa
```

#### Mezcla para labios (alta elongación, tono rosado)

```
Dragonskin 10 NV Part A         45 g
Dragonskin 10 NV Part B         45 g
Pigmento rojo/rosado             5 g
Agente traslúcido (Sil-Poxy)    5 g
─────────────────────────────────
Shore A final: ~8–12
Elongación:    800 %
Curado: 75 min a 25°C
```

#### Piel de prototipo rápido con gelatina (sin silicona)

```python
# Receta de piel de prototipo: gelatina + glicerina + agua
# No dura, pero permite iterar el molde en horas y costo $0.50

def gelatina_skin(bloom_strength: int = 240):
    """
    bloom_strength: 150 (suave) – 250 (firme)
    Relación: bloom alto = más firme (más gelatina)
    """
    recipe = {
        "water_ml":     100,
        "gelatin_g":    14 if bloom_strength >= 200 else 10,  # Knox / bovine
        "glycerin_ml":  15,   # plastificante — evita que se quiebre
        "pigment_g":    0.5,  # tono piel
        "process": [
            "1. Mezclar agua + gelatina en frío → hidratar 5 min",
            "2. Calentar a 70°C sin hervir, mezclar hasta disolver",
            "3. Añadir glicerina + pigmento, mezclar",
            "4. Colar y verter en molde a 55°C (si enfría se gelifica prematuramente)",
            "5. Refrigerar 1h → desmoldar",
            "⚠️ Vida útil: 3–7 días sin conservante; añadir 0.1% ácido sórbico para 2 semanas",
        ]
    }
    return recipe
```

### AN.5.3 Moldes para fabricar la piel

```
Proceso de molde negativo facial:

1. Escanear o medir el cráneo impreso en 3D
2. En Blender: invertir la malla (negativo) con solidify 5 mm
3. Imprimir el molde en PLA (superficie lisa, 0.1 mm layer, 100% relleno)
4. Tratar superficie: lija 400 → 800 → IPA → spray desmoldante (Ease Release 200)
5. Verter silicona desgasificada en dos fases:
   a. Primera capa 2 mm (pincel) → curar 30 min
   b. Capa de refuerzo con gasa de médico embebida en silicona
   c. Capas finales hasta 5–7 mm total
6. Desmolde: con ayuda de aire comprimido suave en los bordes
7. Recorte: tijeras de manicura, bisturí quirúrgico
8. Pegado al cráneo: Sil-Poxy (silicona adhesiva)
```

---

## AN.6 Magnetismo en Animatrónica

### AN.6.1 Aplicaciones de imanes en la cara robótica

| Aplicación | Tipo de imán | Función | Ventaja |
|-----------|-------------|---------|---------|
| **Conexión modular de piel** | N35 Ø10×3 mm | Fijar piel al cráneo sin tornillos | Desmontaje rápido para mantenimiento |
| **Cierre de mandíbula** | N35 Ø8×2 mm | Mantener boca cerrada por defecto | Fuerza pasiva, no gasta energía |
| **Tracker ocular magnético** | Hall sensor | Detectar posición del globo ocular | Retroalimentación de posición |
| **Piel activa** | Array electromagnético | Deformar piel sin servo visible | Expresión ultra suave |
| **Sensor de contacto facial** | Reed switch | Detectar toque en mejilla | Reacción háptica al tacto humano |

### AN.6.2 Sistema magnético de montaje modular de piel

```python
import cadquery as cq

def magnetic_skin_mount(
    magnet_d: float = 10.0,
    magnet_h: float = 3.0,
    n_mounts: int = 8,
    skull_radius: float = 80.0,
):
    """
    Postes con cavidad para imanes N35, distribuidos en el perímetro del cráneo.
    La piel de silicona lleva imanes contrarios incrustados durante el vertido.
    """
    # Posiciones en el perímetro frontal del cráneo
    angles = [i * (360 / n_mounts) for i in range(n_mounts)]
    mounts = cq.Workplane("XY")
    for angle in angles:
        r = math.radians(angle)
        x, y = skull_radius * math.cos(r), skull_radius * math.sin(r)
        mount = (
            cq.Workplane("XY").center(x, y)
            .circle(magnet_d/2 + 2)
            .extrude(magnet_h + 2)
            .faces(">Z").workplane()
            .circle(magnet_d/2)
            .cutBlind(-magnet_h)   # cavidad para el imán
        )
        mounts = mounts.union(mount)
    return mounts
```

### AN.6.3 Leer sensor Hall para posición ocular

```rust
use esp_idf_hal::adc::{AdcDriver, attenuation::DB_11};

fn read_eye_position(adc: &mut AdcDriver, hall_x: AdcChannel, hall_y: AdcChannel)
    -> (f32, f32)
{
    let raw_x = adc.read(&mut hall_x).unwrap() as f32;
    let raw_y = adc.read(&mut hall_y).unwrap() as f32;
    // Normalizar: 0–4095 ADC → −45°..+45°
    let angle_x = (raw_x / 4095.0 - 0.5) * 90.0;
    let angle_y = (raw_y / 4095.0 - 0.5) * 90.0;
    (angle_x, angle_y)
}
```

---

## AN.7 IA de Interacción — LLM + Visión + Audio + Expresión

### AN.7.1 Pipeline completo de interacción

```
[Micrófono] ──► Whisper STT ──► texto del usuario
                                    │
[Cámara]   ──► YOLO / ArcFace ──► quién es, emoción del usuario
                                    │
                              LLM (GPT-4o / Claude)
                              + contexto del robot
                              + memoria episódica (ChromaDB)
                                    │
                              Respuesta textual + emoción target
                              + acciones físicas
                              │                     │
                         Coqui TTS / XTTS      FaceController
                              │                     │
                         Audio + visemas      Servos faciales
                              └──────────────────►  Sincronizados
```

### AN.7.2 Detección de emoción del usuario para respuesta empática

```python
from deepface import DeepFace
import cv2

def analyze_user_emotion(frame: np.ndarray) -> dict:
    """
    Analiza la emoción del humano frente al robot.
    Retorna la emoción dominante y score de confianza.
    """
    try:
        result = DeepFace.analyze(
            frame, actions=['emotion'], enforce_detection=False, silent=True
        )
        emotion   = result[0]['dominant_emotion']
        scores    = result[0]['emotion']
        valence   = scores['happy'] - scores['sad'] - scores['angry'] * 0.5
        return {"emotion": emotion, "scores": scores, "valence": valence}
    except Exception:
        return {"emotion": "neutral", "scores": {}, "valence": 0.0}

# Adaptación de la respuesta del LLM según la emoción del usuario
EMPATHY_SYSTEM_PROMPT = """
Eres un robot asistente con expresión facial. Adapta tu tono:
- Usuario triste → respuesta cálida y reconfortante, expresión de empatía
- Usuario enojado → calma, voz suave, no discutas
- Usuario feliz → entusiasta, usa palabras positivas
- Usuario neutro → informativo y natural

Devuelve SIEMPRE un JSON con:
{
  "text": "tu respuesta",
  "emotion": "happy|sad|thinking|curious|neutral|empathic",
  "actions": ["nod", "tilt_head", "blink"] (lista de acciones opcionales)
}
"""
```

### AN.7.3 LLM con tool calling para control físico del robot

```python
import anthropic
import json

client = anthropic.Anthropic()

FACE_TOOLS = [
    {
        "name": "set_expression",
        "description": "Cambia la expresión facial del robot",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string",
                               "enum": ["happy","sad","angry","surprised",
                                        "thinking","empathic","neutral","excited"]},
                "intensity": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "duration_ms": {"type": "integer"},
            },
            "required": ["expression"],
        }
    },
    {
        "name": "look_at_user",
        "description": "Mueve los ojos hacia el usuario detectado",
        "input_schema": {
            "type": "object",
            "properties": {
                "face_x_norm": {"type": "number"},  # −1.0 izq .. +1.0 der
                "face_y_norm": {"type": "number"},  # −1.0 abajo .. +1.0 arriba
            },
            "required": ["face_x_norm", "face_y_norm"],
        }
    },
    {
        "name": "nod",
        "description": "Hace que el robot asienta con la cabeza",
        "input_schema": {
            "type": "object",
            "properties": {
                "n_times": {"type": "integer", "default": 1},
            },
        }
    },
]

def robot_respond(user_text: str, user_emotion: str, face_position: dict) -> str:
    messages = [
        {"role": "user", "content": f"[Emoción del usuario: {user_emotion}] {user_text}"}
    ]
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system=EMPATHY_SYSTEM_PROMPT,
        tools=FACE_TOOLS,
        messages=messages,
    )
    for block in response.content:
        if block.type == "tool_use":
            dispatch_to_robot(block.name, block.input)
        elif block.type == "text":
            return block.text
    return ""

def dispatch_to_robot(tool: str, params: dict):
    """Envía el comando al ESP32 vía MQTT."""
    import paho.mqtt.publish as publish
    publish.single(f"robot/face/{tool}", json.dumps(params), hostname="broker.local")
```

---

## AN.8 Visemas — Sincronización Labial con Síntesis de Voz

### AN.8.1 ¿Qué es un visema?

Un **visema** es la posición visible de los labios y mandíbula que corresponde a un fonema (sonido).
La sincronización labial convierte la secuencia de fonemas del TTS en movimientos físicos de servos.

### AN.8.2 Tabla de visemas para el español

| Visema | Fonemas | Posición labial | Jaw | Corner | Lip pout | Lip upper |
|--------|---------|----------------|-----|--------|---------|-----------|
| **A** | /a/ | Boca muy abierta | 0.9 | 0.0 | 0.0 | 0.0 |
| **E** | /e/ | Labios estirados | 0.5 | 0.6 | 0.0 | 0.1 |
| **I** | /i/ | Sonrisa plana | 0.3 | 0.9 | 0.0 | 0.0 |
| **O** | /o/ | Labios redondeados | 0.6 | -0.3 | 0.5 | 0.0 |
| **U** | /u/ | Labios fruncidos | 0.4 | -0.5 | 0.9 | 0.0 |
| **P/B/M** | /p/ /b/ /m/ | Labios cerrados | 0.0 | 0.0 | 0.0 | 0.0 |
| **F/V** | /f/ /v/ | Labio inf en dientes | 0.2 | 0.0 | 0.0 | 0.0 |
| **TH** | /θ/ | Lengua entre dientes | 0.3 | 0.0 | 0.0 | 0.2 |
| **D/T/L/N** | /d/ /t/ /l/ /n/ | Dientes juntos | 0.1 | 0.0 | 0.0 | 0.1 |
| **K/G** | /k/ /g/ | Boca entreabierta | 0.4 | 0.1 | 0.0 | 0.1 |
| **S/Z** | /s/ /z/ | Dientes juntos | 0.15 | 0.1 | 0.0 | 0.0 |
| **SH/CH** | /ʃ/ /tʃ/ | Labios adelantados | 0.2 | 0.0 | 0.4 | 0.0 |
| **R** | /r/ /ɾ/ | Boca semi-abierta | 0.35 | 0.0 | 0.1 | 0.0 |
| **REST** | silencios | Boca cerrada | 0.0 | 0.0 | 0.0 | 0.0 |

### AN.8.3 Pipeline de sincronización labial en Python

```python
from kokoro import KPipeline   # TTS con timestamps de fonemas
import numpy as np
import time
import paho.mqtt.client as mqtt

# Mapa visema → posición de servos
VISEME_MAP = {
    "A":   {"jaw": 0.9, "corner": 0.0, "lip_pout": 0.0, "lip_upper": 0.0},
    "E":   {"jaw": 0.5, "corner": 0.6, "lip_pout": 0.0, "lip_upper": 0.1},
    "I":   {"jaw": 0.3, "corner": 0.9, "lip_pout": 0.0, "lip_upper": 0.0},
    "O":   {"jaw": 0.6, "corner":-0.3, "lip_pout": 0.5, "lip_upper": 0.0},
    "U":   {"jaw": 0.4, "corner":-0.5, "lip_pout": 0.9, "lip_upper": 0.0},
    "PBM": {"jaw": 0.0, "corner": 0.0, "lip_pout": 0.0, "lip_upper": 0.0},
    "FV":  {"jaw": 0.2, "corner": 0.0, "lip_pout": 0.0, "lip_upper": 0.0},
    "DTL": {"jaw": 0.1, "corner": 0.0, "lip_pout": 0.0, "lip_upper": 0.1},
    "KG":  {"jaw": 0.4, "corner": 0.1, "lip_pout": 0.0, "lip_upper": 0.1},
    "SZ":  {"jaw": 0.15,"corner": 0.1, "lip_pout": 0.0, "lip_upper": 0.0},
    "R":   {"jaw": 0.35,"corner": 0.0, "lip_pout": 0.1, "lip_upper": 0.0},
    "REST":{"jaw": 0.0, "corner": 0.0, "lip_pout": 0.0, "lip_upper": 0.0},
}

PHONEME_TO_VISEME = {
    "a":"A","á":"A","e":"E","é":"E","i":"I","í":"I",
    "o":"O","ó":"O","u":"U","ú":"U","ü":"U",
    "p":"PBM","b":"PBM","m":"PBM",
    "f":"FV","v":"FV",
    "d":"DTL","t":"DTL","l":"DTL","n":"DTL",
    "k":"KG","g":"KG","q":"KG",
    "s":"SZ","z":"SZ","c":"SZ",
    "r":"R","rr":"R",
}

class LipSyncPlayer:
    def __init__(self, mqtt_client, robot_id: str):
        self.mqtt  = mqtt_client
        self.topic = f"robots/{robot_id}/face/lipsync"

    def play(self, text: str, audio_timestamps: list[tuple[str, float, float]]):
        """
        audio_timestamps: lista de (fonema, t_start, t_end) en segundos
        Sincronizado con la reproducción de audio.
        """
        t0 = time.time()
        for phoneme, t_start, t_end in audio_timestamps:
            # Esperar hasta que sea el momento
            wait = t_start - (time.time() - t0)
            if wait > 0:
                time.sleep(wait)

            viseme_key = PHONEME_TO_VISEME.get(phoneme.lower(), "REST")
            pose = VISEME_MAP[viseme_key]

            self.mqtt.publish(self.topic, json.dumps({
                "viseme": viseme_key,
                "duration_ms": int((t_end - t_start) * 1000),
                **pose
            }), qos=0)

        # Al terminar el audio, cerrar la boca
        time.sleep(0.2)
        self.mqtt.publish(self.topic, json.dumps(VISEME_MAP["REST"]), qos=0)
```

---

## AN.9 Visión por Computadora para la Cara del Robot

### AN.9.1 Seguimiento facial para los ojos

```python
import cv2
import mediapipe as mp
import numpy as np

class FaceTracker:
    def __init__(self):
        self.mp_face = mp.solutions.face_detection
        self.detector = self.mp_face.FaceDetection(
            model_selection=1, min_detection_confidence=0.6
        )
        self.mp_mesh = mp.solutions.face_mesh
        self.mesh = self.mp_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )

    def get_face_target(self, frame: np.ndarray) -> tuple[float, float] | None:
        """
        Retorna (pan_norm, tilt_norm) en rango −1.0..+1.0
        para apuntar los ojos del robot hacia el rostro detectado.
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.detector.process(rgb)
        if not res.detections:
            return None
        det = res.detections[0]
        bb = det.location_data.relative_bounding_box
        cx = bb.xmin + bb.width  / 2
        cy = bb.ymin + bb.height / 2
        pan_norm  = -(cx - 0.5) * 2   # 0=centro → inv. porque el robot se "ve" al espejo
        tilt_norm = -(cy - 0.5) * 2
        return pan_norm, tilt_norm

    def detect_user_emotion(self, frame: np.ndarray) -> str:
        """Emoción del usuario desde MediaPipe + heurística de landmarks."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.mesh.process(rgb)
        if not res.multi_face_landmarks:
            return "neutral"
        lm = res.multi_face_landmarks[0].landmark
        # Distancia boca abierta / ancho de boca (aproximación)
        mouth_open   = abs(lm[13].y - lm[14].y)   # labio sup-inf
        mouth_width  = abs(lm[61].x - lm[291].x)
        brow_raise   = (lm[223].y + lm[443].y) / 2  # cejas promedio
        if mouth_open > 0.04 and brow_raise < 0.35:
            return "surprised"
        if mouth_width > 0.15 and mouth_open < 0.02:
            return "happy"
        if brow_raise > 0.45:
            return "sad"
        return "neutral"
```

### AN.9.2 Reconocimiento facial para memoria episódica

```python
from deepface import DeepFace
import chromadb
import numpy as np

class FaceMemory:
    """El robot recuerda a las personas que ha conocido y adapta su comportamiento."""

    def __init__(self, db_path: str = "./face_memory"):
        self.chroma = chromadb.PersistentClient(path=db_path)
        self.col = self.chroma.get_or_create_collection("known_faces")

    def enroll(self, frame: np.ndarray, name: str, notes: str = ""):
        embedding = self._embed(frame)
        if embedding is None:
            return False
        self.col.upsert(
            ids=[name],
            embeddings=[embedding.tolist()],
            metadatas=[{"name": name, "notes": notes, "visits": 1}],
        )
        return True

    def identify(self, frame: np.ndarray, threshold: float = 0.6) -> dict | None:
        embedding = self._embed(frame)
        if embedding is None:
            return None
        result = self.col.query(query_embeddings=[embedding.tolist()], n_results=1)
        if not result["ids"][0]:
            return None
        distance = result["distances"][0][0]
        if distance > threshold:
            return None   # desconocido
        meta = result["metadatas"][0][0]
        # Incrementar contador de visitas
        meta["visits"] += 1
        self.col.upsert(ids=[meta["name"]], metadatas=[meta],
                        embeddings=[embedding.tolist()])
        return meta

    def _embed(self, frame: np.ndarray) -> np.ndarray | None:
        try:
            rep = DeepFace.represent(frame, model_name="ArcFace",
                                     enforce_detection=False)[0]
            return np.array(rep["embedding"])
        except Exception:
            return None
```

---

## AN.10 Modelos de Comunicación Robot-Humano

### AN.10.1 Modelo de turno de conversación (Turn-Taking)

```python
import threading
import time
from enum import Enum

class ConversationState(Enum):
    LISTENING  = "listening"
    PROCESSING = "processing"
    SPEAKING   = "speaking"
    WAITING    = "waiting"   # fin de turno del robot, espera respuesta

class TurnTakingModel:
    """
    Gestiona quién habla: el robot no interrumpe, el humano puede interrumpir.
    Inspirado en el modelo de Levinson & Torreira (2015).
    """
    def __init__(self):
        self.state = ConversationState.LISTENING
        self.silence_threshold_s = 1.5    # silencio del usuario = fin de turno
        self.overlap_tolerance_s = 0.3    # el robot espera overlap antes de ceder

    def on_user_voice_start(self):
        if self.state == ConversationState.SPEAKING:
            # Usuario interrumpe: el robot cede el turno
            self.robot_stop_speaking()
            self.state = ConversationState.LISTENING

    def on_user_voice_end(self):
        if self.state == ConversationState.LISTENING:
            # Esperar el silencio umbral antes de responder
            threading.Timer(self.silence_threshold_s, self._take_turn).start()

    def _take_turn(self):
        if self.state == ConversationState.LISTENING:
            self.state = ConversationState.PROCESSING
            # Aquí inicia el pipeline: STT → LLM → TTS + visemas

    def robot_stop_speaking(self):
        # Señal al módulo de audio y servos para detenerse limpiamente
        pass
```

### AN.10.2 Proxémica — distancia social en el robot

```python
class ProxemicsModel:
    """
    Edward Hall (1966): el espacio personal define el comportamiento social.
    El robot adapta su comportamiento según la distancia al usuario.
    """
    INTIMATE   = (0.0, 0.45)   # < 45 cm: muy próximo → el robot retrocede
    PERSONAL   = (0.45, 1.2)  # 45 cm – 1.2 m: conversación natural ← ÓPTIMO
    SOCIAL     = (1.2, 3.7)   # 1.2 – 3.7 m: voz más alta, gestos amplios
    PUBLIC     = (3.7, 999)   # > 3.7 m: modo espera / atracción

    def get_zone(self, distance_m: float) -> str:
        for zone, (lo, hi) in {
            "intimate": self.INTIMATE, "personal": self.PERSONAL,
            "social": self.SOCIAL,    "public": self.PUBLIC,
        }.items():
            if lo <= distance_m < hi:
                return zone
        return "public"

    def adapt_behavior(self, distance_m: float) -> dict:
        zone = self.get_zone(distance_m)
        return {
            "intimate": {"tts_volume": 0.4, "gesture_scale": 0.5,
                         "eye_contact": False, "action": "back_up"},
            "personal": {"tts_volume": 0.7, "gesture_scale": 1.0,
                         "eye_contact": True,  "action": "engage"},
            "social":   {"tts_volume": 1.0, "gesture_scale": 1.3,
                         "eye_contact": True,  "action": "wave"},
            "public":   {"tts_volume": 0.3, "gesture_scale": 0.3,
                         "eye_contact": False, "action": "attract"},
        }[zone]
```

### AN.10.3 Memoria episódica y continuidad de conversación

```python
import chromadb
import json
from datetime import datetime
from anthropic import Anthropic

class EpisodicMemory:
    """
    El robot recuerda conversaciones pasadas y construye una relación
    con cada usuario a lo largo del tiempo.
    """
    def __init__(self):
        self.chroma = chromadb.PersistentClient("./episodic_memory")
        self.episodes = self.chroma.get_or_create_collection("episodes")
        self.anthropic = Anthropic()

    def store_episode(self, person: str, summary: str, sentiment: float):
        self.episodes.add(
            ids=[f"{person}_{datetime.now().isoformat()}"],
            documents=[summary],
            metadatas=[{"person": person, "sentiment": sentiment,
                        "date": datetime.now().isoformat()}],
        )

    def recall(self, person: str, current_topic: str, n: int = 3) -> str:
        results = self.episodes.query(
            query_texts=[f"{person}: {current_topic}"],
            where={"person": person},
            n_results=n,
        )
        if not results["documents"][0]:
            return "Primera vez que hablo con esta persona."
        memories = "\n".join(results["documents"][0])
        return f"Recuerdos de {person}:\n{memories}"

    def build_system_prompt(self, person: str | None, topic: str) -> str:
        if person:
            memories = self.recall(person, topic)
        else:
            memories = "Persona desconocida."
        return f"""Eres un robot animatrónico con memoria y expresión facial.
Tienes recuerdos de conversaciones anteriores que debes usar naturalmente.
{memories}
Responde de forma natural, breve (< 3 oraciones) y empática.
Devuelve JSON: {{"text": "...", "emotion": "...", "actions": [...]}}"""
```

---

## AN.11 Integración Completa — Pipeline de Expresión en Tiempo Real

### AN.11.1 Arquitectura del sistema completo

```
┌─────────────────────────────────────────────────────────────────┐
│                    PC / Raspberry Pi 5                           │
│                                                                   │
│  [Cámara USB]──►[FaceTracker]──►[Identidad + Emoción usuario]   │
│                                          │                        │
│  [Micrófono]──►[Whisper STT]──►[Texto]──┤                        │
│                                          ▼                        │
│                               [LLM + Memoria]                    │
│                                    │                              │
│          ┌─────────────────────────┤                              │
│          │                         │                              │
│   [Coqui TTS]──►[Audio]    [Expresión target + acciones]         │
│          │                         │                              │
│   [LipSync Player]                 │                              │
│          │                         │                              │
└──────────┼─────────────────────────┼──────────────────────────────┘
           │ MQTT robots/face/...    │ MQTT robots/face/expression
           ▼                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ESP32 (Rust)                                   │
│                                                                   │
│  [MQTT client]──►[FaceController]──►[PCA9685]──►[12 servos]     │
│                       │                                           │
│               [EyeController]──►[4 PWM]──►[4 servos ojos]       │
│                       │                                           │
│               [NeckController]──►[2 PWM]──►[2 servos cuello]    │
│                                                                   │
│  [Hall sensors]──►[Eye position feedback]                         │
│  [FSR sensors]──►[Touch detection]                                │
└─────────────────────────────────────────────────────────────────┘
```

### AN.11.2 Bucle principal en Python

```python
import asyncio
import cv2
from face_tracker import FaceTracker
from face_memory import FaceMemory
from episodic_memory import EpisodicMemory
from lip_sync import LipSyncPlayer
from turn_taking import TurnTakingModel
from proxemics import ProxemicsModel
import whisper
import paho.mqtt.client as mqtt

class AnimatronicBrain:
    def __init__(self, robot_id: str = "robot-01"):
        self.robot_id     = robot_id
        self.tracker      = FaceTracker()
        self.face_memory  = FaceMemory()
        self.episode_mem  = EpisodicMemory()
        self.turn_model   = TurnTakingModel()
        self.proxemics    = ProxemicsModel()
        self.stt          = whisper.load_model("small")
        self.mqtt_client  = mqtt.Client("animatronic-brain")
        self.mqtt_client.connect("broker.local", 1883)
        self.lip_sync     = LipSyncPlayer(self.mqtt_client, robot_id)
        self.cap          = cv2.VideoCapture(0)

    async def vision_loop(self):
        """Corre a 30 FPS — sigue cara y actualiza el estado social."""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                await asyncio.sleep(0.033)
                continue

            # Seguimiento ocular
            target = self.tracker.get_face_target(frame)
            if target:
                pan, tilt = target
                self.mqtt_client.publish(
                    f"robots/{self.robot_id}/face/eyes",
                    json.dumps({"pan": pan * 30, "tilt": tilt * 20}), qos=0
                )

            # Emoción del usuario cada 5 frames
            if self.frame_count % 5 == 0:
                emotion = self.tracker.detect_user_emotion(frame)
                self.current_user_emotion = emotion

            self.frame_count += 1
            await asyncio.sleep(0.033)

    async def conversation_loop(self):
        """Espera habla del usuario → responde con expresión + voz + movimiento."""
        while True:
            # Grabar audio cuando el usuario habla
            audio = await self.record_user_speech()
            result = self.stt.transcribe(audio, language="es")
            user_text = result["text"].strip()
            if not user_text:
                continue

            # Identificar quién habla
            frame = self.cap.read()[1]
            person_meta = self.face_memory.identify(frame)
            person_name = person_meta["name"] if person_meta else "desconocido"

            # Construir contexto y llamar al LLM
            system = self.episode_mem.build_system_prompt(person_name, user_text)
            response = await self.call_llm(system, user_text,
                                           self.current_user_emotion)

            # Generar audio y sincronizar labios
            audio_file, timestamps = await self.tts_with_timestamps(response["text"])
            self.lip_sync.play(response["text"], timestamps)
            await self.play_audio(audio_file)

            # Guardar episodio
            self.episode_mem.store_episode(
                person_name, f"User: {user_text} | Robot: {response['text']}",
                sentiment=1.0 if response.get("emotion") == "happy" else 0.0
            )

    async def run(self):
        self.frame_count = 0
        self.current_user_emotion = "neutral"
        await asyncio.gather(self.vision_loop(), self.conversation_loop())

if __name__ == "__main__":
    brain = AnimatronicBrain()
    asyncio.run(brain.run())
```

---

## Checklist de entrega del proyecto animatrónico

### Mecánica
- [ ] Cráneo diseñado en CadQuery, dividido en mitades ≤ 200×200 mm, impreso en PETG
- [ ] Mecanismo de yoke para 2 ojos con 2 DOF cada uno (4 servos)
- [ ] Párpados funcionales (servo o nylon) con parpadeo natural
- [ ] Mandíbula articulada con servo MG996R
- [ ] Al menos 6 músculos faciales activos (cejas, comisuras, labios)
- [ ] Cuello con 2 DOF (pan + tilt)

### Piel y acabados
- [ ] Molde negativo impreso en PLA + tratado con desmoldante
- [ ] Piel de silicona vertida en molde (Platin-30 o Dragonskin)
- [ ] Imanes de montaje magnético integrados
- [ ] Pigmentación y textura apropiada

### Electrónica
- [ ] ESP32 como cerebro principal
- [ ] PCA9685 para 16 canales PWM (12 cara + 4 ojos)
- [ ] Cámara (OV2640 o USB) integrada en órbita ocular
- [ ] Micrófono MEMS I2S (INMP441 o SPH0645)
- [ ] Altavoz + amplificador MAX98357A
- [ ] Alimentación: 5V 3A (servo) + 3.3V (lógica)

### Software
- [ ] FaceTracker: seguimiento con MediaPipe a 30 FPS
- [ ] FaceMemory: reconocimiento con ArcFace + ChromaDB
- [ ] STT: Whisper.cpp o API (latencia < 1 s)
- [ ] LLM con tool calling: expresión + acciones físicas
- [ ] TTS: Coqui XTTS con voz clonada
- [ ] LipSync: visemas sincronizados con audio
- [ ] TurnTaking: no interrumpe al usuario
- [ ] EpisodicMemory: recuerda personas y conversaciones
- [ ] MQTT: toda la comunicación PC ↔ ESP32

### Expresiones validadas
- [ ] Alegría: comisuras arriba, párpados semi-cerrados
- [ ] Tristeza: comisuras abajo, cejas internas arriba
- [ ] Enojo: corrugator máximo, cejas bajas
- [ ] Sorpresa: cejas arriba, boca abierta, ojos abiertos
- [ ] Empatía: cabeza ladeada, expresión suave, contacto visual
- [ ] Pensamiento: mirada arriba-izquierda, ceja levantada

---

## BOM del proyecto animatrónico completo

| Categoría | Componente | Qty | Costo aprox. |
|-----------|-----------|-----|-------------|
| **Servos cara** | MG90S / SG90 (12 canales) | 12 | $24 |
| **Servo mandíbula** | MG996R | 1 | $5 |
| **Servos cuello** | DS3218 (18 kg·cm) | 2 | $16 |
| **Driver PWM** | PCA9685 16ch I²C | 1 | $3 |
| **MCU** | ESP32-S3 DevKit | 1 | $7 |
| **Cámara** | OV2640 / USB 1080p | 1 | $5–15 |
| **Micrófono** | INMP441 I2S MEMS | 2 | $4 |
| **Altavoz** | 8 Ω / 3 W + MAX98357A | 1 | $4 |
| **Imanes montaje** | N35 Ø10×3 mm ×20 | 20 | $3 |
| **Silicona** | Platin-30 Part A+B 500 g | 1 kit | $30 |
| **Pigmentos** | Silicone pigment kit | 1 | $8 |
| **Filamento** | PETG 1 kg | 1 | $20 |
| **Filamento** | TPU 85A 500 g | 1 | $12 |
| **Filamento** | TPU 95A 500 g | 1 | $12 |
| **Alambre nylon** | 0.5 mm, 10 m | 1 | $2 |
| **Rodamientos** | 608ZZ ×10 | 10 | $5 |
| **Ejes acero** | 8 mm × 60 mm ×6 | 6 | $6 |
| **Epoxy** | Loctite EA 9466 / Sil-Poxy | 2 | $10 |
| **Total estimado** | | | **~$176** |

---

*Módulo de animatrónica diseñado para el stack del curso: Rust (ESP32) + Python (visión + IA) + CadQuery (diseño 3D) + PETG/TPU/silicona (fabricación). Todas las piezas estructurales se imprimen en 200×200×200 mm divididas si es necesario.*
