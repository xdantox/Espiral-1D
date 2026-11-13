# Mapa del repositorio: Espiral-1D

## Núcleo del modelo

- `Heff.py`: define los acoples, parámetros físicos y el Hamiltoniano efectivo.
- `E.py`: calcula energetías y observables del sistema.
- `chain.py`: orquesta la dinámica y la integración del modelo.
- `cadena0.py`: construye estados iniciales para la cadena.

## Diagnóstico y análisis

- `relax.py`: análisis de relajación, fase, dimerización y deformación.
- `fft.py`: espectro en frecuencia y vector de onda como función del tiempo.
- `transmision.py`: medición de transmisión y atenuación de magnones.
- `check.py`: validación rápida del estado final.
- `history.py`: análisis temporal de la historia de la simulación.
- `fitq.py`: ajuste de q y parámetros de fondo.
- `plot_*.py`: visualizaciones específicas.
- `fig*.py`: figuras y diagnósticos puntuales.

## Optimización y ansatz

- `direct3.py`
- `direct3_PBC.py`
- `direct3_PBC_folded.py`
- `Jmodified.py`
- `JDS.py`
- `Jdiscreta.py`

Estos scripts suelen servir para construir estados iniciales, parametrizarlos y ajustar el ansatz.

## Visualización y resultados

- `animation.py`
- `gif.py`
- `soliton_transition.gif`
- `*.png`, `*.pdf`: figuras generadas por las simulaciones.

## Recomendación de uso

Cuando se quiere revisar el proyecto por primera vez, conviene seguir este orden:

1. `Heff.py`
2. `cadena0.py`
3. `chain.py`
4. `relax.py`
5. `fft.py`
6. `transmision.py`

Este orden permite comprender primero el modelo, luego el estado inicial, la dinámica, y finalmente los observables físicos.
