# Mapa del repositorio — Espiral-1D

La raíz contiene prototipos y herramientas de varias etapas de investigación. Los nombres se describen según el código consultado; verifica cada script y sus datos de entrada antes de ejecutarlo. No hay un único punto de entrada que ejecute todos los análisis.

## Modelo y dinámica

- `Heff.py`: parámetros y rutinas para calcular el campo efectivo.
- `E.py`: cálculo de energía.
- `dot.py`: operaciones auxiliares de producto escalar/cálculo vectorizado.
- `chain.py`: orquestación de una simulación de cadena (comprueba primero las rutas de entrada).
- `rot.py`, `rot_methods.py`: rutinas de evolución/integración.
- `B_ext.py`: campo externo.
- `cadena0.py`, `1spin.py`, `rand.py`: estados iniciales y utilidades de espines.

## Estados, ansatz y optimización

- `direct3.py`, `direct3_PBC.py`, `direct3_PBC_folded.py`: variantes de construcción/ajuste de estados con condiciones de contorno y ansatz.
- `JDS.py`, `Jdiscreta.py`, `Jmodified.py`: variantes de modelo/acoplamiento.
- `alpha.py`, `alpha2.py`, `m2.py`: cálculos auxiliares/modelos; revisar su contexto antes de interpretarlos como parte del modelo de espines.

## Análisis de estados e historias

- `relax.py`, `relax3D.py`, `relax3Dexact.py`: relajación y análisis de estados.
- `history.py`: análisis de una secuencia temporal guardada.
- `check.py`, `fitq.py`, `perfil_banda.py`, `plot_ansatz_theta.py`, `plot_q_local.py`: comprobaciones y análisis específicos.
- `stab_termica.py`, `flop.py`, `flopproxy.py`: protocolos de estabilidad térmica y transición.

## Espectros, transporte y visualización

- `fft.py`, `fftBEoptimized.py`: análisis FFT.
- `transmision.py`: análisis de transmisión de magnones.
- `animation.py`, `gif.py`: animaciones.
- `fig.py`, `gap.py`, `gaps.py`, `gapteorico.py`, `satelite.py`, `satelite2.py`, `satelitem.py`: figuras y análisis puntuales.

## Datos y figuras

- Los archivos `.npy` son datos/historias de simulación y pueden ocupar mucho espacio.
- Los `.png`, `.pdf` y `.gif` son resultados visuales.
- `helix_spin_chain.html` y `html.html` son recursos HTML.

## Orden recomendado para una primera lectura

1. `Heff.py` y `E.py` para las convenciones del modelo.
2. `cadena0.py` para entender cómo se construyen ciertos estados iniciales.
3. `chain.py` y las rutinas de `rot*.py` para la dinámica.
4. `relax3D.py`/`relax.py` para diagnóstico.
5. `fft.py` y `transmision.py` para observables.

Este orden es orientativo: algunos scripts son independientes, experimentales o requieren archivos locales que no están disponibles para una instalación limpia.
