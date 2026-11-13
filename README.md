# Espiral-1D

Simulaciones y análisis numérico de cadenas magnéticas unidimensionales: espirales de espín, estados antiferromagnéticos, solitones y transporte de magnones.

> **Estado del repositorio:** investigación en desarrollo. La raíz contiene scripts de distintas etapas y datos/resultados auxiliares; no todos los scripts son una canalización única ni se han validado para ejecutarse sin los datos locales correspondientes.

## Qué contiene

El código explora la dinámica de espines y el análisis de estados modulados, incluyendo ajuste de ansatz, espectros en espacio y frecuencia, estabilidad térmica y transmisión de excitaciones.

## Para empezar

1. Revisa el [mapa de archivos](docs/REPO_MAP.md) para identificar los módulos y el orden de lectura.
2. Usa Python 3.10 o posterior e instala las dependencias:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
   python -m pip install -r requirements.txt
   ```

3. Antes de ejecutar un análisis, comprueba los nombres/rutas de datos que espera el script. Varios programas cargan historias `.npy` específicas que no están incluidas en el repositorio. Los análisis que importan `relax3D.py` requieren ese módulo y, según el caso, sus datos de entrada.

No se recomienda lanzar `chain.py`, `stab_termica.py` u otros cálculos largos sin revisar primero sus parámetros, tamaño de salida y requisitos de datos.

## Módulos orientativos

- `Heff.py`, `E.py`, `dot.py`: campo efectivo, energía y operaciones del modelo.
- `chain.py`, `rot.py`, `rot_methods.py`: dinámica e integradores.
- `cadena0.py`, `direct3*.py`: construcción/optimización de estados y ansatz.
- `relax.py`, `relax3D.py`, `history.py`: diagnósticos de estados e historias.
- `fft.py`, `fftBEoptimized.py`, `transmision.py`: análisis espectral y transporte.
- `stab_termica.py`, `flopproxy.py`: pruebas de estabilidad/ruido y protocolos relacionados.
- `plot_*.py`, `fig*.py`, `gap*.py`, `satelite*.py`: análisis y visualizaciones específicas.

La lista no es exhaustiva; consulta [docs/REPO_MAP.md](docs/REPO_MAP.md) antes de elegir un script.

## Datos y resultados

Las historias de simulación pueden ser grandes y suelen guardarse como `.npy`. El `.gitignore` excluye formatos binarios comunes para evitar subir resultados pesados accidentalmente. Los archivos ya versionados permanecen en el repositorio; no se borran ni se mueven automáticamente.

## Reproducibilidad y límites actuales

- Algunos scripts usan rutas de datos escritas directamente en el código; adapta esas rutas a tus archivos locales.
- Fija y registra las semillas aleatorias al comparar corridas. No todos los scripts están parametrizados para recibirlas desde la línea de comandos.
- Las salidas pueden depender del tamaño de sistema, parámetros físicos, bibliotecas y hardware. Registra esos valores junto con cada figura o historia.
- No se declara una licencia de uso en este README; consulta al autor antes de reutilizar el código si no hay una licencia publicada.

## Documentación

- [Mapa del repositorio y dependencias entre scripts](docs/REPO_MAP.md)
