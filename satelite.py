import numpy as np
from dataclasses import dataclass, field
from scipy.special import jv
from functools import lru_cache
from scipy.sparse import dok_matrix
import matplotlib.pyplot as plt

#CAPA 1 PARAMETROS

@dataclass(frozen=True)
class ModelParams:
    """
    Contenedor inmutable para los parámetros físicos del Hamiltoniano.
    Calcula automáticamente parámetros derivados (Q, epsilon) al instanciarse.
    """
    # --- Parámetros Físicos ---
    J: float      # Jnn: Primeros vecinos
    dJ: float     # dJnn: Dimerización (delta J)
    J2: float     # Jnnn: Segundos vecinos
    K: float      # Ka: Bicuadrático
    D: float      # De: Anisotropía In-Plane
    S: float      # Magnitud del Spin

    # --- Parámetros de la Textura Magnética ---
    q: float      # Vector de la espiral magnética
    gamma: float  # Fase de dimerización
    alpha: float  # Amplitud de deformación en sitio (alpha_ind)
    phi_0: float  # Fase global arbitraria (phi_ind)
    
    # --- Parámetros Derivados (Calculados en __post_init__) ---
    Q: float = field(init=False)       # Vector de la red de solitones (2q)
    epsilon: float = field(init=False) # Amplitud efectiva de enlace
    
    def __post_init__(self):
        # 1. Conmensurabilidad Q = 2q
        object.__setattr__(self, 'Q', 2 * self.q)
        
        # 2. Relación geométrica sitio-enlace
        # epsilon = 2 * alpha * sin(q)
        eps_calc = 2 * self.alpha * np.sin(self.q)
        object.__setattr__(self, 'epsilon', eps_calc)

@dataclass(frozen=True)
class SimConfig:
    """
    Parámetros numéricos y de control para la simulación.
    """
    N_max: int    # Cutoff de la base de armónicos (|m| <= N_max)
    N_k: int      # Resolución de la zona de Brillouin (puntos k)

def get_default_params():
    """
    Carga los parámetros del minimizador de energías proporcionado.
    """
    return ModelParams(
        # Constantes de Acoplamiento
        J       = -46.75,    # Jnn
        dJ      = -44.85,    # dJnn
        J2      = -2.6,      # Jnnn
        K       = -45.4,     # Ka
        D       = -0.76,     # De
        S       = 1.0,      # <--- VERIFICAR SPIN (Asumido 1.0)
        
        # Parámetros Variacionales
        q       = 1.03321,
        gamma   = -1.2867,
        alpha   = -0.0036,  # alpha_ind (Amplitud en el sitio)
        phi_0   = 0.0000    # phi_ind
    )

# --- Test de Sanidad ---
if __name__ == "__main__":
    params = get_default_params()
    

#CAPA 2 LAMBDA FACTORY

class LambdaFactory:
    """
    Capa 2: Fábrica de Coeficientes Efectivos (Lambda).
    
    Responsabilidad:
    Calcula las amplitudes de interacción Lambda_m independientes de k,
    basadas en las expansiones de Jacobi-Anger y las tablas de paridad
    definidas en el modelo teórico.
    """
    
    def __init__(self, params: ModelParams, config: SimConfig):
        self.p = params
        self.c = config
        
        # Pre-cálculo de argumentos de Bessel para eficiencia
        # Argumento estándar (J, K_hopping): epsilon
        self.arg_std = self.p.epsilon
        # Argumento duplicado (K_masa, D): 2 * epsilon
        self.arg_dbl = 2 * self.p.epsilon
        # Argumento segundos vecinos (J2): 2 * epsilon * cos(Q/2)
        # Nota: Q = 2q, entonces cos(Q/2) = cos(q)
        self.arg_J2 = 2 * self.p.epsilon * np.cos(self.p.q)

    @lru_cache(maxsize=128)
    def _bessel(self, m: int, arg: float) -> float:
        """Wrapper de Bessel Jv con caché."""
        return jv(m, arg)

    def _get_neumann(self, m: int) -> float:
        """Factor 2 para la expansión de Jacobi-Anger (1 si m=0)."""
        return 1.0 if m == 0 else 2.0

    def _get_parity_sign(self, m: int) -> int:
        """
        Retorna (-1)^k.
        Si m es par (2k) -> (-1)^(m/2).
        Si m es impar (2k+1) -> (-1)^((m-1)/2).
        """
        if m % 2 == 0:
            k = m // 2
        else:
            k = (m - 1) // 2
        return 1 if k % 2 == 0 else -1

    def get_J(self, m: int):
        """
        Coeficientes para Intercambio Primeros Vecinos (J).
        Argumento Bessel: epsilon.
        """
        # Factor global común: -2 * S * J_m(eps) * (-1)^k * (Neumann)
        # Nota: El -2 viene de la definición del Hamiltoniano efectivo.
        # El (-1)^k viene de la expansión de cos/sin.
        
        bessel_val = self._bessel(m, self.arg_std)
        neumann = self._get_neumann(m)
        sign = self._get_parity_sign(m)
        
        # Prefactor base: - S * J_m * factor_expansion
        # (El 2 del neumann se multiplica por el prefactor del H)
        prefactor = -1.0 * self.p.S * bessel_val * sign * neumann

        cq = np.cos(self.p.q)
        sq = np.sin(self.p.q)
        c2g = np.cos(2 * self.p.gamma)
        s2g = np.sin(2 * self.p.gamma)

        if m % 2 == 0:
            # Par: Proviene de cos(Theta_n)
            # Sym: cos(q)cos(2g), Anti: sin(q)sin(2g)
            lam_sym = prefactor * (self.p.J * cq * c2g + self.p.dJ * sq * s2g)
            lam_anti = prefactor * (self.p.dJ * cq * c2g + self.p.J * sq * s2g)
        else:
            # Impar: Proviene de -sin(Theta_n)
            # Sym: -sin(q)cos(2g), Anti: cos(q)sin(2g)
            
            lam_sym = -prefactor * (self.p.J * sq * c2g - self.p.dJ * cq * s2g)
            # Nota: El antisimétrico tiene signo opuesto en la definición de J impar
            # Anti J impar = +2K... -> Necesitamos invertir el signo del prefactor (que lleva un -)
            lam_anti = -prefactor * (self.p.dJ * sq * c2g - self.p.J * cq * s2g)

        return {'sym': lam_sym, 'anti': lam_anti}

    def get_J_bare(self, m: int):
        """
        Coeficientes para el Hopping Longitudinal Desnudo (Sector u).
        
        Física:
        Corresponde al término - sum [ J + (-1)^n dJ ] u_n u_{n+1}.
        Al no estar modulado por la textura angular (cos DeltaTheta), 
        su espectro en la base de solitones es una Delta de Kronecker en m=0.
        """
        # 1. Regla de Selección Espectral (Solo existe en m=0)
        if m != 0:
            return {'sym': 0.0, 'anti': 0.0}
        
        # 2. Amplitudes Estáticas (Para m=0)
        # Absorben el signo menos global del Hamiltoniano y el Spin S.
        # No llevan factores de Bessel ni de Neumann (son términos exactos).
        
        # Parte Uniforme (-J sum u_n u_{n+1}) -> Canal Simétrico
        lam_sym = -1.0 * self.p.J * self.p.S
        
        # Parte Alternante (-dJ sum (-1)^n u_n u_{n+1}) -> Canal Antisimétrico
        lam_anti = -1.0 * self.p.dJ * self.p.S
        
        return {'sym': lam_sym, 'anti': lam_anti}
    
    def get_J2(self, m: int):
        """
        Coeficientes para Segundos Vecinos (J2).
        Argumento Bessel: 2*epsilon*cos(q).
        Solo contribuye al canal Directo (Sym).
        """
        bessel_val = self._bessel(m, self.arg_J2)
        neumann = self._get_neumann(m)
        
        # Lambda = -2 * J2 * S * J_m * cos(2q + m*pi/2)
        # cos(2q + m*pi/2) maneja la paridad automáticamente:
        # m par: (-1)^(m/2) cos(2q)
        # m impar: -(-1)^((m-1)/2) sin(2q)
        
        phase_term = np.cos(2 * self.p.q + m * np.pi / 2)
        
        lam_val = -1.0 * self.p.J2 * self.p.S * bessel_val * neumann * phase_term
        
        return {'sym': lam_val, 'anti': 0.0}
    def get_J2_bare(self, m: int):
        """
        Coeficientes para el Hopping Longitudinal Desnudo de Segundos Vecinos (J2).
        
        Física:
        Corresponde al término - sum J2 u_n u_{n+2}.
        1. No tiene modulación angular (Solo existe en m=0).
        2. No tiene dimerización (Anti = 0), ya que la distancia n->n+2 
           suma un enlace corto y uno largo, cancelando la alternancia.
        """
        # 1. Regla de Selección Espectral
        if m != 0:
            return {'sym': 0.0, 'anti': 0.0}
        
        # 2. Amplitud Estática (m=0)
        # Uniforme: -J2 * S
        lam_sym = -1.0 * self.p.J2 * self.p.S
        
        return {'sym': lam_sym, 'anti': 0.0}
    def get_D_bare(self, m: int):
        """
        Anisotropía de Eje Difícil (D) para el sector Longitudinal (u).
        
        Física:
        H = D * sum(u_n^2).
        Es un término 'bare' (sin modulación angular).
        Solo contribuye a la masa diagonal en m=0.
        """
        # 1. Regla de Selección (Solo m=0)
        if m != 0:
            return {'sym': 0.0, 'anti': 0.0}
            
        # 2. Amplitud Estática (m=0)
        # Retornamos el valor D puro.
        # El factor 2 (por ser u^2) se aplicará en el GammaBuilder o 
        # se debe documentar que Lambda aqui es D.
        # Nota: Para consistencia con el código H_0 antiguo que sumaba 2*D,
        # retornamos D aquí y GammaBuilder aplicará el factor 2 del operador.
        
        return {'sym': self.p.D, 'anti': 0.0}
    def get_K(self, m: int):
        """
        Calcula TODOS los coeficientes Lambda para la interacción Bicuadrática (K).
        
        Unifica la lógica de los sectores 'u' y 'v' en una sola llamada.
        - Sector v: Depende de argumentos dobles (2*epsilon).
        - Sector u (Masa): Es 1/2 del sector v.
        - Sector u (Hopping): Depende de argumentos simples (epsilon).
        """
        
        # =========================================================
        # PARTE 1: LÓGICA DE ARGUMENTO DOBLE (Sector v + Masa u)
        # =========================================================
        # Prefactor base: -K * S^2
        prefactor_dbl = -1.0 * self.p.K * (self.p.S**2)
        
        # Bessel y Neumann con argumento DOBLE (2*epsilon)
        b_dbl = self._bessel(m, self.arg_dbl) 
        neu = self._get_neumann(m)
        sign = self._get_parity_sign(m)
        
        base_dbl = prefactor_dbl * b_dbl * neu * sign
        
        c2q, s2q = np.cos(2*self.p.q), np.sin(2*self.p.q)
        c4g, s4g = np.cos(4*self.p.gamma), np.sin(4*self.p.gamma)
        
        # --- Cálculo de Canales Sym/Anti (Base v) ---
        if m % 2 == 0:
            # Par: Mantiene signos
            lam_v_sym = base_dbl * (c2q * c4g)
            lam_v_anti = base_dbl * (s2q * s4g)
        else:
            # Impar: Sym invierte signo (-), Anti mantiene (+)
            lam_v_sym = -base_dbl * (s2q * c4g)
            lam_v_anti = base_dbl * (c2q * s4g)
            
        # --- Derivación Masa u (1/2 de v) ---
        lam_u_mass_sym = 0.5 * lam_v_sym
        lam_u_mass_anti = 0.5 * lam_v_anti

        # =========================================================
        # PARTE 2: LÓGICA DE ARGUMENTO SIMPLE (Hopping u)
        # =========================================================
        # Prefactor Hopping: -2 * K * S^2
        prefactor_hop = -2.0 * self.p.K * (self.p.S**2)
        
        # Bessel con argumento ESTÁNDAR (epsilon)
        b_std = self._bessel(m, self.arg_std)
        
        base_hop = prefactor_hop * b_std * neu * sign
        
        cq, sq = np.cos(self.p.q), np.sin(self.p.q)
        c2g, s2g = np.cos(2*self.p.gamma), np.sin(2*self.p.gamma)
        
        # --- Cálculo de Canales Hopping u ---
        if m % 2 == 0:
            # Par
            lam_hop_sym = base_hop * (cq * c2g)
            lam_hop_anti = base_hop * (sq * s2g)
        else:
            # Impar
            lam_hop_sym = -base_hop * (sq * c2g)
            lam_hop_anti = base_hop * (cq * s2g)

        # =========================================================
        # RETORNO UNIFICADO
        # =========================================================
        return {
            # Componentes para Sector V (Goldstone)
            'sym': lam_v_sym,
            'anti': lam_v_anti,
            
            # Componentes para Sector U (Masa + Hopping)
            'mass_sym': lam_u_mass_sym,
            'mass_anti': lam_u_mass_anti,
            'hop_sym': lam_hop_sym,
            'hop_anti': lam_hop_anti
        }
    def get_K_bare(self, m: int):
        """
        Término BARE (Isotrópico) de la Interacción Bicuadrática (Sector u).
        
        Física:
        Proviene de la parte constante de la linealización: 
        cos^2(DeltaTheta) = 1/2 + 1/2 cos(2*DeltaTheta).
        
        El Hamiltoniano de masa es:
        H = -2 K S^2 * sum [ -1/2 (u_n^2 + u_{n+1}^2) * (1/2 + ...) ]
        
        Tomando solo el término constante (1/2):
        H_iso = + K S^2 * sum(u_n^2).
        
        Reglas:
        1. Solo existe en m=0 (No tiene modulación espacial).
        2. Es puramente simétrico (No tiene alternancia).
        """
        # 1. Regla de Selección Espectral
        if m != 0:
            return {'sym': 0.0, 'anti': 0.0}
            
        # 2. Amplitud Estática (m=0)
        # Lambda = + K * S^2
        # El GammaBuilder aplicará el factor 2 (Bose) resultando en un gap de 2KS^2.
        lam_sym = 1.0 * self.p.K * (self.p.S**2)
        
        return {'sym': lam_sym, 'anti': 0.0}
    def get_D_plane_bare(self, m: int):
        """
        Término BARE (Estático) de la Anisotropía In-Plane (Sector u).
        
        Física:
        H_bare = + (D/2) * sum(u_n^2).
        Proviene del promedio de la corrección de longitud S^z.
        Solo existe en m=0.
        """
        # 1. Regla de Selección (Solo m=0)
        if m != 0:
            return {'sym': 0.0, 'anti': 0.0}
            
        # 2. Amplitud Estática (m=0)
        # El coeficiente es +D/2. 
        # (El GammaBuilder aplicará el factor 2 del operador u^2, resultando en un gap D).
        return {'sym': 0.5 * self.p.D, 'anti': 0.0}

    def get_D_plane(self, m: int):
        """
        Coeficientes MODULADOS para Anisotropía In-Plane (D).
        Argumento Bessel: 2*epsilon.
        
        Genera:
        1. Sector v: -D * v^2 * cos(2theta)
        2. Sector u: -(D/2) * u^2 * cos(2theta)  [Mismo perfil, mitad de amplitud]
        """
        # --- Cálculo Base (Equivalente al sector Transversal -D) ---
        
        # Prefactor Base: 2 * D * Bessel * Neumann * Signo
        # (El signo menos del Hamiltoniano -D se maneja en la lógica de paridad abajo)
        bessel_val = self._bessel(m, self.arg_dbl)
        neumann = self._get_neumann(m)
        sign = self._get_parity_sign(m)
        
        base = -1.0 * self.p.D * bessel_val * neumann * sign
        
        c2g = np.cos(2 * self.p.gamma)
        s2g = np.sin(2 * self.p.gamma)
        
        # Lógica de Paridad para Sector v (Referencia)
        if m % 2 == 0:
            # Par (cos): -D * cos -> Signo (-)
            lam_v_sym = base * c2g
            lam_v_anti = -base * s2g   # Signo (+) por convención antisimétrica
        else:
            # Impar (sin): -D * (-sin) -> Signo (+)
            lam_v_sym = -base * c2g
            lam_v_anti = -base * s2g  
            
        # --- Derivación del Sector u (Longitudinal) ---
        # H_u_mod = -(D/2) * u^2 * cos(2theta)
        # Es exactamente la mitad del coeficiente transversal.
        lam_u_sym = 0.5 * lam_v_sym
        lam_u_anti = 0.5 * lam_v_anti
            
        return {
            'v_sym': lam_v_sym, 
            'v_anti': lam_v_anti,
            'u_sym': lam_u_sym,
            'u_anti': lam_u_anti
        }
    def precompute_all(self):
        """
        Calcula y almacena todos los coeficientes Lambda para el rango de armónicos
        relevante (|m| <= N_max + 1).

        Retorna:
            lambdas (dict): Estructura { m: { 'J': {...}, 'J_bare': {...}, ... } }
        """
        # Definimos el rango de armónicos.
        # Se añade +1 al N_max como margen de seguridad para evitar errores de 
        # índice en los bordes de la matriz de bloques.
        limit = self.c.N_max + 1
        m_vals = range(-limit, limit + 1)
        
        lambdas = {}
        
        for m in m_vals:
            lambdas[m] = {
                # --- 1. Intercambio Primeros Vecinos ---
                'J':      self.get_J(m),      # Modulado (Principalmente sector v)
                'J_bare': self.get_J_bare(m), # Estático (Solo m=0, sector u)

                # --- 2. Intercambio Segundos Vecinos ---
                'J2':      self.get_J2(m),      # Modulado (v)
                'J2_bare': self.get_J2_bare(m), # Estático (Solo m=0, sector u)

                # --- 3. Anisotropía (D) ---
                'D_bare':  self.get_D_bare(m),  # Masa Estática (Solo m=0, sector u)
                'D_plane': self.get_D_plane(m), # Modulada (Devuelve dict con claves para u y v)
                'D_plane_bare': self.get_D_plane_bare(m), # Estática (Solo m=0, sector u)

                # --- 4. Interacción Bicuadrática (K) ---
                # Nota: get_K ya contiene la lógica interna 'if sector == u/v'
                # por lo que devolverá la estructura correcta para el sector activo.
                'K_bare': self.get_K_bare(m),    # Estático (Solo m=0, sector u)
                'K': self.get_K(m)
                
            }
            
        return lambdas

#Capa 3 GAMMA FACTORY

class GammaFactory:
    def __init__(self, params, config, lambda_dict):
        self.p = params
        self.c = config
        self.lambdas = lambda_dict

    # Eliminamos _get_vertex_factor porque aplicaremos las fórmulas finales directas

    def get_gamma_J(self, m, k):
        """
        Calcula el acople de intercambio J total.
        
        CORRECCIÓN:
        - Para m=0, el sector u suma la parte 'Bare' (J puro) Y la parte 'Modulada' (J0 de Bessel).
        - Para m!=0, solo contribuye la parte modulada.
        - El sector v siempre toma todo de la parte modulada (ya incluye la geometría correcta).
        """
        val_u = 0.0 + 0.0j
        val_v = 0.0 + 0.0j
        
        mq = m * self.p.q 

        # ---------------------------------------------------------
        # 1. TÉRMINOS BARE (Solo contribuyen a u en m=0)
        # ---------------------------------------------------------
        if m == 0 and 'J_bare' in self.lambdas[0]:
            lb = self.lambdas[0]['J_bare']
            
            # Física: Salto a primeros vecinos sin modulación de textura
            term_sym = lb['sym'] * (-2.0 * np.cos(k)) 
            term_anti = lb['anti'] * (-2.0j * np.sin(k))
            
            val_u += (term_sym + term_anti)

        # ---------------------------------------------------------
        # 2. TÉRMINOS MODULADOS (Contribuyen a todo m, incluido m=0)
        # ---------------------------------------------------------
        # Aquí entra la expansión de Jacobi-Anger:
        # m=0 -> J0(eps) (Renormalización media)
        # m!=0 -> Jm(eps) (Harmónicos)
        
        if 'J' in self.lambdas[m]:
            lm = self.lambdas[m]['J']

            # --- SECTOR LONGITUDINAL (u) ---
            # Sumamos la corrección modulada SIEMPRE.
            # Para m=0, esto añade la corrección de J0(eps) al bare.
            
            # Nota: Verifica el signo de tu Hamiltoniano efectivo.
            # Usualmente es -Sum J_eff * u_n * u_{n+1}
            # El factor cos(mq) viene de la transformada de Fourier del salto n->n+1
            
            # Sym: -Lambda * cos(mq)
            sym_u  = lm['sym'] * (-np.cos(mq))
            
            # Anti: i * Lambda * sin(mq)
            anti_u = lm['anti'] * (1.0j * np.sin(mq))
            
            val_u += (sym_u + anti_u)
            
            # --- SECTOR TRANSVERSAL (v) ---
            # El sector v obtiene toda su física de aquí.
            
            k_half = k / 2.0
            phase_diff = mq - k_half 

            sym_v  = lm['sym']  * (2.0 * np.sin(k_half) * np.sin(phase_diff))
            anti_v = lm['anti'] * (2.0j * np.sin(k_half) * np.cos(phase_diff))
            
            val_v += (sym_v + anti_v)

        return {'val_u': val_u, 'val_v': val_v}
    def get_gamma_J2(self, m, k):
        """
        Calcula los vértices para Segundos Vecinos (J2).
        
        Diferencias con J1:
        1. Distancia de salto doble: k -> 2k, q -> 2q.
        2. Sin término Anti: J2 conecta sitios de la misma paridad (n->n+2),
           por lo que la dimerización (alternancia) se cancela o es nula.
        """
        # Inicialización de acumuladores
        val_u = 0.0 + 0.0j
        val_v = 0.0 + 0.0j

        # Argumento geométrico doble (m * 2Q)
        # Nota: Esto es la transferencia de momento efectiva para la distancia 2
        mQ = m * (2.0 * self.p.q)

        # ---------------------------------------------------------
        # 1. TÉRMINOS BARE (Solo contribuyen a u en m=0)
        # ---------------------------------------------------------
        if m == 0 and 'J2_bare' in self.lambdas[0]:
            lb = self.lambdas[0]['J2_bare']
            
            # Física: Hopping desnudo a segundos vecinos (n -> n+2)
            # Factor 2.0: Suma de ida (e^{-i2k}) y vuelta (e^{i2k})
            # Fórmula: -2 * J2 * cos(2k)
            term_sym = lb['sym'] * 2.0 * np.cos(2.0 * k)
            
            # Sumar al sector u (v toma su rigidez de la modulación)
            val_u += term_sym
            
            # Nota: No hay 'anti' en J2 bare.

        # ---------------------------------------------------------
        # 2. TÉRMINOS MODULADOS (m cualquiera)
        # ---------------------------------------------------------
        if 'J2' in self.lambdas[m]:
            lm = self.lambdas[m]['J2']

            # --- SECTOR LONGITUDINAL (u) ---
            # Corrección de masa/hopping modulada
            # Fórmula: -Lambda * cos(mQ)
            # (El signo negativo viene del Hamiltoniano estándar de Heisenberg)
            
            sym_u = lm['sym'] * (-np.cos(mQ))
            val_u += sym_u
            
            # --- SECTOR TRANSVERSAL (v) ---
            # Protección de Goldstone para distancia 2
            # Fórmula J1: sin(k/2) * sin((mQ-k)/2)
            # Fórmula J2: sin(k)   * sin(mQ - k)   <-- Argumentos dobles
            
            term_geo_v = 2.0 * np.sin(k) * np.sin(mQ - k)
            sym_v = lm['sym'] * term_geo_v
            
            val_v += sym_v

            # Nota: Asumimos Anti=0 para J2. Si tuvieras un modelo exótico con
            # dimerización en segundos vecinos, aquí sumarías:
            # val_u += 1j * lm['anti'] * np.sin(mQ) ... etc.

        return {'val_u': val_u, 'val_v': val_v}
    def get_gamma_K(self, m, k):
        """
        Calcula los vértices para el término Bicuadrático K (S_i . S_j)^2.
        Versión Unificada (BdG): Calcula val_u y val_v simultáneamente.
        """
        # Inicialización de acumuladores complejos
        val_u = 0.0 + 0.0j
        val_v = 0.0 + 0.0j

        # Argumentos geométricos
        # mQ/2 = m*q (Transferencia de momento efectiva)
        mQ_half = m * self.p.q 

        # ---------------------------------------------------------
        # 1. TÉRMINO BARE (m=0, Isotrópico)
        # ---------------------------------------------------------
        # Solo afecta al sector longitudinal (u) como un Gap constante.
        if m == 0 and 'K_bare' in self.lambdas[0]:
            lb = self.lambdas[0]['K_bare']
            # Física: 2 * K * S^2 (Gap masivo)
            # Solo sumamos a val_u. val_v (Goldstone) permanece en 0.
            term_bare = lb['sym'] * 2.0
            val_u += term_bare

        # ---------------------------------------------------------
        # 2. TÉRMINOS MODULADOS (Todo m)
        # ---------------------------------------------------------
        if 'K' in self.lambdas[m]:
            lm = self.lambdas[m]['K']

            # =========================================================
            # SECTOR LONGITUDINAL (u)
            # Sumamos las dos contribuciones físicas: Masa + Hopping
            # =========================================================
            
            # --- A. MASA LONGITUDINAL ---
            # Claves esperadas: 'mass_sym', 'mass_anti'
            if 'mass_sym' in lm:
                # Sym: -2 * Lambda * cos(mQ/2)
                term_mass_sym = lm['mass_sym'] * (-2.0 * np.cos(mQ_half))
                
                # Anti: 2i * Lambda * sin(mQ/2)
                term_mass_anti = lm['mass_anti'] * (2.0j * np.sin(mQ_half))
                
                val_u += (term_mass_sym + term_mass_anti)

            # --- B. HOPPING LONGITUDINAL ---
            # Claves esperadas: 'hop_sym', 'hop_anti'
            if 'hop_sym' in lm:
                arg_hop = k - mQ_half
                
                # Sym: Lambda * cos(k - mQ/2)
                term_hop_sym = lm['hop_sym'] * np.cos(arg_hop)
                
                # Anti: -i * Lambda * sin(mQ/2 - k)
                # Preservamos tu lógica original de argumentos para el seno:
                arg_shadow = mQ_half - k
                term_hop_anti = lm['hop_anti'] * (-1.0j * np.sin(arg_shadow))
                
                val_u += (term_hop_sym + term_hop_anti)

            # =========================================================
            # SECTOR TRANSVERSAL (v)
            # Solo Hopping Efectivo (Goldstone)
            # =========================================================
            
            # Claves esperadas: 'sym', 'anti' (Heredadas de la lógica base de v)
            if 'sym' in lm:
                k_half = k / 2.0
                # (mQ - k)/2 = mQ_half - k/2
                arg_diff = mQ_half - k_half 
                
                # Sym: 4 * Lambda * sin((mQ-k)/2) * sin(k/2)
                term_geo_sym = 4.0 * np.sin(arg_diff) * np.sin(k_half)
                term_v_sym = lm['sym'] * term_geo_sym
                
                # Anti: 4i * Lambda * cos((mQ-k)/2) * sin(k/2)
                term_geo_anti = 4.0j * np.cos(arg_diff) * np.sin(k_half)
                term_v_anti = lm['anti'] * term_geo_anti
                
                val_v += (term_v_sym + term_v_anti)

        # Retorno unificado para el Assembler
        return {'val_u': val_u, 'val_v': val_v}
    def get_gamma_D(self, m, k):
        """
        Calcula los vértices para la Anisotropía de Eje Difícil (D_xx).
        Base Unificada (BdG).
        
        Física:
        - D penaliza fluctuaciones en el eje ortogonal al plano de la espiral.
        - En coordenadas locales, el eje ortogonal corresponde al campo 'u'.
        - Por tanto, actúa como una masa (Gap) constante solo para val_u.
        """
        # Inicialización de acumuladores
        val_u = 0.0 + 0.0j
        val_v = 0.0 + 0.0j

        # ---------------------------------------------------------
        # ANISOTROPÍA DE EJE DIFÍCIL (D_xx) - BARE
        # ---------------------------------------------------------
        # Invariante ante rotación -> Solo m=0
        
        if m == 0 and 'D_bare' in self.lambdas[0]:
            lb = self.lambdas[0]['D_bare']
            
            # --- SECTOR U (Longitudinal / Out-of-Plane) ---
            # Fórmula Física: 2 * D (Gap masivo constante)
            # El factor 2 proviene de la simetría de Bose del operador u^2 + u^dag^2
            term_sym = lb['sym'] * 2.0
            
            val_u += term_sym
            
            # --- SECTOR V (Transversal / In-Plane) ---
            # D_xx es ortogonal al plano de movimiento de v.
            # En la aproximación estándar, no genera masa para v.
            # val_v se mantiene en 0.0.

        # Nota: La Anisotropía In-Plane (D_plane) que sí afecta a v y u
        # se maneja en el método separado 'get_gamma_D_plane'.

        return {'val_u': val_u, 'val_v': val_v}
    def get_gamma_D_plane(self, m, k, key='D_plane'):
        """
        Calcula los vértices para la Anisotropía en el Plano (D_yy).
        
        Maneja dos casos físicos distintos:
        1. 'D_plane_bare' (m=0): Gap estático. Solo diagonal. NO tiene G+/G-.
        2. 'D_plane' (m!=0): Modulación cos(2theta). Genera Sidebands G+/G-.
        """
        
        # ---------------------------------------------------------
        # 1. TÉRMINO BARE (Estático, Diagonal)
        # ---------------------------------------------------------
        if key == 'D_plane_bare':
            # Solo existe para m=0
            if m == 0 and 'D_plane_bare' in self.lambdas[0]:
                lb = self.lambdas[0]['D_plane_bare']
                
                # --- Sector U ---
                # Física: Gap D.
                # Lambda viene como D/2. El factor 2 es por la simetría de Bose de u^2.
                # Resultado: val_u = D.
                val_u = lb.get('sym', 0.0) * 2.0
                
                # --- Sector V ---
                # El término bare D(u^2) no afecta a v directamente.
                val_v = 0.0 + 0.0j
                
                # Retornamos valores simples para la diagonal
                return {'val_u': val_u, 'val_v': val_v}
            
            # Si no hay datos, retorno nulo seguro
            return {'val_u': 0j, 'val_v': 0j}

        # ---------------------------------------------------------
        # 2. TÉRMINOS MODULADOS (Sidebands G+ / G-)
        # ---------------------------------------------------------
        elif key == 'D_plane' and 'D_plane' in self.lambdas[m]:
            lm = self.lambdas[m]['D_plane']
            
            # Fase Dinámica y Factor de división espectral
            phase_dyn = np.exp(-1j * m * (self.p.phi_0 + self.p.q))
            sideband_factor = 0.5
            
            # --- Definir Paridad (Base Euler) ---
            if m % 2 == 0:
                c_sym_pos,  c_sym_neg  = 1.0,  1.0
                c_anti_pos, c_anti_neg = 1.0j, -1.0j
            else:
                c_sym_pos,  c_sym_neg  = 1.0j, -1.0j
                c_anti_pos, c_anti_neg = 1.0,  1.0

            # --- Sector U (Longitudinal) ---
            raw_u_sym  = lm.get('u_sym', 0.0)
            raw_u_anti = lm.get('u_anti', 0.0)
            
            u_plus = (raw_u_sym * c_sym_pos + raw_u_anti * c_anti_pos) * phase_dyn * sideband_factor
            u_minus = (raw_u_sym * c_sym_neg + raw_u_anti * c_anti_neg) * phase_dyn * sideband_factor

            # --- Sector V (Transversal) ---
            raw_v_sym  = lm.get('v_sym', 0.0)
            raw_v_anti = lm.get('v_anti', 0.0)
            
            v_plus = (raw_v_sym * c_sym_pos + raw_v_anti * c_anti_pos) * phase_dyn * sideband_factor
            v_minus = (raw_v_sym * c_sym_neg + raw_v_anti * c_anti_neg) * phase_dyn * sideband_factor

            # Retornamos estructura de Sidebands
            return {
                'G_plus':  {'val_u': u_plus,  'val_v': v_plus},
                'G_minus': {'val_u': u_minus, 'val_v': v_minus}
            }
            
        # Fallback por defecto (Sidebands vacíos)
        return {'G_plus': {'val_u': 0j, 'val_v': 0j}, 'G_minus': {'val_u': 0j, 'val_v': 0j}}

# Capa 4: HAMILTONIAN ASSEMBLER (Base Unificada BdG)
class HamiltonianAssembler:
    def __init__(self, params, config, gamma_factory, N_max):
        """
        Capa 4: Ensamblaje de la Super-Matriz de Floquet (BdG).
        
        En la nueva base unificada, el vector de estado en cada armónico 'n' es:
            Psi_n = ( u_n, v_n )^T
        
        Por lo tanto, los bloques 2x2 representan la interacción de las especies u y v.
        
        Args:
            params: Parámetros físicos.
            config: Configuración de simulación (Ya no define el sector exclusivo).
            gamma_factory: Instancia de Layer 3 (GammaFactory) que retorna {val_u, val_v}.
            N_max: Cutoff de armónicos (-N_max ... +N_max).
        """
        self.p = params
        self.c = config
        self.factory = gamma_factory
        self.N_max = N_max
        
        # Dimensión total: 
        # (2 especies: u, v) * (2*N_max + 1 armónicos)
        self.dim_blocks = 2 * N_max + 1
        self.dim_total = 2 * self.dim_blocks

    def _get_block_indices(self, n):
        """
        Convierte el índice de armónico n (-N...N) a índices de matriz global.
        
        Mapeo en Base BdG:
        - Índice par (2k)   -> Sector u (Longitudinal) del armónico n
        - Índice impar (2k+1) -> Sector v (Transversal) del armónico n
        """
        # Mapeamos n=[-N_max ... +N_max] -> k=[0 ... 2*N_max]
        k = n + self.N_max
        base_idx = 2 * k
        return base_idx, base_idx + 1

    def _make_hessian_block(self, val_u, val_v):
        """
        Construye el bloque de interacción 2x2 para las especies (u, v).
        
        Args:
            val_u (complex): Vértice efectivo para el sector u.
            val_v (complex): Vértice efectivo para el sector v.
            
        Returns:
            np.array: Bloque diagonal 2x2.
            [[ val_u,   0   ],
             [   0  , val_v ]]
             
        Nota: En el Hamiltoniano de espín linealizado sin términos Dzyaloshinskii-Moriya 
        cruzados explícitos, los sectores u y v se desacoplan en el bloque elemental.
        """
        return np.array([
            [val_u, 0.0 + 0.0j], 
            [0.0 + 0.0j, val_v]
        ], dtype=complex)

    def _add_to_sparse(self, H_sparse, n_row, n_col, block_2x2):
        """
        Suma un bloque 2x2 a la matriz dispersa (DOK) en la posición de bloque (n, n').
        Maneja internamente los límites de la matriz (Boundaries).
        """
        # Chequeo de límites (Boundaries)
        if abs(n_row) > self.N_max or abs(n_col) > self.N_max:
            return

        # Obtener índices reales de la matriz gigante
        r0, r1 = self._get_block_indices(n_row) # r0=u, r1=v
        c0, c1 = self._get_block_indices(n_col) # c0=u, c1=v
        
        # Inyección elemento a elemento (acumulativa)
        # H_uu
        if block_2x2[0, 0] != 0:
            H_sparse[r0, c0] += block_2x2[0, 0]
        # H_uv
        if block_2x2[0, 1] != 0:
            H_sparse[r0, c1] += block_2x2[0, 1]
        # H_vu
        if block_2x2[1, 0] != 0:
            H_sparse[r1, c0] += block_2x2[1, 0]
        # H_vv
        if block_2x2[1, 1] != 0:
            H_sparse[r1, c1] += block_2x2[1, 1]

    def _inject_interaction(self, H_sparse, n_from, n_to, block):
        """
        [NUEVO] Método auxiliar para inyectar interacciones Hermíticas.
        Si inyectamos una interacción M en H[n, n+m], debemos inyectar
        M_dagger en H[n+m, n] para asegurar que el Hamiltoniano sea Hermítico.
        """
        # 1. Bloque Directo (n -> n_to)
        self._add_to_sparse(H_sparse, n_from, n_to, block)
        
        # 2. Bloque Conjugado (n_to -> n)
        # Si n_from == n_to (Diagonal), el bloque ya debe ser Hermítico por construcción
        # y no debemos sumarlo dos veces.
        if n_from != n_to:
            block_dagger = np.conjugate(block.T)
            self._add_to_sparse(H_sparse, n_to, n_from, block_dagger)
    def build_k(self, k):
        """
        Construye la matriz H_eff(k) completa en formato disperso.
        
        Args:
            k (float): Momento del cuasi-partícula en la primera zona de Brillouin.
            
        Returns:
            scipy.sparse.dok_matrix: Matriz Hamiltoniana de dimensión (2*dim_blocks, 2*dim_blocks).
        """
        # Usamos Dictionary of Keys (DOK) para construcción eficiente
        H_sparse = dok_matrix((self.dim_total, self.dim_total), dtype=complex)

        # Pre-calcular listas de armónicos para evitar regeneración en bucles
        all_m = list(self.factory.lambdas.keys())
        m_pos = [m for m in all_m if m >= 0] # Para términos cinéticos simétricos (J, J2, K)

        # =========================================================================
        # BUCLE PRINCIPAL: Iteramos sobre las filas (Bandas de Salida: Bra <n|)
        # =========================================================================
        for n in range(-self.N_max, self.N_max + 1):
            
            # Momento Local Percibido por la banda n (k_n = k + nQ)
            k_local = k + n * self.p.Q

            # ---------------------------------------------------------------------
            # 1. TÉRMINOS CINÉTICOS ESTÁNDAR (J, J2, K, D_xx)
            #    Estos términos acoplan n -> n + m (simétricamente)
            # ---------------------------------------------------------------------
            for m in m_pos: # m = 0, 1, 2...
                
                # Destino del salto: Ket |n + m>
                n_target = n + m
                
                # Si el destino se sale de la matriz truncada, ignoramos
                if abs(n_target) > self.N_max:
                    continue

                # Acumulador para el vértice total en este canal (n -> n+m)
                # Sumamos J + J2 + K + D_hard_axis
                val_u_total = 0.0 + 0.0j
                val_v_total = 0.0 + 0.0j
                
                # --- A. Intercambio J1 ---
                if 'J' in self.factory.lambdas.get(m, {}):
                    # Ojo: get_gamma_J ya maneja internamente J_bare si m=0
                    g = self.factory.get_gamma_J(m, k_local)
                    val_u_total += g['val_u']
                    val_v_total += g['val_v']

                # --- B. Intercambio J2 ---
                if 'J2' in self.factory.lambdas.get(m, {}):
                    g = self.factory.get_gamma_J2(m, k_local)
                    val_u_total += g['val_u']
                    val_v_total += g['val_v']
                
                # --- C. Bicuadrático K ---
                if 'K' in self.factory.lambdas.get(m, {}):
                    g = self.factory.get_gamma_K(m, k_local)
                    val_u_total += g['val_u']
                    val_v_total += g['val_v']

                # --- D. Anisotropía Eje Difícil (Solo m=0) ---
                if m == 0 and 'D_bare' in self.factory.lambdas.get(0, {}):
                    g = self.factory.get_gamma_D(0, k_local) # key implícita
                    val_u_total += g['val_u']
                    val_v_total += g['val_v']

                # --- INYECCIÓN EN LA MATRIZ ---
                # Si hay alguna interacción no nula, la inyectamos
                if val_u_total != 0 or val_v_total != 0:
                    block = self._make_hessian_block(val_u_total, val_v_total)
                    
                    if m == 0:
                        # Diagonal: Se suma directamente (No inject_interaction para evitar doble conteo)
                        self._add_to_sparse(H_sparse, n, n, block)
                    else:
                        # Fuera de diagonal: Inyecta H[n, n+m] y H[n+m, n]^*
                        self._inject_interaction(H_sparse, n, n_target, block)

            # ---------------------------------------------------------------------
            # 2. ANISOTROPÍA EN EL PLANO (D_plane) - TRATAMIENTO ESPECIAL
            #    Tiene lógica bifurcada: Bare (m=0) vs Sidebands (m!=0)
            # ---------------------------------------------------------------------
            
            # --- A. CASO BARE (Masa Estática, m=0) ---
            # Se suma a la diagonal existente
            if 'D_plane_bare' in self.factory.lambdas.get(0, {}):
                # Llamada con key explícita
                g_bare = self.factory.get_gamma_D_plane(0, k_local, key='D_plane_bare')
                
                if g_bare['val_u'] != 0 or g_bare['val_v'] != 0:
                    block_bare = self._make_hessian_block(g_bare['val_u'], g_bare['val_v'])
                    self._add_to_sparse(H_sparse, n, n, block_bare)

            # --- B. CASO MODULADO (Sidebands, m >= 1) ---
            # Itera sobre los armónicos de modulación de la anisotropía
            # Nota: Usamos 'm' como índice de la serie de Fourier de D(x), 
            # pero el salto en la matriz es m +/- 1.
            
            # Recorremos todos los m donde exista D_plane
            for m in m_pos: 
                if 'D_plane' in self.factory.lambdas[m]:
                    # 1. Obtener vértices bifurcados
                    res_mod = self.factory.get_gamma_D_plane(m, k_local, key='D_plane')
                    
                    # 2. Construir bloques 2x2
                    # G+ conecta n -> n + (m+1)
                    u_p, v_p = res_mod['G_plus']['val_u'], res_mod['G_plus']['val_v']
                    block_plus = self._make_hessian_block(u_p, v_p)
                    
                    # G- conecta n -> n + (m-1)
                    u_m, v_m = res_mod['G_minus']['val_u'], res_mod['G_minus']['val_v']
                    block_minus = self._make_hessian_block(u_m, v_m)

                    # 3. Inyección Canal Superior (G+)
                    target_up = n + (m + 1)
                    if abs(target_up) <= self.N_max:
                        self._inject_interaction(H_sparse, n, target_up, block_plus)

                    # 4. Inyección Canal Inferior (G-)
                    target_dn = n + (m - 1)
                    if abs(target_dn) <= self.N_max:
                        # IMPORTANTE: Aquí inyectamos block_minus
                        # La función _inject_interaction se encarga de poner el conjugado en (target_dn, n)
                        self._inject_interaction(H_sparse, n, target_dn, block_minus)

        return H_sparse
