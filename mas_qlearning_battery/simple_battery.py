"""
simple_battery.py
=================
Modelo físico simplificado de batería para el sistema MAS Q-Learning.

Parámetros actualizados a los del Reto 3
-----------------------------------------
    capacity_kwh = 200 kWh  (antes 50)
    max_power_kw =  50 kW   (antes 10)
    efficiency   =  0.95/dirección → ~0.90 round-trip
    soc_min      =  0.05    → 10 kWh mínimo (5% de 200)
    duration     =  4 h     (200 kWh / 50 kW)

Rol en el MAS
--------------
La batería actúa como BUFFER PASIVO de la microred, no como agente
negociador. Cada timestep:
  - Si (solar + eólica) > demanda → carga con el excedente
  - Si (solar + eólica) < demanda → descarga para cubrir el déficit

Esta lógica se ejecuta ANTES de la negociación FIPA-ACL entre AS y AE,
reduciendo la demanda efectiva que los productores deben cubrir.

El SoC resultante se pasa como variable de estado a los agentes AS y AE
para que aprendan cómo adaptar su estrategia de negociación según el
nivel de reserva de la batería.
"""


class SimpleBattery:
    """
    Batería con eficiencia constante y SoC mínimo configurable.

    Parámetros
    ----------
    capacity_kwh  : float  Capacidad total [kWh].                Default 200.0
    initial_soc   : float  SoC inicial por episodio [0-1].       Default 0.5
    charge_eff    : float  Eficiencia de carga [0-1].            Default 0.95
    discharge_eff : float  Eficiencia de descarga [0-1].         Default 0.95
    max_power_kw  : float  Potencia máxima carga/descarga [kW].  Default 50.0
    soc_min       : float  SoC mínimo de seguridad [0-1].        Default 0.05
    """

    def __init__(
        self,
        capacity_kwh: float = 200.0,
        initial_soc: float = 0.5,
        charge_eff: float = 0.95,
        discharge_eff: float = 0.95,
        max_power_kw: float = 50.0,
        soc_min: float = 0.05
    ):
        self.capacity     = capacity_kwh
        self.initial_soc  = initial_soc
        self.soc          = initial_soc
        self.charge_eff   = charge_eff
        self.discharge_eff = discharge_eff
        self.max_power_kw = max_power_kw
        self.soc_min      = soc_min

    # ------------------------------------------------------------------
    # Operaciones físicas
    # ------------------------------------------------------------------

    def charge(self, kw: float) -> float:
        """
        Carga la batería con hasta kw kW.
        Limita por capacidad disponible y potencia máxima.
        Returns kW realmente consumidos del exterior.
        """
        kw = max(0.0, min(float(kw), self.max_power_kw))
        max_storable_kwh = (1.0 - self.soc) * self.capacity
        max_kw_input = max_storable_kwh / self.charge_eff
        actual_kw = min(kw, max_kw_input)
        self.soc += (actual_kw * self.charge_eff) / self.capacity
        self.soc = min(1.0, self.soc)
        return actual_kw

    def discharge(self, kw: float) -> float:
        """
        Descarga la batería hasta kw kW.
        Respeta soc_min: no descarga por debajo del nivel mínimo.
        Returns kW realmente entregados al exterior.
        """
        kw = max(0.0, min(float(kw), self.max_power_kw))
        usable_kwh = max(0.0, (self.soc - self.soc_min) * self.capacity)
        max_kw_output = usable_kwh * self.discharge_eff
        actual_kw = min(kw, max_kw_output)
        self.soc -= actual_kw / (self.capacity * self.discharge_eff)
        self.soc = max(self.soc_min, self.soc)
        return actual_kw

    # ------------------------------------------------------------------
    # Consultas de capacidad (sin modificar SoC)
    # ------------------------------------------------------------------

    def available_discharge_kw(self) -> float:
        """Máxima potencia entregable respetando soc_min [kW]."""
        usable = max(0.0, (self.soc - self.soc_min) * self.capacity)
        return min(usable * self.discharge_eff, self.max_power_kw)

    def available_charge_kw(self) -> float:
        """Máxima potencia absorbible [kW]."""
        room = (1.0 - self.soc) * self.capacity
        return min(room / self.charge_eff, self.max_power_kw)

    # ------------------------------------------------------------------
    # Gestión de episodio
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reinicia SoC al valor inicial. Llamar al inicio de cada episodio."""
        self.soc = self.initial_soc

    def __repr__(self) -> str:
        return (
            f"SimpleBattery(soc={self.soc:.3f}, "
            f"capacity={self.capacity} kWh, "
            f"max_power={self.max_power_kw} kW, "
            f"soc_min={self.soc_min})"
        )
