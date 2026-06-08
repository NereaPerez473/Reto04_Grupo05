"""
strategies.py
=============
Estrategias de negociación para agentes productores de energía.

Inspirado en el patrón de la profesora (05bluffing.py) pero adaptado al
dominio energético: en lugar de buyer/seller con un precio único, cada
estrategia define cómo un productor declara su potencia disponible y
a qué precio la oferta.

Cada estrategia recibe:
    real_power_kw   → potencia real disponible en ese timestep (del CSV)
    import_price    → precio de mercado actual en €/kWh (del CSV de precio)

Y devuelve una EnergyProposal con:
    declared_energy_kw  → lo que el agente comunica al consumidor (puede ser ≠ real)
    price_eur_kwh       → precio al que ofrece la energía
    real_energy_kw      → la potencia que REALMENTE puede entregar (privado)

La discrepancia declared vs real es el núcleo del análisis de estrategias:
- Si declared > real  → engaño     → el consumidor recibe menos de lo prometido
- Si declared < real  → ocultación → el consumidor paga más por menos cantidad
- Si declared == real → honestidad → no hay distorsión

Referencia en el paper: analizar cómo cada estrategia afecta al coste total
del consumidor y a los ingresos de los productores.
"""

from dataclasses import dataclass


@dataclass
class EnergyProposal:
    """
    Propuesta de energía generada por un agente productor.

    Atributos
    ---------
    declared_energy_kw : float
        Potencia declarada al consumidor en el mensaje PROPOSE.
        En estrategias manipuladoras difiere de real_energy_kw.
    price_eur_kwh : float
        Precio ofertado en €/kWh.
    real_energy_kw : float
        Potencia real entregable. Solo conocida por el agente productor.
        El consumidor nunca ve este valor directamente — lo infiere
        post-hoc cuando el INFORM reporta la entrega real.
    """
    declared_energy_kw: float
    price_eur_kwh: float
    real_energy_kw: float


class NegotiationStrategies:
    """
    Colección de estrategias de negociación para agentes productores.

    Uso
    ---
    proposal = NegotiationStrategies.apply("deception", real_power, price)
    # o directamente:
    proposal = NegotiationStrategies.deception(real_power, price)
    """

    @staticmethod
    def honest(real_power_kw: float, import_price: float) -> EnergyProposal:
        """
        Honesta: declara potencia real y cobra un 15% por debajo del mercado.

        Lógica: el productor compite con la red siendo ligeramente más barato.
        No hay distorsión entre declarado y entregado.
        Sirve como baseline para comparar con las otras estrategias.
        """
        return EnergyProposal(
            declared_energy_kw=real_power_kw,
            price_eur_kwh=import_price * 0.85,
            real_energy_kw=real_power_kw
        )

    @staticmethod
    def deception(real_power_kw: float, import_price: float,
                  bluff_factor: float = 1.3) -> EnergyProposal:
        """
        Engaño (bluffing): sobredeclara potencia para ganar la subasta.

        El productor declara un 30% más de lo que tiene (bluff_factor=1.3)
        y ofrece un precio muy competitivo (20% bajo mercado) para asegurarse
        de ser seleccionado. Sin embargo, en la fase de entrega solo puede
        aportar su potencia real.

        Consecuencia para el consumidor: recibe menos energía de la pactada,
        debe cubrir el déficit comprando a la red al precio de mercado → coste extra.
        Esto es el "coste oculto del engaño" que aparece en el análisis.

        Parámetro
        ---------
        bluff_factor : float  Multiplicador de sobredeclaración (>1). Default 1.3.
        """
        return EnergyProposal(
            declared_energy_kw=real_power_kw * bluff_factor,
            price_eur_kwh=import_price * 0.80,
            real_energy_kw=real_power_kw
        )

    @staticmethod
    def hide_information(real_power_kw: float, import_price: float,
                         hide_factor: float = 0.70) -> EnergyProposal:
        """
        Ocultar información: subdeclara potencia para crear escasez artificial.

        El productor solo declara un 70% de lo que tiene (hide_factor=0.7),
        simulando que tiene menos disponibilidad de la real. Esto justifica
        cobrar un 10% por encima del mercado (precio premium por "escasez").

        Consecuencia: el consumidor paga más por menos energía.
        Si el precio ofertado supera el precio de mercado, el consumidor
        racional prefiere comprar a la red → el agente pierde la venta.
        Esto genera un trade-off interesante: precio alto vs riesgo de rechazo.

        Parámetro
        ---------
        hide_factor : float  Fracción de potencia declarada (<1). Default 0.70.
        """
        return EnergyProposal(
            declared_energy_kw=real_power_kw * hide_factor,
            price_eur_kwh=import_price * 1.10,
            real_energy_kw=real_power_kw
        )

    @classmethod
    def apply(cls, strategy_name: str, real_power_kw: float,
              import_price: float, **kwargs) -> EnergyProposal:
        """
        Punto de entrada genérico: selecciona estrategia por nombre.

        Parámetros
        ----------
        strategy_name : str   "honest" | "deception" | "hide_information"
        real_power_kw : float  Potencia real disponible [kW].
        import_price  : float  Precio de red en €/kWh.
        **kwargs               Parámetros opcionales (bluff_factor, hide_factor).

        Raises
        ------
        ValueError si strategy_name no está reconocido.
        """
        dispatch = {
            "honest":           cls.honest,
            "deception":        cls.deception,
            "hide_information": cls.hide_information,
        }
        if strategy_name not in dispatch:
            raise ValueError(
                f"Estrategia desconocida: '{strategy_name}'. "
                f"Opciones válidas: {list(dispatch.keys())}"
            )
        return dispatch[strategy_name](real_power_kw, import_price, **kwargs)
