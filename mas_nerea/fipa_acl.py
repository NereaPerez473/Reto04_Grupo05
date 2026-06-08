"""
fipa_acl.py
===========
Utilidades de mensajería FIPA-ACL.

Serializa y deserializa mensajes en JSON, igual que en los ejemplos de clase
pero con soporte para content de tipo dict (además de str).

Performativas usadas en este proyecto:
  cfp              → AC invita a AS/AE a presentar propuestas
  propose          → AS/AE responde con oferta de energía
  accept-proposal  → AC acepta una oferta (incluye cantidad y precio)
  reject-proposal  → AC rechaza una oferta
  inform           → AS/AE confirma entrega real tras aceptación
"""

import json


def create_message(performative: str, sender: str, receiver: str,
                   content) -> str:
    """
    Crea un mensaje FIPA-ACL serializado en JSON.

    Parámetros
    ----------
    performative : str    Tipo de acto comunicativo (cfp, propose, etc.)
    sender       : str    Nombre del agente emisor.
    receiver     : str    Nombre del agente receptor.
    content      : str|dict  Contenido del mensaje.

    Returns
    -------
    str : JSON listo para enviar por socket.
    """
    return json.dumps({
        "performative": performative,
        "sender": sender,
        "receiver": receiver,
        "content": content
    })


def parse_message(message_str: str) -> dict:
    """
    Deserializa un mensaje FIPA-ACL recibido por socket.

    Returns
    -------
    dict con claves: performative, sender, receiver, content.
    """
    return json.loads(message_str)
