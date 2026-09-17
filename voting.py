"""
Sistema multi-agente de votación médica — LangChain 1.x + LangGraph

Diversidad de modelos: cada rol del comité corre sobre un proveedor distinto
(OpenAI / Anthropic). Dos modelos de la misma familia tienden a equivocarse en
los mismos casos y a validarse entre sí, con lo cual la votación deja de aportar
información. Mezclar proveedores hace que los votos sean más independientes.

Instalación:
    pip install -U langchain langchain-openai langchain-anthropic \
                   langchain-tavily langgraph python-dotenv

Se asume un .env ya cargado con OPENAI_API_KEY, ANTHROPIC_API_KEY y
TAVILY_API_KEY. Ningún cliente recibe la key por parámetro: todos las leen del
entorno.
"""

from __future__ import annotations
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
import operator
from datetime import date
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

load_dotenv()  # idempotente: si el .env ya se cargó antes, no sobreescribe nada

HOY = date.today().strftime("%d de %B de %Y")


# ---------------------------------------------------------------------------
# 1. Estado compartido
# ---------------------------------------------------------------------------
# `log` y `fuentes` llevan un reducer (operator.add) porque dos nodos paralelos
# escriben en ellas: sin reducer LangGraph lanzaría InvalidUpdateError.
# El resto de claves las escribe un solo nodo, así que no lo necesitan.
class MedicalState(TypedDict):
    case: str
    action: str
    eye_specialist_vote: str
    eye_reasoning: str
    cardiac_specialist_vote: str
    cardiac_reasoning: str
    final_decision: str
    final_reasoning: str
    fuentes: Annotated[list[str], operator.add]
    log: Annotated[list[str], operator.add]


# ---------------------------------------------------------------------------
# 2. Esquemas de salida estructurada
# ---------------------------------------------------------------------------
class SpecialistVerdict(BaseModel):
    """Veredicto de un especialista sobre la acción médica tomada."""

    voto: Literal["CORRECTO", "INCORRECTO", "INDECISO"] = Field(
        description="CORRECTO si la acción es apropiada y segura; "
        "INCORRECTO si no lo es; INDECISO solo si falta información crítica."
    )
    razonamiento: str = Field(description="Explicación clínica detallada del voto.")
    fuentes: list[str] = Field(
        default_factory=list,
        description="URLs consultadas con la herramienta de búsqueda que respaldan el voto.",
    )


class FinalVerdict(BaseModel):
    """Decisión final del coordinador tras tabular los votos."""

    decision: Literal["CORRECTO", "INCORRECTO", "ANÁLISIS MIXTO"]
    justificacion: str = Field(
        description="Síntesis de ambos veredictos y motivo de la decisión final."
    )
    riesgos_pendientes: list[str] = Field(
        default_factory=list,
        description="Riesgos o verificaciones que quedan abiertos, si los hay.",
    )


# ---------------------------------------------------------------------------
# 3. Registro de modelos: un proveedor por rol
# ---------------------------------------------------------------------------
# Cambiar de proveedor es editar un string. init_chat_model resuelve el cliente
# correcto (ChatOpenAI, ChatAnthropic, ...) siempre que el paquete esté instalado.
MODEL_IDS = {
    "oftalmologia": "openai:gpt-4o-mini",
    "cardiologia": "anthropic:claude-sonnet-5",
    "coordinacion": "anthropic:claude-opus-5",
}

# Desde Claude 4.7 (y por tanto en Sonnet 5 y Opus 5) los parámetros de muestreo
# —temperature, top_p, top_k— están deprecados: la API devuelve 400 con solo
# incluir el campo, sin importar el valor. No basta con mandar 1.0, hay que
# omitirlo. gpt-4o-mini sí los acepta, así que la decisión es por modelo.
#
# Se usa lista de permitidos en lugar de lista de bloqueados: cada modelo nuevo
# de Anthropic hereda el comportamiento restrictivo, así que negar por defecto
# evita que el próximo salto de versión vuelva a romper el script.
ANTHROPIC_CON_SAMPLING = (
    "claude-3",
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
)


def acepta_sampling(model_id: str) -> bool:
    """True si el modelo todavía admite temperature / top_p / top_k."""
    proveedor, _, nombre = model_id.partition(":")
    if proveedor != "anthropic":
        return True
    return nombre.startswith(ANTHROPIC_CON_SAMPLING)


def construir_modelo(rol: str, temperature: float = 0.2):
    """Instancia el modelo del rol, omitiendo temperature si el modelo la rechaza."""
    model_id = MODEL_IDS[rol]
    extra = {"temperature": temperature} if acepta_sampling(model_id) else {}
    return init_chat_model(model_id, **extra)


# Nota: no habilites extended thinking en los modelos de Anthropic junto con
# `response_format=`. create_agent fuerza tool_choice para la salida
# estructurada y Anthropic rechaza esa combinación (langchain#35539).
llm_oftalmologia = construir_modelo("oftalmologia", temperature=0.2)
llm_cardiologia = construir_modelo("cardiologia", temperature=0.2)
llm_coordinacion = construir_modelo("coordinacion", temperature=0)


# ---------------------------------------------------------------------------
# 4. Herramienta de búsqueda y agentes
# ---------------------------------------------------------------------------
# Una sola instancia compartida por los tres agentes. topic="general" sirve para
# literatura clínica; usa "news" si te interesan alertas sanitarias recientes.
web_search = TavilySearch(
    max_results=4,
    topic="general",
    search_depth="advanced",
)

INSTRUCCIONES_COMUNES = f"""\
Hoy es {HOY}.

Antes de emitir tu voto, usa la herramienta de búsqueda web al menos una vez para
verificar la práctica clínica vigente (guías, contraindicaciones, alertas de
seguridad). Prioriza fuentes como guías de sociedades médicas, PubMed, Cochrane,
FDA/EMA o la OMS. Incluye en `fuentes` las URLs que realmente consultaste.

Si el caso queda fuera de tu especialidad, valora únicamente lo que sí te compete
y dilo explícitamente en tu razonamiento.
"""

eye_specialist = create_agent(
    model=llm_oftalmologia,
    tools=[web_search],
    system_prompt=(
        "Eres un oftalmólogo experto que audita decisiones clínicas.\n\n"
        + INSTRUCCIONES_COMUNES
    ),
    response_format=SpecialistVerdict,
)

cardiac_specialist = create_agent(
    model=llm_cardiologia,
    tools=[web_search],
    system_prompt=(
        "Eres un cardiólogo experto que audita decisiones clínicas.\n\n"
        + INSTRUCCIONES_COMUNES
    ),
    response_format=SpecialistVerdict,
)

coordinator = create_agent(
    model=llm_coordinacion,
    tools=[web_search],
    system_prompt=(
        "Eres el coordinador de un comité clínico. Recibes los veredictos de un "
        "oftalmólogo y un cardiólogo, emitidos por modelos distintos, y debes "
        "emitir la decisión final.\n\n"
        "Reglas:\n"
        "- Si ambos coinciden, adopta ese voto y explica por qué es consistente.\n"
        "- Si discrepan, busca en la web para desempatar y marca la decisión como "
        "ANÁLISIS MIXTO salvo que la evidencia sea concluyente hacia un lado.\n"
        "- Si alguno votó INDECISO, señala qué información falta.\n"
        "- No asumas que un veredicto es mejor por venir de un modelo más grande: "
        "pesa la evidencia citada, no la fuente.\n\n"
        + INSTRUCCIONES_COMUNES
    ),
    response_format=FinalVerdict,
)


# ---------------------------------------------------------------------------
# 5. Nodos del grafo
# ---------------------------------------------------------------------------
def _consultar_especialista(agent, case: str, action: str) -> SpecialistVerdict:
    """Ejecuta un agente especialista y devuelve su veredicto estructurado."""
    pregunta = (
        f"CASO: {case}\n"
        f"ACCIÓN MÉDICA TOMADA: {action}\n\n"
        "Evalúa si la acción fue apropiada, segura y acorde a la mejor evidencia."
    )

    result = agent.invoke({"messages": [{"role": "user", "content": pregunta}]})

    verdict = result.get("structured_response")
    if verdict is None:  # el modelo no devolvió el esquema: degradamos con gracia
        ultimo = result["messages"][-1].content
        return SpecialistVerdict(voto="INDECISO", razonamiento=str(ultimo))
    return verdict


def eye_specialist_agent(state: MedicalState) -> dict:
    """Agente especializado en salud ocular."""
    v = _consultar_especialista(eye_specialist, state["case"], state["action"])
    # Devolvemos solo las claves que este nodo modifica (actualización parcial).
    return {
        "eye_specialist_vote": v.voto,
        "eye_reasoning": v.razonamiento,
        "fuentes": v.fuentes,
    }


def cardiac_specialist_agent(state: MedicalState) -> dict:
    """Agente especializado en salud cardiaca."""
    v = _consultar_especialista(cardiac_specialist, state["case"], state["action"])
    return {
        "cardiac_specialist_vote": v.voto,
        "cardiac_reasoning": v.razonamiento,
        "fuentes": v.fuentes,
    }


def coordinator_agent(state: MedicalState) -> dict:
    """Agente coordinador: tabula los votos y emite la decisión final."""
    eye_vote = state.get("eye_specialist_vote", "INDECISO")
    cardiac_vote = state.get("cardiac_specialist_vote", "INDECISO")
    conteo = (
        "Consenso: ambos especialistas coinciden."
        if eye_vote == cardiac_vote
        else "Votos divididos: los especialistas no coinciden."
    )

    pregunta = f"""CASO: {state['case']}
ACCIÓN: {state['action']}

VEREDICTO OFTALMOLOGÍA ({MODEL_IDS['oftalmologia']}) -> {eye_vote}
Razonamiento: {state.get('eye_reasoning', 'N/A')}

VEREDICTO CARDIOLOGÍA ({MODEL_IDS['cardiologia']}) -> {cardiac_vote}
Razonamiento: {state.get('cardiac_reasoning', 'N/A')}

Tabulación automática: {conteo}

Emite la decisión final."""

    result = coordinator.invoke({"messages": [{"role": "user", "content": pregunta}]})
    final: FinalVerdict | None = result.get("structured_response")

    if final is None:
        decision = eye_vote if eye_vote == cardiac_vote else "ANÁLISIS MIXTO"
        justificacion = str(result["messages"][-1].content)
        riesgos: list[str] = []
    else:
        decision, justificacion, riesgos = (
            final.decision,
            final.justificacion,
            final.riesgos_pendientes,
        )

    urls = list(dict.fromkeys(state.get("fuentes", [])))
    bloque_fuentes = "\n".join("  - " + u for u in urls) if urls else "  (ninguna)"

    resumen = f"""
=== RESUMEN DE VOTACIÓN ===
CASO: {state['case']}
ACCIÓN: {state['action']}

ESPECIALISTA EN SALUD OCULAR [{MODEL_IDS['oftalmologia']}]: {eye_vote}
  Razonamiento: {state.get('eye_reasoning', 'N/A')}

ESPECIALISTA EN SALUD CARDIACA [{MODEL_IDS['cardiologia']}]: {cardiac_vote}
  Razonamiento: {state.get('cardiac_reasoning', 'N/A')}

DECISIÓN FINAL [{MODEL_IDS['coordinacion']}]: {decision}
OBSERVACIÓN: {conteo}
JUSTIFICACIÓN: {justificacion}
RIESGOS PENDIENTES: {', '.join(riesgos) if riesgos else 'ninguno señalado'}

FUENTES CONSULTADAS:
{bloque_fuentes}
"""

    return {
        "final_decision": decision,
        "final_reasoning": justificacion,
        "log": [resumen],
    }


# ---------------------------------------------------------------------------
# 6. Construcción del grafo
# ---------------------------------------------------------------------------
def build_medical_voting_graph():
    """Construye el grafo: los dos especialistas en paralelo, luego el coordinador."""
    workflow = StateGraph(MedicalState)

    workflow.add_node("eye_specialist", eye_specialist_agent)
    workflow.add_node("cardiac_specialist", cardiac_specialist_agent)
    workflow.add_node("coordinator", coordinator_agent)

    # Fan-out: dos aristas desde START -> ejecución paralela real.
    # Con proveedores distintos el paralelismo también reparte el rate limit.
    workflow.add_edge(START, "eye_specialist")
    workflow.add_edge(START, "cardiac_specialist")

    # Fan-in: el coordinador espera a que ambas ramas terminen.
    workflow.add_edge("eye_specialist", "coordinator")
    workflow.add_edge("cardiac_specialist", "coordinator")
    workflow.add_edge("coordinator", END)

    return workflow.compile()


GRAPH = build_medical_voting_graph()


def evaluate_medical_case(case: str, action: str) -> MedicalState:
    """Evalúa un caso médico con el sistema multi-agente."""
    initial_state: MedicalState = {
        "case": case,
        "action": action,
        "eye_specialist_vote": "",
        "eye_reasoning": "",
        "cardiac_specialist_vote": "",
        "cardiac_reasoning": "",
        "final_decision": "",
        "final_reasoning": "",
        "fuentes": [],
        "log": [],
    }
    return GRAPH.invoke(initial_state)


# ---------------------------------------------------------------------------
# 7. Ejemplo de uso
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_cases = [
        {
            "case": "Paciente de 65 años con presión arterial elevada (160/100) y diabetes tipo 2",
            "action": "Se prescribió un betabloqueante y se recomendó cambios en la dieta",
        },
        {
            "case": "Paciente con migrañas frecuentes y visión borrosa en un ojo",
            "action": "Se realizó una resonancia magnética cerebral y se derivó a oftalmología",
        },
        {
            "case": "Paciente de 55 años con dolor en el pecho y antecedente de infarto",
            "action": "Se administró aspirina inmediatamente y se realizó un electrocardiograma",
        },
    ]

    print("COMITÉ:")
    for rol, model_id in MODEL_IDS.items():
        print(f"  {rol:<14} -> {model_id}")

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'=' * 60}")
        print(f"EVALUANDO CASO {i}")
        print(f"{'=' * 60}")

        result = evaluate_medical_case(
            case=test_case["case"],
            action=test_case["action"],
        )

        for entrada in result.get("log", []):
            print(entrada)