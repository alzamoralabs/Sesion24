from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict
from langgraph.graph import StateGraph, END
import json

from langchain_openai import ChatOpenAI
from typing import TypedDict
from langgraph.graph import StateGraph, END
import json

# Define el estado compartido entre agentes
class MedicalState(TypedDict):
    case: str
    action: str
    eye_specialist_vote: str
    cardiac_specialist_vote: str
    eye_reasoning: str
    cardiac_reasoning: str
    final_decision: str
    messages: list

# Inicializa cliente de OpenAI
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

def eye_specialist_agent(state: MedicalState) -> MedicalState:
    """
    Agente especializado en salud ocular que evalúa acciones médicas
    """
    prompt = f"""Eres un oftalmólogo experto. Evalúa el siguiente caso médico y la acción tomada.

CASO: {state['case']}
ACCIÓN MÉDICA TOMADA: {state['action']}

Responde en JSON con este formato exacto:
{{
    "voto": "CORRECTO" o "INCORRECTO",
    "razonamiento": "Tu explicación detallada"
}}

Considera si la acción es apropiada, segura y basada en mejores prácticas médicas."""

    from langchain_core.messages import HumanMessage
    
    response = llm.invoke([HumanMessage(content=prompt)])
    response_text = response.content
    
    try:
        # Extrae JSON del contenido
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        json_str = response_text[json_start:json_end]
        response_json = json.loads(json_str)
        state["eye_specialist_vote"] = response_json.get("voto", "INDECISO")
        state["eye_reasoning"] = response_json.get("razonamiento", "")
    except (json.JSONDecodeError, ValueError):
        state["eye_specialist_vote"] = "INDECISO"
        state["eye_reasoning"] = response_text
    
    return state

def cardiac_specialist_agent(state: MedicalState) -> MedicalState:
    """
    Agente especializado en salud cardiaca que evalúa acciones médicas
    """
    prompt = f"""Eres un cardiólogo experto. Evalúa el siguiente caso médico y la acción tomada.

CASO: {state['case']}
ACCIÓN MÉDICA TOMADA: {state['action']}

Responde en JSON con este formato exacto:
{{
    "voto": "CORRECTO" o "INCORRECTO",
    "razonamiento": "Tu explicación detallada"
}}

Considera si la acción es apropiada, segura y basada en mejores prácticas médicas."""


    
    response = llm.invoke([HumanMessage(content=prompt)])
    response_text = response.content
    
    try:
        # Extrae JSON del contenido
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        json_str = response_text[json_start:json_end]
        response_json = json.loads(json_str)
        state["cardiac_specialist_vote"] = response_json.get("voto", "INCORRECTO")
        state["cardiac_reasoning"] = response_json.get("razonamiento", "")
    except (json.JSONDecodeError, ValueError):
        state["cardiac_specialist_vote"] = "INDECISO"
        state["cardiac_reasoning"] = response_text
    
    return state

def coordinator_agent(state: MedicalState) -> MedicalState:
    """
    Agente coordinador que tabula los votos y da una decisión final
    """
    eye_vote = state.get("eye_specialist_vote", "INDECISO")
    cardiac_vote = state.get("cardiac_specialist_vote", "INDECISO")
    
    # Lógica de votación: mayoría simple
    if eye_vote == cardiac_vote:
        final_decision = eye_vote
        consensus = "Consenso alcanzado"
    else:
        final_decision = "ANÁLISIS MIXTO"
        consensus = "Votos divididos - requiere revisión adicional"
    
    summary = f"""
=== RESUMEN DE VOTACIÓN ===
CASO: {state['case']}
ACCIÓN: {state['action']}

ESPECIALISTA EN SALUD OCULAR: {eye_vote}
  Razonamiento: {state.get('eye_reasoning', 'N/A')}

ESPECIALISTA EN SALUD CARDIACA: {cardiac_vote}
  Razonamiento: {state.get('cardiac_reasoning', 'N/A')}

DECISIÓN FINAL: {final_decision}
OBSERVACIÓN: {consensus}
"""
    
    state["final_decision"] = final_decision
    state["messages"].append(summary)
    
    return state

def build_medical_voting_graph():
    """
    Construye el grafo del sistema multi-agente
    """
    workflow = StateGraph(MedicalState)
    
    # Añade nodos para cada agente
    workflow.add_node("eye_specialist", eye_specialist_agent)
    workflow.add_node("cardiac_specialist", cardiac_specialist_agent)
    workflow.add_node("coordinator", coordinator_agent)
    
    # Define el flujo: los especialistas trabajan en paralelo, luego el coordinador
    workflow.set_entry_point("eye_specialist")
    
    workflow.add_edge("eye_specialist", "cardiac_specialist")
    workflow.add_edge("cardiac_specialist", "coordinator")
    workflow.add_edge("coordinator", END)
    
    return workflow.compile()

def evaluate_medical_case(case: str, action: str) -> dict:
    """
    Evalúa un caso médico con el sistema multi-agente
    """
    graph = build_medical_voting_graph()
    
    initial_state: MedicalState = {
        "case": case,
        "action": action,
        "eye_specialist_vote": "",
        "cardiac_specialist_vote": "",
        "eye_reasoning": "",
        "cardiac_reasoning": "",
        "final_decision": "",
        "messages": []
    }
    
    result = graph.invoke(initial_state)
    return result

# Ejemplo de uso
if __name__ == "__main__":
    # Casos de prueba
    test_cases = [
        {
            "case": "Paciente de 65 años con presión arterial elevada (160/100) y diabetes tipo 2",
            "action": "Se prescribió un betabloqueante y se recomendó cambios en la dieta"
        },
        {
            "case": "Paciente con migrañas frecuentes y visión borrosa en un ojo",
            "action": "Se realizó una resonancia magnética cerebral y se derivó a oftalmología"
        },
        {
            "case": "Paciente de 55 años con dolor en el pecho y antecedente de infarto",
            "action": "Se administró aspirina inmediatamente y se realizó un electrocardiograma"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"EVALUANDO CASO {i}")
        print(f"{'='*60}")
        
        result = evaluate_medical_case(
            case=test_case["case"],
            action=test_case["action"]
        )
        
        for message in result.get("messages", []):
            print(message)