# 5 Agentes con Personalidades Únicas:

# Coordinador: Facilita la discusión y sintetiza ideas
# Creative: Propone ideas disruptivas e innovadoras
# Analyst: Valida ideas con datos y pensamiento crítico
# Brand Expert: Asegura coherencia de marca
# Market Specialist: Experto en jóvenes viajeros

# Flujo de Conversación:
# El coordinador inicia, luego los 4 agentes hablan secuencialmente, con el coordinador sintetizando cada ronda.
# El chat termina cuando hay consenso o se alcanzan 8 turnos.
# Estructura de Estado:
# Utiliza ChatState para mantener historial, contar turnos y rastrear cuál es el siguiente agente. El enrutador (router_node)
# dirige la conversación de forma inteligente. """

import os
import json
from typing import Annotated, Any, Literal
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import StreamWriter
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from dotenv import load_dotenv
load_dotenv()

# Configurar modelo
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# Definir estado del grafo
class ChatState(TypedDict):
    messages: list
    next_agent: str
    chat_history: list
    final_plan: str
    turn_count: int
    max_turns: int

# Prompts para cada agente
SYSTEM_PROMPTS = {
    "coordinator": """Eres un coordinador de chat grupal experto. Tu rol es:
        1. Facilitar la discusión entre los agentes
        2. Asegurar que todos participen
        3. Sintetizar ideas y dirigir hacia consenso
        4. Cuando detectes que se ha llegado a buen consenso, declara que el plan está listo
        5. Sé breve y enfocado (máximo 3 párrafos)""",
            
    "creative": """Eres un especialista creativo e innovador. Tu personalidad:
        - Piensas fuera de la caja y propones ideas disruptivas
        - Te entusiasma explorar nuevas tendencias en redes sociales y viralidad
        - Eres optimista y ves oportunidades donde otros ven obstáculos
        - Hablas con pasión sobre experiencias de usuario innovadoras
        Contribuye ideas creativas para el marketing bancario dirigido a jóvenes viajeros.""",
            
    "analyst": """Eres un analista de datos riguroso y crítico. Tu personalidad:
        - Buscas evidencia y datos antes de comprometerte con ideas
        - Eres escéptico pero constructivo
        - Te interesa entender mercados, comportamientos y ROI
        - Haces preguntas difíciles para validar ideas
        Analiza críticamente el plan de marketing con base en datos de mercado.""",
            
    "brand_expert": """Eres un experto en branding y posicionamiento. Tu personalidad:
        - Te importa mucho la identidad y valores de la marca
        - Piensas en coherencia, diferenciación y positioning
        - Eres diplomático pero firme en temas de marca
        - Te interesa crear conexiones emocionales auténticas
        Asegúrate de que el plan fortalezca la marca bancaria entre jóvenes.""",
            
    "market_specialist": """Eres un especialista en comportamiento del segmento joven viajero. Tu personalidad:
        - Conoces profundamente a millennials y Gen Z
        - Entiendes sus valores: experiencias, autenticidad, sostenibilidad
        - Hablas su idioma cultural y digital
        - Eres empático y conectas con sus motivaciones reales
        Aporta perspectivas sobre qué realmente motiva a los jóvenes viajeros."""
    }

def create_agent_node(agent_name: str):
    """Factory para crear nodos de agentes"""
    def agent_node(state: dict) -> dict:
        system_prompt = SYSTEM_PROMPTS[agent_name]
        
        # Contexto del chat
        chat_history = state.get("chat_history", [])
        chat_context = "\n".join([
            f"{msg.get('role', 'unknown').upper()}: {msg.get('content', '')}"
            for msg in chat_history[-6:]  # Últimos 3 intercambios
        ])
        
        turn_count = state.get("turn_count", 0)
        
        # Construcción del prompt
        if agent_name == "coordinator":
            user_prompt = f"""CHAT ACTUAL:
                {chat_context}

                TURNO #{turn_count}

                Como coordinador:
                1. Resume brevemente el progreso actual
                2. Guía la discusión hacia el siguiente aspecto importante
                3. Si el plan está completo y hay consenso, declara: "PLAN LISTO Y CONSENSUADO"
                4. De lo contrario, sugiere qué aspecto abordar a continuación"""
        else:
            user_prompt = f"""CHAT ACTUAL:
                {chat_context}

                TURNO #{turn_count}

                Como {agent_name}:
                1. Proporciona tu perspectiva única
                2. Construye sobre las ideas previas o desafíalas constructivamente
                3. Sugiere acciones concretas si es relevante
                4. Sé conciso pero sustancial (2-3 párrafos)"""
        
        # Llamar al modelo
        response = llm.invoke([
            {"type": "system", "content": system_prompt},
            {"type": "user", "content": user_prompt}
        ])
        
        agent_response = response.content
        
        # Actualizar historial
        new_history = chat_history + [
            {"role": agent_name, "content": agent_response}
        ]
        
        # Determinar siguiente agente
        if agent_name == "coordinator" and "PLAN LISTO Y CONSENSUADO" in agent_response:
            next_agent_name = "END"
        else:
            agents_cycle = ["creative", "analyst", "brand_expert", "market_specialist"]
            current_index = agents_cycle.index(agent_name) if agent_name in agents_cycle else -1
            next_agent_name = agents_cycle[(current_index + 1) % len(agents_cycle)]
        
        return {
            "messages": state.get("messages", []) + [AIMessage(content=agent_response)],
            "chat_history": new_history,
            "next_agent": next_agent_name,
            "turn_count": turn_count + 1,
            "max_turns": state.get("max_turns", 8),
            "final_plan": agent_response if next_agent_name == "END" else state.get("final_plan", "")
        }
    
    return agent_node

def router_node(state: dict) -> Literal["creative", "analyst", "brand_expert", "market_specialist", "coordinator", END]:
    """Router que dirige al siguiente agente"""
    turn_count = state.get("turn_count", 0)
    max_turns = state.get("max_turns", 8)
    next_agent = state.get("next_agent", "END")
    
    if turn_count >= max_turns:
        return END
    
    if next_agent == "END":
        return END
    
    return next_agent

# Construir el grafo
graph_builder = StateGraph(ChatState)

# Agregar nodos de agentes
graph_builder.add_node("creative", create_agent_node("creative"))
graph_builder.add_node("analyst", create_agent_node("analyst"))
graph_builder.add_node("brand_expert", create_agent_node("brand_expert"))
graph_builder.add_node("market_specialist", create_agent_node("market_specialist"))
graph_builder.add_node("coordinator", create_agent_node("coordinator"))

# Conexiones del grafo
graph_builder.add_edge(START, "coordinator")
graph_builder.add_conditional_edges("coordinator", router_node)
graph_builder.add_conditional_edges("creative", router_node)
graph_builder.add_conditional_edges("analyst", router_node)
graph_builder.add_conditional_edges("brand_expert", router_node)
graph_builder.add_conditional_edges("market_specialist", router_node)

# Compilar grafo
graph = graph_builder.compile()

# Ejecutar el sistema multiagente
def run_marketing_chat():
    """Ejecutar la sesión de chat colaborativo"""
    
    initial_prompt = """OBJETIVO: Diseñar un plan de marketing para un banco ficticio dirigido a jóvenes viajeros (18-35 años).

CONTEXTO: 
- Los clientes objetivo son millennials y Gen Z que viajan frecuentemente
- Valoran experiencias, autenticidad y tecnología
- Buscan productos bancarios simplificados y digitales
- Les importa la sostenibilidad y responsabilidad social

TAREAS A ABORDAR:
1. Canales de comunicación principales
2. Propuesta de valor única
3. Productos/servicios específicos
4. Estrategia de contenido y activaciones
5. Métricas de éxito

Colaboren para crear un plan integral y consensuado."""

    initial_state = {
        "messages": [HumanMessage(content=initial_prompt)],
        "chat_history": [{"role": "system", "content": initial_prompt}],
        "next_agent": "coordinator",
        "turn_count": 0,
        "max_turns": 8,
        "final_plan": ""
    }
    
    print("=" * 80)
    print("SISTEMA MULTIAGENTE: CHAT COLABORATIVO DE MARKETING")
    print("=" * 80)
    print(f"\n📋 OBJETIVO:\n{initial_prompt}\n")
    print("=" * 80)
    print("Iniciando sesión de chat colaborativo...\n")
    
    # Ejecutar el grafo
    for output in graph.stream(initial_state, stream_mode="updates"):
        for node, state in output.items():
            if node != "__start__":
                agent_name = node
                response = state["messages"][-1].content if state["messages"] else ""
                
                # Mostrar respuesta formateada
                print(f"\n🤖 {agent_name.upper().replace('_', ' ')}")
                print("-" * 40)
                print(response)
                print()
    
    return state

if __name__ == "__main__":
    final_state = run_marketing_chat()
    
    print("\n" + "=" * 80)
    print("SESIÓN COMPLETADA")
    print("=" * 80)
    print(f"Total de turnos: {final_state.get('turn_count', 0)}")
    print("\n📝 RESUMEN DEL CHAT COLABORATIVO:")
    print("-" * 80)
    
    chat_history = final_state.get('chat_history', [])
    for msg in chat_history:
        print(f"\n[{msg['role'].upper()}]")
        print(msg['content'][:500] + "..." if len(msg['content']) > 500 else msg['content'])