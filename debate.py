from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from typing import Optional, List, Dict, Tuple

load_dotenv()

class DebateManager:
    """Gestiona un debate entre dos agentes de IA."""
    
    def __init__(self, max_rounds: int = 5, model: str = "gpt-4o"):
        """
        Inicializa el gestor de debate.
        
        Args:
            max_rounds: Número máximo de rondas
            model: Modelo a utilizar ("gpt-4o" o "llama3.2")
        """
        self.max_rounds = max_rounds
        self.model = model
        self.conversation = []
        self.consensus_reached = False
        self.consensus_text = None
        
        # Crear agentes
        self.agent_a = self._create_llm()
        self.agent_b = self._create_llm()
        
        # Prompts del sistema
        self.prompt_a = ("Eres un agente de IA que destaca fuertemente los BENEFICIOS de la IA en la atención médica humana. "
                        "Presenta argumentos sólidos y fundamentados. "
                        "Si llegas a consenso con el otro agente, comienza tu respuesta con 'ACUERDO:'")
        
        self.prompt_b = ("Eres un agente de IA que destaca fuertemente los RIESGOS de la IA en la atención médica humana. "
                        "Presenta argumentos sólidos y fundamentados. "
                        "Si llegas a consenso con el otro agente, comienza tu respuesta con 'ACUERDO:'")
    
    def _create_llm(self):
        """Crea un modelo LLM."""
        if self.model == "gpt-4o":
            return ChatOpenAI(model_name="gpt-4o", temperature=0.7)
        else:
            return ChatOllama(model="llama3.2", temperature=0.7)
    
    def _prepare_context(self, agent_name: str) -> str:
        """Prepara el contexto del debate para que cada agente sepa qué pasó."""
        if not self.conversation:
            return ""
        
        context = "\n\nHistorial reciente:\n"
        for msg in self.conversation[-4:]:
            role = msg['role']
            content = msg['content'][:150]
            context += f"- [{role}]: {content}...\n"
        
        return context
    
    def _get_agent_response(self, llm, agent_name: str, system_prompt: str) -> Optional[str]:
        """Obtiene la respuesta de un agente."""
        try:
            context = self._prepare_context(agent_name)
            user_message = f"Continúa el debate argumentando tu posición.{context}"
            
            # Construir los mensajes en el formato que espera ChatOpenAI/ChatOllama
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
            
            # Invocar el modelo
            response = llm.invoke(messages)
            content = response.content
            
            print(f"\n[{agent_name}]: {content}\n")
            return content
        
        except Exception as e:
            print(f"Error al obtener respuesta de {agent_name}: {e}")
            import traceback
            traceback.print_exc()
        
        return None
    
    def _check_for_consensus(self, content: str) -> bool:
        """Verifica si hay consenso en el mensaje."""
        return "ACUERDO:" in content.upper()
    
    def run(self, topic: str = None) -> Tuple[bool, str]:
        """
        Ejecuta el debate.
        
        Args:
            topic: Tema del debate (opcional)
            
        Returns:
            Tupla con (hay_consenso, conclusión)
        """
        if topic is None:
            topic = "Discute sobre el impacto de la inteligencia artificial en la medicina humana."
        
        print("\n" + "="*70)
        print("🤜🤛 INICIANDO DEBATE")
        print("="*70 + "\n")
        
        initial_message = {"role": "moderator", "content": topic}
        self.conversation.append(initial_message)
        print(f"[MODERADOR]: {topic}\n")
        
        # Rondas de debate
        for round_num in range(self.max_rounds):
            print(f"\n{'='*70}\nRonda {round_num + 1}/{self.max_rounds}\n{'='*70}")
            
            # Agent A responde
            response_a = self._get_agent_response(self.agent_a, "🐰AGENTE A (BENEFICIOS)", self.prompt_a)
            if response_a:
                self.conversation.append({"role": "agent_A", "content": response_a})
                
                if self._check_for_consensus(response_a):
                    self.consensus_reached = True
                    self.consensus_text = response_a
                    print("✓ AGENTE A propone CONSENSO")
                    break
            else:
                print("✗ Error en respuesta de AGENTE A 🐰")
                continue
            
            # Agent B responde
            response_b = self._get_agent_response(self.agent_b, "🦊AGENTE B (RIESGOS)", self.prompt_b)
            if response_b:
                self.conversation.append({"role": "agent_B", "content": response_b})
                
                if self._check_for_consensus(response_b):
                    self.consensus_reached = True
                    self.consensus_text = response_b
                    print("✓ AGENTE B propone CONSENSO")
                    break
            else:
                print("✗ Error en respuesta de AGENTE B 🦊")
        
        return self._generate_final_summary()
    
    def _generate_final_summary(self) -> Tuple[bool, str]:
        """Genera un resumen final del debate."""
        print("\n" + "="*70)
        print("DEBATE CONCLUIDO")
        print("="*70 + "\n")
        
        if self.consensus_reached and self.consensus_text:
            print("✓ SE ALCANZÓ CONSENSO\n")
            print(f"CONCLUSIÓN:\n{self.consensus_text}\n")
            return True, self.consensus_text
        else:
            print("✗ NO SE ALCANZÓ CONSENSO\n")
            print("ÚLTIMO ARGUMENTO DEL DEBATE:")
            if self.conversation:
                last_msg = self.conversation[-1]
                print(f"[{last_msg['role']}]: {last_msg['content']}\n")
                return False, last_msg['content']
            return False, "Sin conclusión disponible"
    
    def get_conversation_history(self) -> List[Dict]:
        """Retorna el historial completo del debate."""
        return self.conversation
    
    def print_full_debate(self):
        """Imprime el debate completo formateado."""
        print("\n" + "="*70)
        print("DEBATE COMPLETO 👌")
        print("="*70 + "\n")
        
        for msg in self.conversation:
            role = msg['role'].upper()
            content = msg['content']
            print(f"[{role}]:\n{content}\n")
            print("-" * 70 + "\n")


def main():
    """Función principal para ejecutar el debate."""
    
    # Crear gestor de debate
    debate_manager = DebateManager(
        max_rounds=5,
        model="gpt-4o"  # Cambiar a "llama3.2" si lo prefieres
    )
    
    # Ejecutar debate
    consensus, conclusion = debate_manager.run(
        topic="Discute sobre el impacto de la inteligencia artificial en la medicina humana."
    )
    
    # Mostrar resultado
    print("\n" + "="*70)
    print("RESULTADO FINAL")
    print("="*70)
    print(f"Consenso alcanzado: {'SÍ ✓' if consensus else 'NO ✗'}")
    print(f"\nConclusión final:\n{conclusion}\n")
    
    # Opcional: mostrar debate completo
    # debate_manager.print_full_debate()


if __name__ == "__main__":
    main()