from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from dotenv import load_dotenv
from utils import format_messages, format_message_content
load_dotenv()

AGREED = False

agent_a = create_agent(
    model=ChatOpenAI(model_name="gpt-4o", temperature=0.5),
    #model=ChatOllama(model="llama3.2", temperature=0.5),
    system_prompt="Eres un agente de IA que destaca fuertemente los BENEFICIOS de la IA en la atención médica humana."
)
agent_b = create_agent(
    model=ChatOpenAI(model_name="gpt-4o", temperature=0.5),
    #model=ChatOllama(model="llama3.2", temperature=0.5),
    system_prompt="Eres un agente de IA que destaca fuertemente los RIESGOS de la IA en la atención médica humana."
)
conversation = []  # shared conversation log (list of messages)
# Initial prompt to start the debate
initial_prompt = "Discute sobre el impacto de la inteligencia artificial en la medicina humana. " \
"Si llegas a concenso con otro agente o con el usuario, indica lo propio al escribir en el chat 'AGREED: [tu conclusión]'. "
conversation.append({"role": "user", "content": initial_prompt})

# Debate rounds
MAX_ROUNDS = 5
for round in range(MAX_ROUNDS):
    # Agent A responds to Agent B
    response_a = agent_a.invoke(conversation[-1])["messages"]
    print(format_messages(response_a))
    conversation.append({"role": "agent_A", "content": response_a[-1].content})
    # Check for consensus or termination keyword
    if "AGREED" in response_a[-1].content.upper():
        AGREED = True
        break  # Agent A signaled agreement (just an example condition)
    # Agent B responds to Agent A
    response_b = agent_b.invoke(conversation[-1])["messages"]
    print(format_messages(response_b))
    conversation.append({"role": "agent_B", "content": response_b[-1].content})
    print(format_messages(response_b))
    if "AGREED" in response_b[-1].content.upper():
        AGREED = True
        break

print("\n========================= DEBATE CONCLUIDO =========================\n")

# After debate, decide final answer:
final_answer = None
if "AGREED" in response_a[-1].content.upper() or "AGREED" in response_b[-1].content.upper():
    # If any agent explicitly agreed, use that as consensus (assuming they state it)
    print("AGREED. EL CONCENSO ES:")
    final_answer = conversation[-1]["content"]
else:
    # Otherwise, for simplicity, take the last statement by Agent B as the final answer
    print("NO HAY CONCENSO. ULTIMA DECLARACIÓN FUE:")
    final_answer = conversation[-1]["content"]

print("FINAL ANSWER: ", final_answer)