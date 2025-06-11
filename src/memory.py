from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=False, k=5)


def get_chat_history() -> str:
    return memory.load_memory_variables({})["chat_history"]


def save_memory(user_input: str, assistant_response: str)->None:
    memory.save_context({"input": user_input}, {"output": assistant_response})

def clear_memory() -> None:
    memory.clear()
    print("Mémoire effacée.")