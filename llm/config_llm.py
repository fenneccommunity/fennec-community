from dataclasses import dataclass

@dataclass
class llm_config:
      max_token:int = 2048
      temperature : float = 0.3
      top_p : float = 0.9
      top_k : int = 50

      # hugginface model
      hugginface_model : str = "EleutherAI/gpt-neo-125M"

      # gemini model
      gemini_model : str = "gemini-3-flash-preview"

      # mistral model
      mistral_model : str = "mistral-large-latest"

      # groq model
      groq_model : str = "llama-3.3-70b-versatile"

      # ollama model
      ollama_model : str = "llama2"
      ollama_base_url : str = "http://127.0.0.1:11434"
      time_out : int = 200