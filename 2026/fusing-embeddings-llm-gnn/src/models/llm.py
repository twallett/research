import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, PreTrainedTokenizerBase
import torch
from sentence_transformers import SentenceTransformer
# %%-------------------------------------------------------------------------------------------------

CONFIG_FILENAME = "config.json"
# meta-llama/Llama-3.2-3B-Instruct
# mistralai/Mistral-7B-Instruct-v0.2
@dataclass
class LLMConfig:
    model_type: str
    model_name: str
    embedding_model_name: str
    device: str
    dtype: str
    max_new_tokens: int
    temperature: float
    default_prompt: str
    system_prompts: Dict[str, str]
    hf_token: Optional[str] = None

    @staticmethod
    def load(config_path: Path) -> "LLMConfig":
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config file: {config_path}")
        data = json.loads(config_path.read_text(encoding="utf-8"))
        required = {"model_type","model_name","embedding_model_name","device","dtype","max_new_tokens","temperature","default_prompt","system_prompts"}
        missing = [k for k in required if k not in data]
        return LLMConfig(
            model_type=str(data["model_type"]),
            model_name=str(data["model_name"]),
            embedding_model_name=str(data["embedding_model_name"]),
            device=str(data["device"]),
            dtype=str(data["dtype"]),
            max_new_tokens=int(data["max_new_tokens"]),
            temperature=float(data["temperature"]),
            default_prompt=str(data["default_prompt"]),
            system_prompts=dict(data["system_prompts"]),
            hf_token=data.get("hf_token"),
        )

    def save(self, config_path: Path) -> None:
        data = asdict(self)
        Path(config_path).write_text(json.dumps(data, indent=2), encoding="utf-8")

# %%-------------------------------------------------------------------------------------------------

class LLM:
    def __init__(self, config_path: Optional[str] = None):
        module_dir = Path(__file__).resolve().parent
        self.config_path = Path(config_path) if config_path else module_dir / CONFIG_FILENAME
        self.config = LLMConfig.load(self.config_path)
        self._tok: Optional[PreTrainedTokenizerBase] = None
        self._gen_model = None
        self._embed_model = None
        self._embed_backend: str = "auto"

    def _match_system_prompt(self) -> str:
        name = self.config.model_name.lower()
        sys_prompts = self.config.system_prompts or {}
        if "llama" in name:
            return sys_prompts.get("llama", sys_prompts.get("default", ""))
        if "mistral" in name:
            return sys_prompts.get("mistral", sys_prompts.get("default", ""))
        return sys_prompts.get("default", "")

    def _resolve_dtype(self):
        if torch is None:
            return None
        if self.config.dtype == "float16":
            return torch.float16
        if self.config.dtype == "bfloat16":
            return torch.bfloat16
        return torch.float32

    def _load_generator(self):
        if self._gen_model is not None:
            return
        self._tok = AutoTokenizer.from_pretrained(self.config.model_name, token=self.config.hf_token)
        self._gen_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=self._resolve_dtype(),
            device_map=None,
            token=self.config.hf_token,
        )
        if torch is not None and self.config.device:
            device = self.config.device
            self._gen_model.to(device)

    def _load_embedder(self):
        if self._embed_model is not None:
            return
        if SentenceTransformer is not None:
            self._embed_model = SentenceTransformer(self.config.embedding_model_name, device=self.config.device)
            self._embed_backend = "sentence-transformers"
            return
        if AutoTokenizer is None or AutoModel is None:
            raise RuntimeError("transformers is required for embeddings but is not installed.")
        self._embed_tok = AutoTokenizer.from_pretrained(self.config.embedding_model_name, token=self.config.hf_token)
        self._embed_model = AutoModel.from_pretrained(
            self.config.embedding_model_name,
            torch_dtype=self._resolve_dtype(),
            token=self.config.hf_token,
        )
        if torch is not None and self.config.device:
            self._embed_model.to(self.config.device)
        self._embed_backend = "hf"

    def get_embedding(self, text: str) -> List[float]:
        if not isinstance(text, str) or len(text.strip()) == 0:
            raise ValueError("text must be a non-empty string")
        self._load_embedder()
        if self._embed_backend == "sentence-transformers":
            vec = self._embed_model.encode([text], normalize_embeddings=True)[0]
            return [float(x) for x in vec.tolist()] if hasattr(vec, "tolist") else [float(x) for x in vec]
        tok = self._embed_tok(text, return_tensors="pt", truncation=True)
        if torch is not None and self.config.device:
            tok = {k: v.to(self.config.device) for k, v in tok.items()}
        outputs = self._embed_model(**tok)
        hidden = outputs.last_hidden_state
        attn_mask = tok.get("attention_mask")
        if attn_mask is not None:
            mask = attn_mask.unsqueeze(-1).float()
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            pooled = summed / counts
        else:
            pooled = hidden.mean(dim=1)
        vec = pooled[0].detach().cpu().numpy()
        return [float(x) for x in vec.tolist()]

    def generate_answer(
        self,
        query: str,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        self._load_generator()
        tok = self._tok
        assert tok is not None
        sys_p = system_prompt if system_prompt is not None else self._match_system_prompt()
        usr_p = user_prompt if user_prompt is not None else self.config.default_prompt
        messages = []
        if sys_p:
            messages.append({"role": "system", "content": sys_p})
        if usr_p:
            messages.append({"role": "user", "content": f"{usr_p}\n\nQuestion: {query}"})
        else:
            messages.append({"role": "user", "content": query})
        use_chat = getattr(tok, "chat_template", None)
        if use_chat:
            prompt_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            if sys_p:
                prompt_text = f"<|system|>\n{sys_p}\n</s>\n<|user|>\n{usr_p}\n\nQuestion: {query}\n</s>\n<|assistant|>"
            else:
                prompt_text = f"{usr_p}\n\nQuestion: {query}\nAnswer:"
        inputs = tok(prompt_text, return_tensors="pt")
        if torch is not None and self.config.device:
            inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        resolved_max_new = int(max_new_tokens or self.config.max_new_tokens)
        resolved_temp = float(temperature) if temperature is not None else float(self.config.temperature)
        if resolved_temp <= 0.0:
            gen_kwargs = {
                "max_new_tokens": resolved_max_new,
                "do_sample": False,
                "eos_token_id": tok.eos_token_id,
            }
        else:
            gen_kwargs = {
                "max_new_tokens": resolved_max_new,
                "temperature": resolved_temp,
                "do_sample": True,
                "eos_token_id": tok.eos_token_id,
            }
        with torch.no_grad() if torch is not None else _nullcontext():
            outputs = self._gen_model.generate(**inputs, **gen_kwargs)
        text = tok.decode(outputs[0], skip_special_tokens=True)
        if prompt_text in text:
            return text.split(prompt_text, 1)[-1].strip()
        return text.strip()

# %%-------------------------------------------------------------------------------------------------

class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False
