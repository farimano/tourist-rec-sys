from typing import List
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
import numpy as np

@dataclass
class TextToEmbedding:
  model_name: str="avsolatorio/GIST-small-Embedding-v0"
  max_length: int=512
  batch_size: int=64

  def __post_init__(self):
    self.device = "cpu"
    self.model = SentenceTransformer(self.model_name, device=self.device)
    self.sep_token = self.model.tokenizer.sep_token
    self.pseudo_text_for_auto_padding =  f"{self.sep_token} " * self.max_length
  
  def __call__(self, texts: List[List[str]]) -> np.array:
    prep_texts = [f" {self.sep_token} ".join(text) for text in texts]
    prep_texts.append(self.pseudo_text_for_auto_padding)
    embs = self.model.encode(
        prep_texts, batch_size=self.batch_size, device=self.device, 
        convert_to_numpy=True, normalize_embeddings=True,
    )
    return embs[:-1].mean(axis=0)