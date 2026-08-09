import numpy as np
from typing import Any

def explain_classical(text: str, vectorizer: Any, classifier: Any, prediction_label: int, top_k: int = 5) -> list[str]:
                                                                                               
    if not hasattr(classifier, "coef_"):
        return []
        
    features = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()
    coefs = classifier.coef_[0]
    
    non_zero_indices = features.nonzero()[1]
    if len(non_zero_indices) == 0:
        return []
        
    text_coefs = coefs[non_zero_indices]
    
    if prediction_label == 1:

        sorted_indices = non_zero_indices[np.argsort(text_coefs)[::-1]]
    else:

        sorted_indices = non_zero_indices[np.argsort(text_coefs)]
        
    top_indices = sorted_indices[:top_k]
    return [feature_names[idx] for idx in top_indices]

def _compute_saliency(model: Any, inputs: dict[str, Any], embedding_layer: Any, prediction_label: int, tokens: list[str], top_k: int) -> list[str]:
                                                      
    import torch
    
    embeddings_list = []
    
    def hook(module, input, output):
        output.retain_grad()
        embeddings_list.append(output)
        
    handle = embedding_layer.register_forward_hook(hook)
    
    model.zero_grad()
    
    if "token_ids" in inputs:

        logits = model(inputs["token_ids"])
    else:

        outputs = model(**inputs)
        logits = outputs.logits
        
    score = logits[0, prediction_label]
    score.backward()
    handle.remove()
    if not embeddings_list or embeddings_list[0].grad is None:
        return []
        
    gradients = embeddings_list[0].grad[0]
    saliency = torch.norm(gradients, dim=1).cpu().numpy()
    
    seq_len = min(len(tokens), len(saliency))
    if seq_len == 0:
        return []
        
    top_indices = np.argsort(saliency[:seq_len])[::-1]
    
    flagged = []
    seen = set()
    for idx in top_indices:
        word = tokens[idx]
        if word.startswith("##"):
            word = word[2:]
        if word.lower() not in seen and len(word) > 2 and word not in ["[CLS]", "[SEP]", "<pad>", "<unk>"]:
            seen.add(word.lower())
            flagged.append(word)
        if len(flagged) >= top_k:
            break
            
    return flagged

def explain_deep_learning(text: str, vocabulary: Any, model: Any, prediction_label: int, top_k: int = 5, max_length: int = 300) -> list[str]:
                                                                                                 
    import torch
    from src.data.preprocess import preprocess_text
    
    model.eval()
    cleaned = preprocess_text(text)
    
    tokens = cleaned.split()
    token_ids = vocabulary.encode(cleaned, max_length=max_length)
    if not token_ids:
        return []
        
    inputs = {"token_ids": torch.tensor([token_ids], dtype=torch.long)}
    return _compute_saliency(model, inputs, model.embedding, prediction_label, tokens, top_k)

def explain_transformer(text: str, tokenizer: Any, model: Any, prediction_label: int, top_k: int = 5, max_length: int = 256) -> list[str]:
                                                                                                            
    import torch
    from src.data.preprocess import preprocess_text
    
    model.eval()
    cleaned = preprocess_text(text)
    
    inputs = tokenizer(
        cleaned,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    
    token_ids = inputs["input_ids"][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    
    return _compute_saliency(model, inputs, model.get_input_embeddings(), prediction_label, tokens, top_k)

def explain_prediction(
    text: str,
    tier: str,
    state: dict[str, Any],
    prediction_label: int,
    top_k: int = 5
) -> list[str]:
                                                                      
    if tier == "transformer":
        return explain_transformer(
            text=text,
            tokenizer=state["tokenizer"],
            model=state["model"],
            prediction_label=prediction_label,
            top_k=top_k,
            max_length=state["config"].get("training", {}).get("max_sequence_length", 256),
        )
    elif tier == "deep_learning":
        return explain_deep_learning(
            text=text,
            vocabulary=state["vocabulary"],
            model=state["model"],
            prediction_label=prediction_label,
            top_k=top_k,
            max_length=state["config"].get("vocabulary", {}).get("max_sequence_length", 300),
        )
    else:
        return explain_classical(
            text=text,
            vectorizer=state["vectorizer"],
            classifier=state["classifier"],
            prediction_label=prediction_label,
            top_k=top_k,
        )
