import streamlit as st
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch

# 使用更小的纯Transformer模型
@st.cache_resource
def load_mini_models():
    """加载轻量级Transformer模型"""
    
    # 图片描述 - 使用专门的image-to-text pipeline
    captioner = pipeline("image-to-text", 
                        model="nlpconnect/vit-gpt2-image-captioning")
    
    # 故事生成 - 使用纯Transformer
    story_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    story_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    
    return captioner, story_model, story_tokenizer

def generate_story_transformer(description, style, model, tokenizer, max_length=150):
    """纯Transformer生成故事"""
    
    prompt = f"Write a {style} story about: {description}\n\nStory:"
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            temperature=0.8,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    story = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return story
