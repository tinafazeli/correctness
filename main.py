from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline

model_name = "./model"
tokenizer_name = "./tokenizer"
lora_model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)


nlp_pipeline = pipeline("text-classification", model=lora_model, tokenizer=tokenizer)

def is_correct(word):
    prediction = nlp_pipeline(word)
    if prediction[0]['label'] == 'LABEL_1':
        return True
    else:
        return False

word = "مارمولک"
is_correct = is_correct(word)
print(f"آیا '{word}' کلمه صحیحی است؟ {'1' if is_correct else '0'}")
