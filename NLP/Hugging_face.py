from transformers import pipeline
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification


classifier = pipeline("sentiment-analysis")
result = classifier("the actor were very convincing.")

print(classifier(["I a from Iran.", "I am from Moon.", "I am from Iraq"]))

model_name = "huggingface/distilbert-base-uncased-finetuned-mnli"
classifier_mnli = pipeline("text-classification", model=model_name)
print(classifier_mnli("She loves me. [SEP] she loves me not."))

