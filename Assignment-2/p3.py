import json
import os
from transformers import pipeline

# Load Covid-QA dataset
def load_dataset(filename):
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["data"]

# Load dev and test datasets
dev_data = load_dataset("covid-qa/covid-qa-dev.json")
test_data = load_dataset("covid-qa/covid-qa-test.json")

# Initialize the RoBERTa QA model pipeline
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Function to extract question-answer pairs
def extract_qa_pairs(dataset):
    question_pairs = []
    for article in dataset:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                question_id = qa["id"]
                question = qa["question"]
                answers = [ans["text"] for ans in qa["answers"]]
                question_pairs.append((question_id, context, question, answers))
    return question_pairs

# Extract question pairs from dev and test datasets
dev_question_pairs = extract_qa_pairs(dev_data)
test_question_pairs = extract_qa_pairs(test_data)

# Function to generate predictions
def generate_predictions(question_pairs):
    predictions = {}
    for question_id, context, question, _ in question_pairs:
        result = qa_pipeline(question=question, context=context)
        predictions[question_id] = result["answer"]
    return predictions

# Generate predictions for test dataset
test_predictions = generate_predictions(test_question_pairs)

# Save predictions
with open("predictions.json", "w", encoding="utf-8") as f:
    json.dump(test_predictions, f, indent=4)

# Evaluate using the provided evaluation script
os.system("python evaluate_local.py covid-qa-test.json predictions.json --out-file test_eval_results.json")

# Load evaluation results
with open("test_eval_results.json", "r", encoding="utf-8") as f:
    eval_metrics = json.load(f)

# Extract scores
em_score = eval_metrics["exact"]
f1_score = eval_metrics["f1"]

# Print evaluation scores
print(f"Exact Match (EM): {em_score:.2f}")
print(f"F1 Score: {f1_score:.2f}")

# Log evaluation results to a log file
with open("evaluation_log.txt", "a", encoding="utf-8") as log_file:
    log_file.write("=== Model Evaluation Results ===\n")
    log_file.write(f"Exact Match (EM): {em_score:.2f}\n")
    log_file.write(f"F1 Score: {f1_score:.2f}\n")
    log_file.write("=" * 30 + "\n")

print("Evaluation results saved to evaluation_log.txt ")
