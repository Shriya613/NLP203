#PROBLEM - I
# Import required libraries
import logging
from transformers import pipeline

# Set up logging
logging.basicConfig(
    filename="qa_results.log",  # Log file name
    filemode="w",               # Overwrite the file on each run
    format="%(message)s",       # Simple log message format
    level=logging.INFO          # Log all messages at INFO level
)

# Load the Question-Answering pipeline with RoBERTa
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2", device=0)

# Define in-domain "in the wild" passages
in_domain_passages = [
    {
        "context": """The Eiffel Tower is a wrought-iron lattice tower located in Paris, France. 
        It is named after the engineer Gustave Eiffel, whose company designed and built the tower. 
        It was constructed between 1887 and 1889 as the entrance arch for the 1889 World's Fair.""",
        "questions": [
            "Who designed the Eiffel Tower?",  # Correct
            "When was the Eiffel Tower constructed?",  # Correct
            "Where is the Eiffel Tower located?",  # Correct
            "What is the Eiffel Tower made of?",  # Correct
        ]
    },
    {
        "context": """Albert Einstein was a theoretical physicist, born in Germany in 1879. 
        He developed the theory of relativity, one of the two pillars of modern physics. 
        He won the Nobel Prize in Physics in 1921 for his explanation of the photoelectric effect.""",
        "questions": [
            "Who developed the theory of relativity?",  # Correct
            "When did Einstein win the Nobel Prize?",  # Correct
            "What is the photoelectric effect?",  # Unanswerable
        ]
    }
]

# Define edited passages
edited_passages = [
    {
        "context": """The Eiffel Tower is a steel structure located in Berlin, Germany. 
        It was designed by Karl Benz, a famous German engineer. The tower was initially intended 
        to be a radio broadcast antenna before being turned into a historical monument.""",
        "questions": [
            "Who designed the Eiffel Tower?",  # Incorrect
            "Where is the Eiffel Tower located?",  # Incorrect
        ]
    }
]

# Define out-of-domain passages
out_of_domain_passages = [
    {
        "context": """COVID-19 is caused by the SARS-CoV-2 virus. Vaccines like Pfizer-BioNTech and Moderna 
        were developed to prevent severe illness. The virus spreads through respiratory droplets.""",
        "questions": [
            "What causes COVID-19?",  # Correct
            "Who developed the vaccine?",  # Partial
            "How does the virus spread?",  # Correct
        ]
    },
    {
        "context": """User1: Just watched 'Interstellar', such a mind-blowing movie! 
        User2: Yeah, Christopher Nolan really knows how to direct. The visuals were incredible.""",
        "questions": [
            "Who directed Interstellar?",  # Correct
            "What movie did User1 watch?",  # Correct
            "What did User2 think of the visuals?",  # Correct
        ]
    }
]

# Helper function to run QA and log/print results
def test_qa_pipeline(passages, category):
    logging.info(f"Testing {category} Passages\n")
    print(f"Testing {category} Passages\n")
    for idx, entry in enumerate(passages):
        logging.info(f"Passage {idx+1}:\n{entry['context']}\n")
        print(f"Passage {idx+1}:\n{entry['context']}\n")
        for question in entry["questions"]:
            result = qa_pipeline(question=question, context=entry["context"])
            logging.info(f"Question: {question}")
            logging.info(f"Answer: {result['answer']} (Confidence: {result['score']:.4f})\n")
            print(f"Question: {question}")
            print(f"Answer: {result['answer']} (Confidence: {result['score']:.4f})\n")
        logging.info("-" * 80 + "\n")
        print("-" * 80 + "\n")

# Part 1: Test in-domain passages
test_qa_pipeline(in_domain_passages, "In-Domain 'In the Wild'")

# Part 2: Test edited passages
test_qa_pipeline(edited_passages, "Edited")

# Part 3: Test out-of-domain passages
test_qa_pipeline(out_of_domain_passages, "Out-of-Domain")

# Indicate where the results are saved
print("Results have been saved to 'qa_results.log'.")
