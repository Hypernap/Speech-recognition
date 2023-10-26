from transformers import AutoModelForCausalLM, AutoTokenizer , AutoModelForSeq2SeqLM
import torch

text = """
Quantum computers are a new frontier in computing that operates on the principles of quantum mechanics. Unlike classical computers, which use bits (0 or 1), quantum computers utilize quantum bits or qubits. These qubits can exist in multiple states at the same time through a phenomenon called superposition, allowing quantum computers to perform many calculations simultaneously. Furthermore, qubits can become entangled, where the state of one qubit depends on the state of another, even when they are physically separated. This property enables highly correlated and coordinated operations, making quantum computers potentially much more powerful for specific tasks. Quantum computing has the potential to revolutionize fields like cryptography, materials science, optimization, and artificial intelligence, but it also presents significant technical challenges, particularly in building and maintaining stable qubits and mitigating errors.The development of quantum computers is ongoing, with various hardware platforms such as superconducting qubits and trapped ions being explored. Achieving quantum supremacy, where quantum computers outperform classical computers for practical tasks, remains a key milestone. As the technology matures, we can anticipate significant advancements in various fields and the resolution of some of the challenges that currently limit the widespread adoption of quantum computing, making it an exciting and transformative area of research and development with vast potential.
"""

# Load the pre-trained T5 model and tokenizer for summarization
model_name = "t5-base"  # Use a larger T5 model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Tokenize the text
inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)

# Generate an abstractive summary
summary_ids = model.generate(inputs, max_length=200, num_beams=4, early_stopping=True)

# Decode and display the summary
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Simple post-processing to improve coherence
summary = summary.replace(" .", ".").replace(" ,", ",")

print(summary)
