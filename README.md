# Alpaca-Neo Fine-Tuning
### A QLoRA-Optimized Instruction-Following LLM

---

# Goal

The objective of this project was to transform a **general-purpose language model** (**GPT-Neo 1.3B**) into a specialized **instruction-following assistant**.

Using the **Alpaca dataset**, the project demonstrates how **low-compute optimization techniques** can enable effective LLM fine-tuning on **consumer-grade hardware** such as a **Google Colab T4 GPU**.

---

# Models & Data

| Component | Details |
|-----------|---------|
| **Base Model** | `EleutherAI/gpt-neo-1.3B` |
| **Fine-Tuning Dataset** | `tatsu-lab/alpaca` (52k instruction demonstrations) |
| **Architecture** | Transformer-based Causal Language Model |

---

# Method: PEFT & QLoRA

To make training efficient and accessible, I used **Parameter-Efficient Fine-Tuning (PEFT)** combined with several optimization strategies.

### QLoRA (4-bit Quantization)

- Model weights compressed to **4-bit precision** using **bitsandbytes**
- Reduced VRAM usage by **~70%**
- Maintained strong model performance

### LoRA (Low-Rank Adaptation)

Instead of updating **all 1.3B parameters**, training focuses on lightweight **adapter layers**.

Configuration:

- **Rank:** `16`
- **Target modules:** `q_proj`, `v_proj`

This dramatically reduces training cost while maintaining performance.

### Gradient Checkpointing

Enabled to reduce memory usage during the **backward pass**, allowing the model to fit within **Colab GPU limits**.

---

# Process & Training

Training was conducted in a **Python 3.12 environment** with an emphasis on stability and compute efficiency.

### Data Engineering

The Alpaca dataset was reformatted into a structured prompt template:

```
Instruction
Input
Response
```

Example prompt format:

```
### Instruction:
Explain how a microwave works.

### Response:
```

---

### Persistent Storage

Integrated **Google Drive checkpointing** to ensure training progress was not lost due to:

- Colab session resets
- GPU quota interruptions

---

### Optimization Strategies

Several techniques were used to improve training stability:

- **Early Stopping Callback**
  - Monitors validation loss
  - Prevents overfitting
  - Reduces wasted compute

- **Mixed Precision Training**
  - `fp16` enabled

- **Batch Size Optimization**
  - Tuned to prevent **Out-Of-Memory (OOM)** errors

---

# Results: Before vs After

Fine-tuning successfully **re-wired the model** from generic text continuation to **instruction-following behavior**.

| Instruction | Base GPT-Neo 1.3B | Fine-Tuned (Alpaca-Neo) |
|-------------|------------------|--------------------------|
| Explain a microwave to a 5-year-old | `[News Fragment] Dale Gaskins, a fifth-grader at St. John’s Elementary...` | "A microwave is a device that heats food using special waves called microwaves..." |
| 3 Healthy breakfast ideas | `[Blog Rant] You may have heard of the ‘Diet for Diabetics’...` | 1. Fruit smoothie  2. Avocado toast  3. Eggs with whole-grain toast |
| Formal email for a day off | `[Meta commentary] This is the kind of request that has been coming in...` | "Dear Sir or Madam, I am writing to request a day off from work..." |

The fine-tuned model demonstrates **clear instruction adherence and structured responses**.

---

# Limitations

### Model Size

As a **1.3B parameter model**, GPT-Neo may occasionally produce:

- repetitive phrasing
- circular reasoning
- weaker responses on complex tasks

---

### Hardware Constraints

Training was limited by **Google Colab free-tier GPU restrictions**, requiring a **checkpoint-resume strategy**.

---

### Knowledge Cutoff

The fine-tuned model inherits the **knowledge cutoff of the base GPT-Neo model**.

---


### Load the Model for Inference

The base model is loaded in **4-bit mode**, and the **LoRA adapters** are applied.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = "EleutherAI/gpt-neo-1.3B"

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_4bit=True,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(base_model)

model = PeftModel.from_pretrained(model, "alpaca-neo-lora")

prompt = "Explain reinforcement learning simply."

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=150)

print(tokenizer.decode(outputs[0]))
```



## Model Comparison: Base vs. Fine-Tuned

The following table compares the original **GPT-Neo 1.3B** (Base) against the **Alpaca-Neo** (Fine-Tuned) version. Notice how the base model often hallucinates stories or news articles, while the fine-tuned version adheres to the instruction-response format.

| Instruction | Base GPT-Neo 1.3B | Fine-Tuned (Alpaca-Neo) | Improvement |
| :--- | :--- | :--- | :--- |
| **Write a poem about a lonely robot.** | *[Hallucinated Story]* "A robot is a robot, unless it’s a person... You can't build a robot that can think!" | "I walk in a lonely world... I hear my footsteps fading away... A lonely robot, I'm becoming a friend." | **Creative Alignment**: Moved from a rambling definition to actual poetic structure. |
| **Explain a microwave to a 5-year-old.** | *[News Fragment]* "Dale Gaskins, a fifth-grader at St. John’s Elementary... moved to Dutchess County..." | "A microwave is a device that heats up food... It uses microwaves, which are radio frequency waves..." | **Fact Retention**: Corrected the model from generating random names to explaining the actual concept. |
| **3 Healthy breakfast ideas.** | *[Blog Rant]* "You may have heard of the ‘Diet for Diabetics’... weight when you'll put on weight anyway..." | **1.** Fruit Juice Smoothie **2.** Avocado Toast **3.** Eggs Benedict | **Structural Integrity**: Successfully adopted the numbered list format requested. |
| **Formal email for a day off.** | *[Meta-commentary]* "This is the kind of request that has been coming in... meet up with a friend..." | "Dear Sir or Madam, I am writing to ask you to please give me a day of your time off today..." | **Task Completion**: Shifted from talking *about* a request to actually *writing* the request. |

> **Technical Note:** The fine-tuned model was trained using **QLoRA (4-bit)** for 900 steps. While some circular logic remains (typical for 1.3B models), the adherence to the Alpaca instruction template is nearly 100%.


---

# Project Takeaways

This project demonstrates how **modern fine-tuning techniques** can make LLM development accessible without enterprise-level hardware.

Key lessons:

- **QLoRA + PEFT dramatically reduce compute costs**
- **Instruction datasets like Alpaca can effectively align model behavior**
- **Checkpointing and memory optimization are critical when training on limited GPUs**

The result is a **lightweight instruction-following LLM built entirely on consumer hardware**.
