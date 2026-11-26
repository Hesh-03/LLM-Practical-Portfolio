Lab 04: Fine-Tuning an LLM using LoRA / QLoRA

Module: PUSL 3205 – Large Language Models and Generative AI
Notebook: notebooks/Lab_Sheet_4_Fine_Tuning.ipynb (run in VS Code on the CPU, outputs saved in the notebook)

Laboratory Session

Lab 04 – Fine-Tuning an LLM using LoRA / QLoRA

Objective

Fine-tune a small instruction-tuned language model on a domain-specific question-and-answer dataset using parameter-efficient fine-tuning (LoRA) with Hugging Face PEFT and TRL, and compare the original and fine-tuned models on the same questions, including questions that were not in the training data.

Tasks and Implementation
#	Tasks 
1	Install libraries	transformers, datasets, peft, trl, accelerate, bitsandbytes.
2	Check the GPU	No GPU was available, so everything ran on the CPU in 32-bit (see the GPU section below).
3	Choose a model	Qwen/Qwen2.5-0.5B-Instruct (494 M parameters): a small instruction-tuned causal LM with a chat template.
4	Create the dataset	40 question-and-answer pairs about a fictional computer shop, TechNest Computer Shop (hours, prices, warranty, delivery, repairs, services), in chat format with a system message.
5	Configure LoRA / QLoRA	PEFT LoraConfig: r = 16, alpha = 32, dropout = 0.05, on all attention and MLP projection layers. The notebook uses 4-bit QLoRA when an NVIDIA GPU is present; on my CPU-only computer it automatically used plain LoRA.
6	Supervised fine-tuning	TRL SFTTrainer with SFTConfig: 3 epochs (30 steps), learning rate 2e-4, batch size 4, cosine schedule, maximum length 256.
7–8	Compare models	Asked the original and the fine-tuned model the same 13 test questions (greedy decoding). None of the test questions is in the training data.
–	Save	Saved the LoRA adapter and reloaded it to check it works.

Test question groups: A) trained facts in different words (6 questions), B) new questions that combine trained facts (3), C) facts never given, a hallucination check (2), D) general knowledge, to check the model did not forget (2). Groups A, B and D get an automatic keyword score; group C is judged by reading the answers.

Settings used. Because there was no GPU, I reduced the training length (the notebook's default is 6 epochs) and ran one training run:

Setting	Value used
Device	CPU (float32)
Epochs / steps	3 epochs, 30 steps
Maximum sequence length	256
Generation	Greedy decoding, up to 120 new tokens

Core configuration (full code in the notebook):

python
lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora_config)
trainer = SFTTrainer(model=model, args=sft_config, train_dataset=train_dataset, processing_class=tokenizer)
trainer.train()
Results and Evidence
GPU (Task 2)

What GPU is available? None. torch.cuda.is_available() returned False (results/gpu_info.json), so the model was fine-tuned on the CPU in float32. Because QLoRA (4-bit quantization with bitsandbytes) needs an NVIDIA GPU, plain LoRA was used instead.

Why is a GPU useful for LLM fine-tuning? Fine-tuning is mostly large matrix multiplications, which a GPU performs in parallel on thousands of cores, so it is many times faster than a CPU. The GPU's memory (VRAM) must also hold the model weights, activations, gradients and optimizer state, so it limits the model and batch size. This is why 4-bit QLoRA helps on a small GPU, and why I had to keep the model small (0.5 B parameters) and the training short on a CPU.

GPU check

Dataset

40 question-and-answer pairs (results/training_dataset.json). Each example is about 51–96 tokens (mean 66) after applying the chat template.

Dataset and chat template

Training
Item	Value
Model	Qwen/Qwen2.5-0.5B-Instruct
Mode	LoRA (no GPU, so no 4-bit QLoRA)
Trainable parameters	8,798,208 of 502,830,976 (1.75%)
Training steps	30 (3 epochs × 10 steps)
First / last training loss	4.0991 / 0.4461
Training time	315 seconds (5.2 minutes) on the CPU
LoRA adapter size	46.7 MB (including the tokenizer files)

Training loss Training log

Original vs fine-tuned model
Group	Original correct	Fine-tuned correct	Questions
A. Rephrased	0	1	6
B. New combination	0	1	3
D. General knowledge	2	2	2
Total	2	4	11

The score is an automatic keyword check, so it is a rough measure. All answers are in results/comparison.md.

Score comparison

Example answers (the shop's real data, from the training set, is shown in the last column):

Question	Original model	Fine-tuned model	Shop data
Warranty on a new laptop?	"typically ranges from 12 to 36 months"	"A new laptop comes with a 2-year warranty." ✅	2 years
What time do you open on Sundays?	"we generally follow a standard 9-5 schedule"	"Sunday hours are 08:00 to 17:00." ❌	10:00 AM – 4:00 PM
Is there a student discount?	"Yes, we offer a student discount on most items"	"students get 10% off purchases of LKR 50,000 or more" ❌	5% on accessories and laptops
Cost of a 16GB RAM stick?	"around $20 to $30 USD per GB"	"A 16GB RAM stick costs LKR 25,000." ❌	LKR 18,500
Do you sell gaming chairs? (group C)	"Yes, we do sell gaming chairs..."	"Yes, we have gaming chairs from $1500 to $2800."	not in the data
RTX 5090 price? (group C)	"starts around $3,000-$4,000 USD"	"An RTX 5090 costs LKR 1,238,000."	not in the data

Comparison

Observations

The original model invents details. Asked about a shop it knows nothing about, the original model gave generic or made-up answers: warranties of "12 to 36 months", a "9-5 schedule", RAM at "$20 to $30 USD per GB", a 30-day return window, and it even described itself as "one of the most popular online retailers in Sri Lanka".

Fine-tuning changed the style much more than the facts. After only 30 training steps the model answers in the short, direct style of the training data, uses the LKR currency and the shop's setting, and no longer hedges. But it did not reliably learn the actual facts. The keyword score improved only from 2/11 to 4/11, and only one of the six rephrased questions (the 2-year warranty) was answered correctly. For the others the answers look plausible but have wrong numbers: LKR 25,000 for RAM instead of 18,500, Sunday hours of 08:00–17:00 instead of 10:00–16:00, a 10% student discount instead of 5%, a cheapest laptop of LKR 150,000 instead of 145,000, and a 30-day return instead of 7 days. The one "correct" group B answer (delivery is free) also contains garbled text, so it is a weak pass.

The model was under-trained. The reload test asked "Do you offer student discounts?", which is a question in the training data, and the model still answered "10% off purchases up to LKR 50,000" rather than the trained "5% off accessories and laptops". So even a training question was not memorised. The training loss fell from 4.10 to about 0.45 (below 1.0 from step 9, then fluctuating between about 0.3 and 0.8), but the loss averages over every token, so many easy tokens such as "Yes, we" and "LKR" can bring it down while the specific numbers are still wrong. More epochs would very likely help.

Not-in-training questions (group C). Both models invented answers. The original model confidently said the shop sells gaming chairs and described the RTX 5090. The fine-tuned model also made up prices ("gaming chairs from $1500 to $2800", "LKR 1,238,000") in the short confident shop style. Fine-tuning on facts alone therefore does not teach a model to say "I don't know". That would need training examples that decline to answer, or retrieval (RAG), which is the next lab.

General knowledge was kept (group D). Both models answered "What does RAM stand for?" and "What is an SSD?" correctly, so the short LoRA fine-tuning did not damage general ability. The fine-tuned answers were shorter.

Limitations. One run with a tiny, synthetic dataset, no validation split, greedy decoding, and a crude keyword score. A single run of 3 epochs cannot show how much better the result would be with more training.

Challenges.

No NVIDIA GPU: training ran on the CPU, QLoRA was not possible, and I reduced training to 3 epochs. It still took only 5.2 minutes.
My first attempt on a Google Colab T4 GPU failed with a bfloat16 / compute-capability error, because the T4 does not support bfloat16 in hardware; I then ran the notebook locally on the CPU.
Next time I would train for more epochs (for example 8–10), add several paraphrases of each question to the dataset, and include examples where the shop says it does not have the information.
Learning Outcomes
Learned how parameter-efficient fine-tuning works: LoRA freezes the base model and trains small low-rank adapters (only 1.75% of the parameters here), and QLoRA does the same on a 4-bit base model to save GPU memory.
Learned to prepare an instruction dataset in chat format and to use PEFT and TRL's SFTTrainer for supervised fine-tuning.
Learned why a GPU matters for fine-tuning, and how hardware limits (no GPU, no QLoRA, CPU speed) shape what is possible.
Learned to evaluate fairly: the same questions for both models, questions not in the training data, a hallucination check, and a check that general knowledge is kept. I also learned that the keyword score is a rough measure and that the answers must be read.
Understood that low training loss does not guarantee the model has learned the facts, that fine-tuning on a tiny dataset mainly changes style, and that fine-tuned models can still hallucinate confidently, which is where retrieval (RAG) is useful.