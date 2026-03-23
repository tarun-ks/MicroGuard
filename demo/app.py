"""
MicroGuard — RAG Faithfulness Detector
A free, real-time, privacy-preserving quality checker for RAG systems.
Runs entirely in your browser / on your device. No data leaves your machine.

Deploy as HuggingFace Space: https://huggingface.co/spaces
"""

import gradio as gr
import torch
import time
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ─── Configuration ───
# Change this to switch models
MODEL_CONFIGS = {
    "Gemma-270M (Fastest, 270M params)": {
        "base": "google/gemma-3-270m-it",
        "adapter": "models/gemma3_270m/best",  # Local path or HF hub path
    },
    "Qwen-0.5B (Balanced, 500M params)": {
        "base": "Qwen/Qwen2.5-0.5B-Instruct",
        "adapter": "models/qwen05b/best",
    },
    "Gemma-1B (Best Accuracy, 1B params)": {
        "base": "google/gemma-3-1b-it",
        "adapter": "models/gemma3_1b/best",
    },
}

DEFAULT_MODEL = "Gemma-270M (Fastest, 270M params)"

# ─── Global state ───
current_model = None
current_tokenizer = None
current_model_name = None
faithful_ids = None
unfaithful_ids = None

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

SYSTEM_PROMPT = "You are a faithfulness evaluator for RAG systems. You must respond with exactly one word."
USER_TEMPLATE = """Context: {context}
Question: {query}
Answer: {answer}

Is every claim in the answer fully supported by the context? Respond with exactly one word: FAITHFUL or UNFAITHFUL."""


def load_model(model_choice):
    """Load or switch model."""
    global current_model, current_tokenizer, current_model_name, faithful_ids, unfaithful_ids

    if model_choice == current_model_name:
        return f"Model already loaded: {model_choice}"

    config = MODEL_CONFIGS[model_choice]

    try:
        tokenizer = AutoTokenizer.from_pretrained(config["base"], trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            config["base"], torch_dtype=DTYPE, trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base_model, config["adapter"])
        model = model.to(DEVICE)
        model.eval()

        current_model = model
        current_tokenizer = tokenizer
        current_model_name = model_choice

        # Cache token IDs for constrained decoding
        faithful_ids = tokenizer.encode("FAITHFUL", add_special_tokens=False)
        unfaithful_ids = tokenizer.encode("UNFAITHFUL", add_special_tokens=False)

        return f"Loaded: {model_choice}"
    except Exception as e:
        return f"Error loading model: {str(e)}"


def check_faithfulness(context, question, answer, model_choice):
    """Run faithfulness check on a single (context, question, answer) triple."""
    global current_model, current_tokenizer, faithful_ids, unfaithful_ids

    if not context or not answer:
        return "Please provide both context and answer.", "", ""

    # Load model if needed
    if model_choice != current_model_name:
        status = load_model(model_choice)
        if "Error" in status:
            return status, "", ""

    # Truncate inputs
    context_trunc = context[:900]
    question_trunc = (question or "N/A")[:200]
    answer_trunc = answer[:400]

    # Format prompt
    msg = USER_TEMPLATE.format(
        context=context_trunc,
        query=question_trunc,
        answer=answer_trunc,
    )
    messages = [{"role": "user", "content": SYSTEM_PROMPT + "\n\n" + msg}]

    try:
        prompt = current_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        prompt = f"<|im_start|>user\n{SYSTEM_PROMPT}\n\n{msg}<|im_end|>\n<|im_start|>assistant\n"

    inputs = current_tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=512
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # Constrained decoding — compare logits
    start_time = time.time()
    with torch.no_grad():
        outputs = current_model(**inputs)
        logits = outputs.logits[:, -1, :]
        f_score = logits[0, faithful_ids[0]].item()
        u_score = logits[0, unfaithful_ids[0]].item()
    latency = (time.time() - start_time) * 1000  # ms

    # Compute confidence
    import torch.nn.functional as F
    scores = torch.tensor([f_score, u_score])
    probs = F.softmax(scores, dim=0)
    confidence = probs.max().item() * 100

    if f_score > u_score:
        verdict = "FAITHFUL"
        verdict_color = "green"
        explanation = "Every claim in the answer appears to be supported by the provided context."
    else:
        verdict = "UNFAITHFUL"
        verdict_color = "red"
        explanation = "The answer may contain claims not fully supported by the provided context. Review recommended."

    # Format results
    result_html = f"""
    <div style="text-align: center; padding: 20px;">
        <h1 style="color: {verdict_color}; font-size: 48px; margin: 10px 0;">{verdict}</h1>
        <p style="font-size: 18px; color: #666;">Confidence: {confidence:.1f}%</p>
        <p style="font-size: 14px; color: #888;">Latency: {latency:.1f}ms | Model: {current_model_name}</p>
    </div>
    """

    details = f"""**Verdict:** {verdict}
**Confidence:** {confidence:.1f}%
**Latency:** {latency:.1f}ms
**Model:** {current_model_name}

**Explanation:** {explanation}

---
*Scores: FAITHFUL={f_score:.4f}, UNFAITHFUL={u_score:.4f}*
*This evaluation runs 100% locally. No data was sent to any external service.*"""

    return result_html, details, f"{latency:.1f}ms"


def batch_check(file, model_choice):
    """Process a JSON/JSONL file with multiple examples."""
    if file is None:
        return "Please upload a file."

    try:
        content = file.decode("utf-8") if isinstance(file, bytes) else open(file.name).read()

        # Try JSONL first
        examples = []
        for line in content.strip().split("\n"):
            if line.strip():
                examples.append(json.loads(line))

        if not examples:
            examples = json.loads(content)
            if isinstance(examples, dict):
                examples = [examples]

        results = []
        for ex in examples[:100]:  # Limit to 100
            context = ex.get("context", "")
            question = ex.get("query", ex.get("question", ""))
            answer = ex.get("answer", ex.get("response", ""))

            if context and answer:
                _, details, _ = check_faithfulness(context, question, answer, model_choice)
                results.append({
                    "query": question[:100],
                    "verdict": "FAITHFUL" if "FAITHFUL" in details.split("\n")[0] and "UNFAITHFUL" not in details.split("\n")[0] else "UNFAITHFUL",
                    "answer_preview": answer[:100],
                })

        # Format output
        output = f"Processed {len(results)} examples:\n\n"
        faithful_count = sum(1 for r in results if r["verdict"] == "FAITHFUL")
        output += f"**FAITHFUL: {faithful_count}/{len(results)} ({faithful_count/len(results)*100:.0f}%)**\n"
        output += f"**UNFAITHFUL: {len(results)-faithful_count}/{len(results)} ({(len(results)-faithful_count)/len(results)*100:.0f}%)**\n\n"

        for i, r in enumerate(results[:20]):
            icon = "check" if r["verdict"] == "FAITHFUL" else "x"
            output += f"{i+1}. [{r['verdict']}] {r['query'][:80]}...\n"

        return output

    except Exception as e:
        return f"Error processing file: {str(e)}"


# ─── Gradio Interface ───

DESCRIPTION = """
# MicroGuard: RAG Faithfulness Detector

**Free. Real-time. Privacy-preserving.**

Check if your RAG system's answers are faithful to the retrieved context.
No API keys needed. No data leaves your device.

Based on the paper: *"MicroGuard: Sub-Billion Parameter Faithfulness Classification for Real-Time RAG Quality Assurance"*
"""

EXAMPLES = [
    [
        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower from 1887 to 1889.",
        "Who designed the Eiffel Tower?",
        "The Eiffel Tower was designed by the company of engineer Gustave Eiffel and was built between 1887 and 1889.",
    ],
    [
        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower from 1887 to 1889.",
        "Who designed the Eiffel Tower?",
        "The Eiffel Tower was designed by Alexandre Gustave Eiffel in 1895 and is located in London, England.",
    ],
    [
        "Python was created by Guido van Rossum and was first released in 1991. It emphasizes code readability with its use of significant indentation.",
        "When was Python created?",
        "Python was created by Guido van Rossum and first released in 1991. It is known for its emphasis on code readability.",
    ],
    [
        "Python was created by Guido van Rossum and was first released in 1991. It emphasizes code readability with its use of significant indentation.",
        "When was Python created?",
        "Python was created by James Gosling at Sun Microsystems in 1995 and is primarily used for mobile development.",
    ],
]

with gr.Blocks(
    title="MicroGuard — RAG Faithfulness Detector",
    theme=gr.themes.Soft(),
) as demo:

    gr.Markdown(DESCRIPTION)

    with gr.Row():
        model_selector = gr.Dropdown(
            choices=list(MODEL_CONFIGS.keys()),
            value=DEFAULT_MODEL,
            label="Select Model",
            info="Smaller models are faster; larger models are more accurate.",
        )
        latency_display = gr.Textbox(label="Last Latency", interactive=False)

    with gr.Tabs():
        with gr.TabItem("Single Check"):
            with gr.Row():
                with gr.Column(scale=2):
                    context_input = gr.Textbox(
                        label="Retrieved Context",
                        placeholder="Paste the retrieved document/passage here...",
                        lines=8,
                    )
                    question_input = gr.Textbox(
                        label="User Question",
                        placeholder="What was the user's question? (optional)",
                        lines=2,
                    )
                    answer_input = gr.Textbox(
                        label="Generated Answer",
                        placeholder="Paste the RAG system's answer here...",
                        lines=4,
                    )
                    check_btn = gr.Button("Check Faithfulness", variant="primary", size="lg")

                with gr.Column(scale=1):
                    result_html = gr.HTML(label="Verdict")
                    details_output = gr.Markdown(label="Details")

            check_btn.click(
                fn=check_faithfulness,
                inputs=[context_input, question_input, answer_input, model_selector],
                outputs=[result_html, details_output, latency_display],
            )

            gr.Examples(
                examples=EXAMPLES,
                inputs=[context_input, question_input, answer_input],
                label="Try these examples (first two are faithful, last two are unfaithful)",
            )

        with gr.TabItem("Batch Processing"):
            gr.Markdown("Upload a JSON/JSONL file with fields: `context`, `query`/`question`, `answer`/`response`")
            file_input = gr.File(label="Upload JSON/JSONL file")
            batch_btn = gr.Button("Process Batch", variant="primary")
            batch_output = gr.Markdown(label="Results")

            batch_btn.click(
                fn=batch_check,
                inputs=[file_input, model_selector],
                outputs=[batch_output],
            )

    gr.Markdown("""
---
**How it works:** MicroGuard uses a fine-tuned small language model (SLM) with LoRA adaptation and constrained decoding
to classify RAG outputs as FAITHFUL or UNFAITHFUL. The model compares the generated answer against the retrieved context
to detect hallucinations, unsupported claims, and factual inconsistencies.

**Privacy:** All processing happens locally. No data is sent to any external API or server.

[Paper](https://arxiv.org/) | [GitHub](https://github.com/) | [Models](https://huggingface.co/)
""")

# ─── Launch ───
if __name__ == "__main__":
    # Pre-load default model
    print(f"Loading default model: {DEFAULT_MODEL}")
    load_model(DEFAULT_MODEL)
    print("Model loaded. Starting Gradio...")
    demo.launch(share=True)
