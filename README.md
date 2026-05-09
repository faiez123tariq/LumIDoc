
<div align="center">

```text
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│        ██╗     ██╗   ██╗███╗   ███╗██╗██████╗  ██████╗  ██████╗     │
│        ██║     ██║   ██║████╗ ████║██║██╔══██╗██╔═══██╗██╔════╝     │
│        ██║     ██║   ██║██╔████╔██║██║██║  ██║██║   ██║██║          │
│        ██║     ██║   ██║██║╚██╔╝██║██║██║  ██║██║   ██║██║          │
│        ███████╗╚██████╔╝██║ ╚═╝ ██║██║██████╔╝╚██████╔╝╚██████╗     │
│        ╚══════╝ ╚═════╝ ╚═╝     ╚═╝╚═╝╚═════╝  ╚═════╝  ╚═════╝     │
│                                                                     │
│              [ every image holds a document waiting ]               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

```

> **LumiDoc** illuminates the structure hidden inside document images —
> turning raw scans into clean, structured Markdown using a fine-tuned Vision-Language Model.

---

## ◈ What is LumiDoc?

LumiDoc is a **document-to-Markdown AI pipeline** built on top of `Qwen2-VL-2B-Instruct`, fine-tuned with **QLoRA** on the Nougat scientific document dataset. Drop in any document image — a research paper, a textbook page, a scanned form — and LumiDoc returns clean, structured Markdown preserving:

* `# Headings` and section hierarchy
* `| Tables |` with full Markdown syntax
* `$Equations$` wrapped in LaTeX
* Bullet lists, numbered lists, captions
* Body text — faithfully transcribed

## ◈ Architecture at a Glance

```text
┌──────────────────────────────────────────────────────────────────┐
│                        LumiDoc Pipeline                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   📄 Document Image                                              │
│        │                                                         │
│        ▼                                                         │
│   ┌─────────────┐     resize 512×512      ┌──────────────────┐   │
│   │  PIL / RGB  │ ──────────────────────► │  Vision Encoder  │   │
│   └─────────────┘                         │  (Qwen2-VL ViT)  │   │
│                                           └────────┬─────────┘   │
│                                                    │             │
│                                           visual tokens          │
│                                                    │             │
│   ┌─────────────────────────────────────────────── ▼ ─────────┐  │
│   │              Language Decoder (28 layers)                 │  │
│   │         Qwen2-VL-2B · RoPE · 32K context window           │  │
│   │                                                           │  │
│   │    ┌──────────────────────────────────────────────────┐   │  │
│   │    │  QLoRA Adapter  r=16 · α=32 · target: all attn   │   │  │
│   │    │  q_proj  k_proj  v_proj  o_proj                  │   │  │
│   │    │  gate_proj  up_proj  down_proj                   │   │  │
│   │    └──────────────────────────────────────────────────┘   │  │
│   └───────────────────────────────────────────────────────────┘  │
│                                    │                             │
│                                    ▼                             │
│                          📝 Structured Markdown                  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

```

## ◈ Model Card

| Property | Value |
| --- | --- |
| **Base Model** | `Qwen/Qwen2-VL-2B-Instruct` |
| **Parameters** | ~2 Billion |
| **Quantization** | 16-bit Float / 4-bit NF4 |
| **Fine-tuning Method** | QLoRA (PEFT) |
| **LoRA Rank** | r = 16 |
| **LoRA Alpha** | α = 32 |
| **Target Modules** | q, k, v, o, gate, up, down projections |
| **Trainable Params** | ~0.8% of total |
| **Dataset** | Nougat (500 samples) |
| **Hardware** | Nvidia T4 GPU |

## ◈ Project Structure

For Hugging Face Spaces, the project is structured to run seamlessly via Gradio without complex downloading scripts.

```text
LumIDoc/
│
├── 📄 app.py                   — Gradio UI + Inference Pipeline
├── 📄 requirements.txt         — HF Spaces Dependencies
├── 📄 README.md                — Space Configuration / Metadata
│
└── 📁 adapter/                 — QLoRA Adapter Weights
    ├── adapter_config.json
    ├── adapter_model.safetensors
    ├── chat_template.jinja
    ├── processor_config.json
    ├── tokenizer.json
    └── tokenizer_config.json

```

## ◈ Quick Start

### 1. Deploy on Hugging Face Spaces (Recommended)

LumiDoc is natively designed to run on Hugging Face Spaces using the Gradio SDK.

1. Create a new Space on Hugging Face and select **Gradio** as the SDK.
2. In the Space Settings, select **Nvidia T4 Small** (or higher) under Space Hardware. *CPU-only tiers will cause out-of-memory errors or extremely slow inference.*
3. Upload the contents of this repository to the Space.
4. **Important:** Ensure your fine-tuned LoRA weights are placed exactly inside an `adapter/` folder in the root directory.
5. The space will automatically install `requirements.txt` and build the environment.

### 2. Run Locally

If you have a local GPU, you can clone and run this repository directly:

```bash
# 1. Clone the repo
git clone [https://github.com/faiez123tariq/LumIDoc.git](https://github.com/faiez123tariq/LumIDoc.git)
cd LumIDoc

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
python app.py

```

Open `http://127.0.0.1:7860` in your browser. Note: The app expects your LoRA weights to be in the same directory or specified `adapter/` folder.

## ◈ Dependencies

Your `requirements.txt` should contain the following for Hugging Face compatibility:

```text
torch==2.1.2
torchvision==0.16.2
transformers==4.46.3
accelerate==1.1.1
peft==0.13.2
huggingface_hub==0.26.2
safetensors==0.4.5
Pillow==10.4.0
qwen-vl-utils==0.0.4
gradio==4.44.1
sentencepiece
numpy<2

```

## ◈ Bonus Experiments

This project includes three ablation studies:

**① Epoch Ablation** — 2 epochs vs 5 epochs, validation loss comparison

**② Prompt Engineering** — standard vs detailed vs minimal prompts, ROUGE comparison

**③ Zero-Shot vs Fine-Tuned** — base model vs LoRA adapter, ROUGE-1 delta

## ◈ Acknowledgements

| Resource | Credit |
| --- | --- |
| Base Model | [Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) by Alibaba Cloud |
| Dataset | [Nougat](https://huggingface.co/datasets/jxu124/nougat) by Meta AI |
| QLoRA | [Dettmers et al., 2023](https://arxiv.org/abs/2305.14314) |
| PEFT | [HuggingFace PEFT library](https://github.com/huggingface/peft) |
| UI | [Gradio](https://gradio.app) |

---

```text
┌──────────────────────────────────────────────────────┐
│  Assignment 05 · Generative AI · AI4009              │
│  National University of Computer & Emerging Sciences │
│  Spring 2026                                         │
└──────────────────────────────────────────────────────┘

```

**Built with** `Qwen2-VL` · `QLoRA` · `Gradio` · `Hugging Face`
