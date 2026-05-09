import os
import torch
import gradio as gr
from PIL import Image
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# --- Ultra-Minimalist SaaS CSS ---
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

.gradio-container { 
    background-color: #f3f4f6; 
    font-family: 'Inter', sans-serif; 
}

/* Centered Header */
#header-container { 
    text-align: center; 
    margin: 3rem 0; 
}
#header-title { 
    color: #000000 !important; /* Full Black Forced */
    font-size: 3rem; 
    font-weight: 800; 
    letter-spacing: -0.03em; 
    margin-bottom: 0.5rem; 
}
#header-subtitle { 
    color: #6b7280; 
    font-size: 1.1rem; 
    font-weight: 300; 
}

/* Unified Card Layout */
.app-card {
    background: #ffffff !important;
    border-radius: 24px !important;
    box-shadow: 0 20px 40px -15px rgba(0,0,0,0.05) !important;
    padding: 2rem !important;
    border: 1px solid #e5e7eb !important;
    margin: 0 auto;
    max-width: 1200px;
}

/* Custom Pill Button */
button.primary { 
    background: #111827 !important; 
    color: white !important; 
    font-weight: 600 !important; 
    border-radius: 9999px !important; 
    border: none !important; 
    padding: 14px 28px !important;
    box-shadow: 0 4px 14px 0 rgba(0, 0, 0, 0.1) !important;
    transition: all 0.3s ease !important;
    margin-top: 1rem !important;
}
button.primary:hover { 
    background: #4f46e5 !important; 
    transform: translateY(-2px) !important; 
    box-shadow: 0 8px 24px rgba(79, 70, 229, 0.25) !important;
}

/* Clean up Gradio defaults */
.gradio-image { border-radius: 12px !important; overflow: hidden; }
footer { display: none !important; }
"""

# --- Path Configuration ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
# Files are in the root directory alongside app.py
LORA_PATH = CURRENT_DIR 

# --- Model Loading ---
print(f"Initializing Base Model: {MODEL_ID}")

# BUGFIX: Force the model onto a specific device instead of using "auto"
device_map = {"": 0} if torch.cuda.is_available() else {"": "cpu"}

base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map=device_map
)

print(f"Applying LoRA Adapters from: {LORA_PATH}")

# BUGFIX: Add is_trainable=False to prevent PEFT from building gradient tracking
model = PeftModel.from_pretrained(
    base_model, 
    LORA_PATH, 
    is_trainable=False
)
processor = AutoProcessor.from_pretrained(LORA_PATH)
model.eval()

# --- Inference Logic ---
def analyze_document(input_img):
    if input_img is None:
        return "Error: No image provided. Please upload a document."
    
    # HARDCODED SYSTEM PROMPT (Hidden from User Interface)
    INTERNAL_PROMPT = "Convert this document image into structured Markdown. Ensure all text, tables, and formatting are captured accurately."

    try:
        # Prepare content for Qwen2-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": input_img},
                    {"type": "text", "text": INTERNAL_PROMPT},
                ],
            }
        ]

        # 1. Apply Template
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # 2. Extract Vision Info
        image_inputs, video_inputs = process_vision_info(messages)
        
        # 3. Process Inputs (Force float16 to match model)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(model.device).to(torch.float16)

        # 4. Generate
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=1024,
                do_sample=False # Keep it deterministic for document tasks
            )
            
            # Trim the prompt tokens
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )

        return output_text[0]
    
    except Exception as e:
        return f"System Error: {str(e)}"

# --- UI Layout ---
with gr.Blocks(css=custom_css, theme=gr.themes.Base()) as demo:
    
    # Minimalist Header - Span removed for pure black text
    gr.HTML("""
        <div id="header-container">
            <h1 id="header-title">LumiDoc</h1>
            <p id="header-subtitle">Intelligent document to markdown extraction.</p>
        </div>
    """)
    
    # Unified Floating Card
    with gr.Column(elem_classes="app-card"):
        with gr.Row():
            
            # Left Side: Input
            with gr.Column(scale=1):
                img_input = gr.Image(type="pil", label="Source Document")
                submit_btn = gr.Button("Extract Data", variant="primary", size="lg")
            
            # Right Side: Output
            with gr.Column(scale=1):
                md_output = gr.Markdown(label="Parsed Result")
                with gr.Accordion("View Developer Source", open=False):
                    raw_output = gr.Code(label="Markdown Output", language="markdown")

    # Map Events 
    submit_btn.click(
        fn=analyze_document, 
        inputs=[img_input], 
        outputs=[md_output]
    ).then(
        fn=lambda x: x,
        inputs=[md_output],
        outputs=[raw_output]
    )

if __name__ == "__main__":
    demo.launch()