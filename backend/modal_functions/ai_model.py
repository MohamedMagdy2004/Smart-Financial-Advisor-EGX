"""
Modal Functions for AI News Analysis
This runs serverless on Modal - perfect for limited local resources!
"""
import json
import torch
import json_repair
from transformers import AutoModelForCausalLM, AutoTokenizer
import modal

# Create Modal stub
app = modal.App("egx-finance-analyzer")

# Create a persistent volume for the model cache
model_cache = modal.Volume.from_name("model-cache", create_if_missing=True)

# Define container image with required libraries
# استبدل تعريف الـ image القديم بالسطور دي:
image = (
    modal.Image.debian_slim()
    .pip_install("numpy")  # خطوة 1: تثبيت numpy لوحدها كطبقة أساسية
    .pip_install(          # خطوة 2: تثبيت باقي المكتبات فوق الطبقة الجاهزة
        "torch",
        "transformers",
        "accelerate",
        "json-repair"

    )
)


@app.cls(
    image=image,
    volumes={"/model_cache": model_cache},
    gpu="T4",  # Use GPU for faster inference
    timeout=600
)
class NewsAnalyzer:
    """Service for analyzing financial news articles"""

    @modal.enter()
    def startup(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = "tegana/qwen2.5-arabic-finance-news-parser"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model on device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            cache_dir="/model_cache"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            cache_dir="/model_cache"
        )
        self.model.eval()
        print("✅ Model loaded successfully!")
    
    @modal.method()
    def analyze(self, article_body: str, output_scheme: str, system_message: str) -> dict:
        """
        Analyze a single news article
        
        Args:
            article_body: The news article text
            output_scheme: JSON schema for expected output
            system_message: System prompt for the model
        
        Returns:
            dict: Parsed JSON with extracted financial information
        """
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": "\n".join([
                "## Article:",
                article_body.strip(),
                "",
                "## Output Scheme:",
                output_scheme,
                "",
                "## Output JSON:"
            ])}
        ]
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        # Generate output
        with torch.no_grad():
            out = self.model.generate(
                inputs.input_ids,
                max_new_tokens=1024,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None
            )
        
        # Decode
        out_ids = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
        raw_text = self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0]
        
        # Parse and repair JSON
        try:
            parsed = json_repair.loads(raw_text)
        except Exception as e:
            print(f"Warning: JSON parsing failed: {e}")
            parsed = {"error": "Failed to parse response"}
        
        return parsed
    
    @modal.method()
    def batch_analyze(self, articles: list, output_scheme: str, system_message: str) -> list:
        """
        Analyze multiple articles in batch
        
        Args:
            articles: List of article texts
            output_scheme: JSON schema for expected output
            system_message: System prompt for the model
        
        Returns:
            list: List of parsed JSON results
        """
        results = []
        for i, article in enumerate(articles):
            print(f"Analyzing article {i+1}/{len(articles)}...")
            result = self.analyze(article, output_scheme, system_message)
            results.append(result)
        return results


# Standalone function for quick invocation (without class)
@app.function(image=image, gpu="T4", timeout=60)
def quick_analyze(article_body: str, output_scheme: str, system_message: str) -> dict:
    """Quick one-off analysis without persistent model"""
    
    tokenizer = AutoTokenizer.from_pretrained("tegana/qwen2.5-arabic-finance-news-parser")
    model = AutoModelForCausalLM.from_pretrained(
        "tegana/qwen2.5-arabic-finance-news-parser",
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    model.eval()
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": "\n".join([
            "## Article:",
            article_body.strip(),
            "",
            "## Output Scheme:",
            output_scheme,
            "",
            "## Output JSON:"
        ])}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        out = model.generate(
            inputs.input_ids,
            max_new_tokens=1024,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None
        )
    
    out_ids = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
    raw_text = tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0]
    
    try:
        parsed = json_repair.loads(raw_text)
    except Exception as e:
        parsed = {"error": f"Failed to parse: {e}"}
    
    return parsed


if __name__ == "__main__":
    # For testing: `modal run ai_model.py`
    print("Modal functions are ready to deploy!")
