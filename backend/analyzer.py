"""
News Analysis Module
Uses AI model to extract financial information from articles
"""
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from config import OUTPUT_SCHEME, SYSTEM_MESSAGE, USE_MODAL, OUTPUT_DIR

logger = logging.getLogger(__name__)


class LocalAnalyzer:
    """Analyzer using local HuggingFace model"""
    
    def __init__(self):
        """Initialize model (requires GPU/CPU locally)"""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import json_repair
            
            self.torch = torch
            self.json_repair = json_repair
            
            MODEL_ID = "tegana/qwen2.5-arabic-finance-news-parser"
            logger.info(f"Loading local model: {MODEL_ID}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                device_map="auto",
                torch_dtype=torch.bfloat16
            )
            self.model.eval()
            logger.info("✅ Local model loaded successfully")
        except ImportError:
            logger.error("Required dependencies not installed. Install: torch transformers accelerate json-repair")
            raise
    
    def analyze_article(self, article: dict) -> dict:
        """
        Analyze single article using local model
        
        Args:
            article: News article dict with 'body', 'headline', 'link', etc.
        
        Returns:
            dict with extracted financial info
        """
        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": "\n".join([
                "## Article:",
                article["body"].strip(),
                "",
                "## Output Scheme:",
                OUTPUT_SCHEME,
                "",
                "## Output JSON:"
            ])}
        ]
        
        # Generate
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        with self.torch.no_grad():
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
        
        # Parse JSON
        try:
            parsed = self.json_repair.loads(raw_text)
        except Exception as e:
            logger.warning(f"JSON parsing failed: {e}")
            parsed = {"error": "Failed to parse model output"}
        
        # Add metadata
        result = {
            "company_name": article["company"],
            "ticker": article["ticker"],
            "news_date": article["date"],
            "headline": article["headline"],
            "link": article["link"],
            "scraped_at": article["scraped_at"],
        }
        result.update(parsed)
        return result


class ModalAnalyzer:
    """Analyzer using Modal serverless compute"""
    
    def __init__(self):
        """Initialize Modal client"""
        try:
            import modal
            self.modal = modal
            self.remote_cls = self.modal.Cls.from_name(
                "egx-finance-analyzer",
                "NewsAnalyzer"
            )
            self.remote_instance = self.remote_cls()
            logger.info("Connected to Modal app and NewsAnalyzer class")
        except ImportError:
            logger.error("Modal not installed. Install with: pip install modal")
            raise
        except Exception as e:
            logger.error(
                "Failed to connect to Modal deployment. "
                "Make sure you deployed with: modal deploy modal_functions/ai_model.py"
            )
            raise RuntimeError(f"Modal connection failed: {e}")
    
    def analyze_article(self, article: dict) -> dict:
        """
        Analyze article using Modal (serverless GPU)
        
        Args:
            article: News article dict
        
        Returns:
            dict with extracted financial info
        """
        parsed = self.remote_instance.analyze.remote(
            article["body"],
            OUTPUT_SCHEME,
            SYSTEM_MESSAGE
        )

        result = {
            "company_name": article["company"],
            "ticker": article["ticker"],
            "news_date": article["date"],
            "headline": article["headline"],
            "link": article["link"],
            "scraped_at": article["scraped_at"],
        }
        if isinstance(parsed, dict):
            result.update(parsed)
        else:
            result["error"] = "Invalid response from Modal analyzer"
        return result


def get_analyzer():
    """
    Factory function to get appropriate analyzer
    
    Returns:
        LocalAnalyzer or ModalAnalyzer based on USE_MODAL setting
    """
    if USE_MODAL:
        logger.info("Using Modal for analysis")
        return ModalAnalyzer()
    else:
        logger.info("Using local model for analysis")
        return LocalAnalyzer()


def analyze_news_batch(articles: list, analyzer=None, max_workers=None, parallel=True) -> list:
    """
    Analyze multiple news articles with optional parallel execution.
    
    Args:
        articles: List of news article dicts
        analyzer: Analyzer instance (uses default if None)
        max_workers: Max concurrent workers for parallel execution. Default: min(len(articles), 5)
        parallel: Enable parallel execution. Sequential if False or len(articles)==1
    
    Returns:
        List of analyzed articles with extracted info (preserves order)
    """
    if not analyzer:
        analyzer = get_analyzer()
    
    if not articles:
        return []
    
    # Determine parallelization strategy
    should_parallelize = parallel and len(articles) > 1
    if max_workers is None:
        max_workers = min(len(articles), 5)
    
    start_time = time.time()
    
    if not should_parallelize:
        # Sequential execution
        logger.info(f"Processing {len(articles)} articles sequentially")
        results = []
        for i, article in enumerate(articles, 1):
            try:
                logger.info(f"Analyzing article {i}/{len(articles)}: {article['headline'][:50]}")
                result = analyzer.analyze_article(article)
                results.append(result)
                logger.info(f"  ✓ {result.get('event_type', '?')} | {result.get('sentiment', '?')} | {result.get('impact_level', '?')}")
            except Exception as e:
                logger.error(f"Error analyzing article {i}: {e}")
                # Append error result to maintain list structure
                results.append({
                    "company_name": article.get("company_name") or article.get("company"),
                    "ticker": article.get("ticker"),
                    "news_date": article.get("news_date") or article.get("date"),
                    "headline": article.get("headline") or article.get("title"),
                    "link": article.get("link") or article.get("url"),
                    "scraped_at": article.get("scraped_at"),
                    "error": str(e)
                })
    else:
        # Parallel execution with ThreadPoolExecutor
        logger.info(f"Processing {len(articles)} articles in parallel: parallel=True, max_workers={max_workers}")
        
        def analyze_one(index_article):
            """Analyze a single article with index tracking"""
            index, article = index_article
            try:
                logger.info(f"[Worker] Analyzing article {index + 1}/{len(articles)}: {article['headline'][:40]}")
                result = analyzer.analyze_article(article)
                return index, result
            except Exception as e:
                logger.error(f"[Worker] Error analyzing article {index + 1}: {e}")
                return index, {
                    "company_name": article.get("company_name") or article.get("company"),
                    "ticker": article.get("ticker"),
                    "news_date": article.get("news_date") or article.get("date"),
                    "headline": article.get("headline") or article.get("title"),
                    "link": article.get("link") or article.get("url"),
                    "scraped_at": article.get("scraped_at"),
                    "error": str(e)
                }
        
        # Initialize result array with None placeholders to maintain order
        ordered_results = [None] * len(articles)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks with index
            futures = {
                executor.submit(analyze_one, (i, article)): i 
                for i, article in enumerate(articles)
            }
            
            # Collect results as they complete, but maintain original order
            completed = 0
            for future in as_completed(futures):
                try:
                    index, result = future.result()
                    ordered_results[index] = result
                    completed += 1
                    logger.info(f"  ✓ Article {index + 1} completed: {result.get('event_type', '?')} | {result.get('sentiment', '?')} | {result.get('impact_level', '?')}")
                except Exception as e:
                    logger.error(f"Future exception: {e}")
                    completed += 1
        
        results = ordered_results
    
    # Log performance metrics
    elapsed_time = time.time() - start_time
    avg_per_article = elapsed_time / len(articles) if articles else 0
    logger.info(
        f"News batch analysis completed: "
        f"parallel={should_parallelize}, "
        f"max_workers={max_workers if should_parallelize else 'N/A'}, "
        f"articles={len(articles)}, "
        f"total_time={elapsed_time:.2f}s, "
        f"avg_per_article={avg_per_article:.2f}s"
    )
    
    return results


def save_results(results: list, stock_code: str) -> str:
    """
    Save analysis results to JSON file
    
    Args:
        results: List of analysis results
        stock_code: Stock ticker
    
    Returns:
        Path to saved file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{stock_code}_{timestamp}.json"
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    logger.info(f"Results saved to: {filepath}")
    return filepath
