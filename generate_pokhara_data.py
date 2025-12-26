#!/usr/bin/env python3
"""
Enhanced Pokhara Sentiment Data Generator
A high-performance, reliable data generator for Nepali-English code-switched comments.
"""

import asyncio
import aiohttp
import json
import hashlib
import logging
import signal
import sys
import traceback
from pathlib import Path
from typing import List, Dict, Set, Optional, Any
import time
from datetime import datetime
from dataclasses import dataclass, field
from tqdm.asyncio import tqdm_asyncio
import argparse


@dataclass
class GenerationConfig:
    """Configuration data class with validation"""
    model_name: str = "gemma:7b"
    ollama_url: str = "http://localhost:11434/api/generate"
    batch_size: int = 40
    max_concurrent_batches: int = 4
    target_counts: Dict[str, int] = field(default_factory=lambda: {
        "neutral": 6000,
        "positive": 4000,
        "negative": 4000,
        "sarcasm": 3000
    })
    pokhara_keywords: List[str] = field(default_factory=lambda: [
        "pokhara", "lakeside", "phewa", "fewa",
        "sarangkot", "peace pagoda", "davis",
        "begnas", "mahendra cave", "bindhyabasini",
        "old bazaar", "hemja", "matepani",
        "kahun", "annapurna"
    ])
    temperature: float = 0.8
    top_p: float = 0.9
    retry_attempts: int = 3
    checkpoint_interval: int = 100
    output_dir: str = "outputs"
    request_timeout: int = 60
    min_words: int = 6
    max_words: int = 25


class PokharaDataGenerator:
    """Enhanced data generator with async processing, error handling, and quality controls"""
    
    def __init__(self, config_path: Optional[str] = None):
        # Setup logging first
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        self.config = self.load_config(config_path)
        self.session: Optional[aiohttp.ClientSession] = None
        self.generated_hashes: Set[str] = set()
        self.checkpoint_data: Dict[str, Any] = {}
        self.shutdown_event = asyncio.Event()
        self.stats = {
            'total_generated': 0,
            'total_attempted': 0,
            'failed_requests': 0,
            'duplicate_rejections': 0,
            'invalid_rejections': 0,
            'start_time': None
        }
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def setup_logging(self):
        """Configure structured logging"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler('pokhara_generation.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_event.set()
    
    def load_config(self, config_path: Optional[str]) -> GenerationConfig:
        """Load configuration from JSON file or use defaults"""
        config = GenerationConfig()
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                # Update config with file values
                for key, value in config_data.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
                
                self.logger.info(f"Configuration loaded from {config_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load config from {config_path}: {e}")
                self.logger.info("Using default configuration")
        
        return config
    
    async def validate_ollama_connection(self) -> bool:
        """Check if Ollama is running and model is available"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "http://localhost:11434/api/tags",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = [model["name"] for model in data.get("models", [])]
                        
                        if self.config.model_name in models:
                            self.logger.info(f"Model {self.config.model_name} is available")
                            return True
                        else:
                            self.logger.error(f"Model {self.config.model_name} not found. Available models: {models}")
                            return False
                    else:
                        self.logger.error(f"Ollama API returned status {response.status}")
                        return False
        except Exception as e:
            self.logger.error(f"Failed to connect to Ollama: {e}")
            self.logger.info("Make sure Ollama is running: ollama serve")
            return False
    
    def is_pokhara_related(self, text: str) -> bool:
        """Check if text contains Pokhara-related keywords"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.config.pokhara_keywords)
    
    def get_text_hash(self, text: str) -> str:
        """Generate MD5 hash for duplicate detection"""
        return hashlib.md5(text.strip().encode()).hexdigest()
    
    def build_prompt(self, category: str, batch_size: int) -> str:
        """Build the generation prompt with proper formatting"""
        base_prompt = f"""You are an expert linguist generating Nepali–English code-switched (Romanized Nepali) social media comments.

GEOGRAPHIC CONSTRAINT (MANDATORY):
All comments MUST be about Pokhara, Nepal ONLY.
Each sentence MUST mention at least one Pokhara-specific place such as:
Pokhara, Lakeside, Phewa Lake, Sarangkot, World Peace Pagoda, Davis Falls, Begnas Lake, Mahendra Cave, Bindhyabasini Temple, Old Bazaar, Hemja, Matepani, Kahun Danda, Annapurna view.

RULES:
- Romanized Nepali only (NO Devanagari)
- Natural Nepali–English code-switching
- Informal Nepali slang allowed (ramro, babbal, jhurr, khatra, dami, hawa)
- Sentence length: {self.config.min_words}–{self.config.max_words} words
- Output STRICT TSV: text<TAB>sentiment
- No numbering, no explanations
- Each comment must be on a new line"""

        task_prompts = {
            "neutral": f"Generate {batch_size} NEUTRAL Pokhara-related comments. Tone: factual, descriptive, balanced. Label: Neutral",
            "positive": f"Generate {batch_size} POSITIVE Pokhara-related comments. Tone: happy, satisfied, praising. Allowed emojis: 😊 😍 👍. Label: Positive",
            "negative": f"Generate {batch_size} NEGATIVE Pokhara-related comments. Tone: dissatisfied, critical but polite. Allowed emojis: 😡 😞. Label: Negative",
            "sarcasm": f"Generate {batch_size} SARCASM Pokhara-related comments. Use positive words but true sentiment must be NEGATIVE. Emoji–text contradiction is mandatory (😂 😭 😍). Label: Negative"
        }
        
        return f"{base_prompt}\n{task_prompts[category]}"
    
    async def generate_batch(self, category: str, batch_size: int) -> List[str]:
        """Generate a batch of comments using Ollama API with retries"""
        prompt = self.build_prompt(category, batch_size)
        
        payload = {
            "model": self.config.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p
            }
        }
        
        for attempt in range(self.config.retry_attempts):
            if self.shutdown_event.is_set():
                return []
            
            try:
                async with self.session.post(
                    self.config.ollama_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.config.request_timeout)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        raw_response = result.get("response", "")
                        self.logger.info(f"Raw Ollama response for {category} (first 500 chars): {raw_response[:500]}")
                        processed = self.process_response(raw_response, category)
                        self.logger.info(f"Processed {len(processed)} valid entries from {category} batch")
                        return processed
                    else:
                        self.logger.warning(f"API returned status {response.status} for {category}")
                        self.stats['failed_requests'] += 1
            except Exception as e:
                self.logger.error(f"Attempt {attempt + 1} failed for {category}: {e}")
                self.logger.error(traceback.format_exc())
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        return []
    
    def process_response(self, response: str, category: str) -> List[str]:
        """Process API response and filter valid entries"""
        valid_entries = []
        
        if not response:
            return valid_entries
        
        for line in response.split('\n'):
            line = line.strip()
            if '\t' not in line:
                continue
            
            parts = line.split('\t', 1)
            if len(parts) != 2:
                continue
            
            text, label = parts
            text = text.strip()
            label = label.strip()
            
            # Validation checks
            if not self.is_pokhara_related(text):
                self.stats['invalid_rejections'] += 1
                continue
            
            text_hash = self.get_text_hash(text)
            if text_hash in self.generated_hashes:
                self.stats['duplicate_rejections'] += 1
                continue
            
            word_count = len(text.split())
            if word_count < self.config.min_words or word_count > self.config.max_words:
                self.stats['invalid_rejections'] += 1
                continue
            
            # Add to valid entries and track hash
            valid_entries.append(f"{text}\t{label}")
            self.generated_hashes.add(text_hash)
        
        return valid_entries
    
    def save_checkpoint(self, category: str, collected: int):
        """Save generation progress to checkpoint file"""
        checkpoint_file = Path(self.config.output_dir) / f"{category}_checkpoint.json"
        checkpoint_data = {
            "collected": collected,
            "timestamp": datetime.now().isoformat(),
            "category": category,
            "target": self.config.target_counts[category]
        }
        
        try:
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save checkpoint for {category}: {e}")
    
    def load_checkpoint(self, category: str) -> int:
        """Load checkpoint data if exists"""
        checkpoint_file = Path(self.config.output_dir) / f"{category}_checkpoint.json"
        
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint = json.load(f)
                    collected = checkpoint.get("collected", 0)
                    self.logger.info(f"Resuming {category} from checkpoint: {collected} samples")
                    return collected
            except Exception as e:
                self.logger.warning(f"Failed to load checkpoint for {category}: {e}")
        
        return 0
    
    def cleanup_checkpoint(self, category: str):
        """Remove checkpoint file after successful completion"""
        checkpoint_file = Path(self.config.output_dir) / f"{category}_checkpoint.json"
        if checkpoint_file.exists():
            try:
                checkpoint_file.unlink()
            except Exception as e:
                self.logger.warning(f"Failed to cleanup checkpoint for {category}: {e}")
    
    async def generate_category(self, category: str) -> int:
        """Generate samples for a specific category with progress tracking"""
        target = self.config.target_counts[category]
        output_path = Path(self.config.output_dir) / f"{category}.tsv"
        output_path.parent.mkdir(exist_ok=True)
        
        # Load checkpoint if available
        collected = self.load_checkpoint(category)
        start_time = time.time()
        last_log_time = start_time
        
        self.logger.info(f"Starting {category} generation (target: {target})")
        
        # Open file in append mode if resuming, create new if starting fresh
        file_mode = 'a' if collected > 0 else 'w'
        
        with open(output_path, file_mode, encoding='utf-8') as f:
            while collected < target and not self.shutdown_event.is_set():
                batch_size = min(self.config.batch_size, target - collected)
                
                # Generate batch
                entries = await self.generate_batch(category, batch_size)
                
                for entry in entries:
                    f.write(entry + '\n')
                    collected += 1
                    self.stats['total_generated'] += 1
                    
                    # Save checkpoint periodically
                    if collected % self.config.checkpoint_interval == 0:
                        self.save_checkpoint(category, collected)
                        
                        # Log progress
                        elapsed = time.time() - start_time
                        rate = collected / elapsed if elapsed > 0 else 0
                        self.logger.info(
                            f"{category}: {collected}/{target} "
                            f"({collected/target*100:.1f}%) - "
                            f"Rate: {rate:.2f} samples/sec"
                        )
                
                # Small delay to avoid overwhelming the API
                await asyncio.sleep(0.1)
                
                # Log progress every 30 seconds
                current_time = time.time()
                if current_time - last_log_time > 30:
                    elapsed = current_time - start_time
                    rate = collected / elapsed if elapsed > 0 else 0
                    self.logger.info(
                        f"{category} progress: {collected}/{target} "
                        f"({collected/target*100:.1f}%) - "
                        f"Rate: {rate:.2f} samples/sec"
                    )
                    last_log_time = current_time
        
        # Cleanup checkpoint if completed
        if collected >= target:
            self.cleanup_checkpoint(category)
        
        elapsed = time.time() - start_time
        self.logger.info(
            f"Completed {category}: {collected} samples in {elapsed:.1f}s "
            f"({collected/elapsed:.2f} samples/sec average)"
        )
        
        return collected
    
    async def generate_all(self) -> Dict[str, int]:
        """Generate samples for all categories concurrently"""
        self.stats['start_time'] = time.time()
        
        # Validate Ollama connection first
        if not await self.validate_ollama_connection():
            raise RuntimeError("Ollama connection failed or model not available")
        
        # Create HTTP session with connection pooling
        connector = aiohttp.TCPConnector(
            limit=self.config.max_concurrent_batches,
            limit_per_host=self.config.max_concurrent_batches
        )
        timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
        
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        ) as self.session:
            # Generate all categories concurrently
            tasks = [
                self.generate_category(category)
                for category in self.config.target_counts.keys()
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results and handle exceptions
            summary = {}
            for category, result in zip(self.config.target_counts.keys(), results):
                if isinstance(result, Exception):
                    self.logger.error(f"{category} failed: {result}")
                    summary[category] = 0
                else:
                    summary[category] = result
            
            return summary
    
    def print_summary(self, results: Dict[str, int]):
        """Print generation summary and statistics"""
        print("\n" + "="*60)
        print("GENERATION SUMMARY")
        print("="*60)
        
        total_generated = sum(results.values())
        total_target = sum(self.config.target_counts.values())
        
        for category, count in results.items():
            target = self.config.target_counts[category]
            percentage = (count / target * 100) if target > 0 else 0
            status = "✅ COMPLETE" if count >= target else "❌ INCOMPLETE"
            print(f"{category.upper():12s}: {count:5d}/{target:5d} ({percentage:5.1f}%) {status}")
        
        print("-" * 60)
        overall_percentage = (total_generated / total_target * 100) if total_target > 0 else 0
        print(f"{'TOTAL':12s}: {total_generated:5d}/{total_target:5d} ({overall_percentage:5.1f}%)")
        
        # Print statistics
        print("\n" + "="*60)
        print("STATISTICS")
        print("="*60)
        
        if self.stats['start_time']:
            elapsed = time.time() - self.stats['start_time']
            rate = total_generated / elapsed if elapsed > 0 else 0
            print(f"Total time: {elapsed:.1f} seconds")
            print(f"Average rate: {rate:.2f} samples/second")
        
        print(f"Failed requests: {self.stats['failed_requests']}")
        print(f"Duplicate rejections: {self.stats['duplicate_rejections']}")
        print(f"Invalid rejections: {self.stats['invalid_rejections']}")
        print(f"Total attempts: {self.stats['total_attempted']}")
        
        success_rate = ((total_generated / self.stats['total_attempted'] * 100) 
                       if self.stats['total_attempted'] > 0 else 0)
        print(f"Success rate: {success_rate:.1f}%")


async def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(description="Generate Pokhara sentiment data")
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration file (JSON)"
    )
    parser.add_argument(
        "--category", "-cat",
        type=str,
        choices=["neutral", "positive", "negative", "sarcasm"],
        help="Generate only specific category"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without generating data"
    )
    
    args = parser.parse_args()
    
    # Create generator instance
    generator = PokharaDataGenerator(args.config)
    
    if args.dry_run:
        print("🔍 DRY RUN MODE")
        print(f"Configuration: {generator.config}")
        print("✅ Configuration loaded successfully")
        return
    
    try:
        if args.category:
            # Generate only specific category
            result = await generator.generate_category(args.category)
            results = {args.category: result}
        else:
            # Generate all categories
            results = await generator.generate_all()
        
        # Print summary
        generator.print_summary(results)
        
    except Exception as e:
        generator.logger.error(f"Generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
