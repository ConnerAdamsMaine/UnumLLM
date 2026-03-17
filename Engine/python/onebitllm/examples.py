"""
Example usage of OneBitLLM Python API
"""

def example_basic_usage():
    """Basic model loading and inference"""
    import onebitllm
    
    # Create configuration
    config = onebitllm.ModelConfig(
        vocab_size=50000,
        hidden_dim=768,
        num_layers=12,
        num_heads=12,
        intermediate_dim=3072,
        max_seq_length=2048,
        activation="gelu",
        dropout=0.1,
    )
    
    # Initialize model
    model = onebitllm.OneBitModel(config)
    
    # Generate text
    output = model.generate(
        prompt="Once upon a time",
        max_tokens=50,
    )
    
    print("Generated:", output)
    return output


def example_with_generation_config():
    """Generate with custom parameters"""
    import onebitllm
    
    config = onebitllm.ModelConfig(
        vocab_size=50000,
        hidden_dim=768,
        num_layers=12,
        num_heads=12,
    )
    
    model = onebitllm.OneBitModel(config)
    
    # Custom generation settings
    gen_config = onebitllm.GenerateConfig(
        max_tokens=100,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2,
    )
    
    output = model.generate_with_config(
        prompt="The future of AI is",
        config=gen_config,
    )
    
    print("Generated:", output)
    return output


def example_batch_inference():
    """Process multiple prompts efficiently"""
    import onebitllm
    
    config = onebitllm.ModelConfig(
        vocab_size=50000,
        hidden_dim=768,
        num_layers=12,
        num_heads=12,
    )
    
    model = onebitllm.OneBitModel(config)
    
    prompts = [
        "Hello, world!",
        "How are you?",
        "What is AI?",
    ]
    
    results = model.generate_batch(prompts, max_tokens=50)
    
    for prompt, output in zip(prompts, results):
        print(f"Prompt: {prompt}")
        print(f"Output: {output}")
        print()
    
    return results


def example_model_saving_loading():
    """Save and load model weights"""
    import onebitllm
    import tempfile
    import os
    
    config = onebitllm.ModelConfig(
        vocab_size=50000,
        hidden_dim=768,
        num_layers=12,
        num_heads=12,
    )
    
    # Create and save model
    model = onebitllm.OneBitModel(config)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.safetensors")
        
        # Save
        model.save(model_path)
        print(f"Model saved to {model_path}")
        
        # Load
        loaded_model = onebitllm.OneBitModel.load(model_path)
        print(f"Model loaded from {model_path}")
        
        # Verify it works
        output = loaded_model.generate("Test", max_tokens=20)
        print(f"Output from loaded model: {output}")
    
    return True


def example_tokenizer():
    """Use tokenizer for preprocessing"""
    import onebitllm
    
    tokenizer = onebitllm.Tokenizer()
    
    # Encode text
    text = "Hello, world! How are you?"
    tokens = tokenizer.encode(text)
    print(f"Tokens: {tokens}")
    
    # Decode tokens
    decoded = tokenizer.decode(tokens)
    print(f"Decoded: {decoded}")
    
    # Get token info
    vocab_size = tokenizer.vocab_size()
    print(f"Vocabulary size: {vocab_size}")
    
    return tokens


def example_model_info():
    """Get model information"""
    import onebitllm
    
    config = onebitllm.ModelConfig(
        vocab_size=50000,
        hidden_dim=768,
        num_layers=12,
        num_heads=12,
        intermediate_dim=3072,
    )
    
    model = onebitllm.OneBitModel(config)
    
    # Get statistics
    param_count = model.parameter_count()
    print(f"Total parameters: {param_count:,}")
    
    memory_usage = model.memory_usage_mb()
    print(f"Memory usage: {memory_usage:.2f} MB")
    
    # Inference speed
    import time
    prompt = "The quick brown fox"
    
    start = time.time()
    output = model.generate(prompt, max_tokens=10)
    elapsed = time.time() - start
    
    print(f"Generated {len(output.split())} tokens in {elapsed:.3f}s")
    
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("OneBitLLM Python API Examples")
    print("=" * 60)
    print()
    
    try:
        print("Example 1: Basic Usage")
        print("-" * 60)
        example_basic_usage()
    except Exception as e:
        print(f"Error: {e}")
    
    print()
    
    try:
        print("Example 2: Generation with Custom Config")
        print("-" * 60)
        example_with_generation_config()
    except Exception as e:
        print(f"Error: {e}")
    
    print()
    
    try:
        print("Example 3: Tokenizer")
        print("-" * 60)
        example_tokenizer()
    except Exception as e:
        print(f"Error: {e}")
    
    print()
    print("=" * 60)
    print("Examples complete!")
    print("=" * 60)
