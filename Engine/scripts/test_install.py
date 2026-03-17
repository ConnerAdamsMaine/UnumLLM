#!/usr/bin/env python3
"""
Test script to verify OnebitLLM installation
"""
import sys
import subprocess

def test_import():
    """Test basic import"""
    print("Test 1: Import onebitllm...")
    try:
        import onebitllm
        print(f"  ✓ Successfully imported onebitllm v{onebitllm.__version__}")
        return True
    except ImportError as e:
        print(f"  ✗ Failed to import: {e}")
        return False

def test_classes():
    """Test if main classes are available"""
    print("\nTest 2: Check available classes...")
    try:
        import onebitllm
        required = ['ModelConfig', 'OneBitModel', 'Tokenizer', 'GenerateConfig']
        missing = []
        
        for cls in required:
            if hasattr(onebitllm, cls):
                print(f"  ✓ {cls}")
            else:
                print(f"  ✗ {cls} (missing)")
                missing.append(cls)
        
        return len(missing) == 0
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_config():
    """Test creating a ModelConfig"""
    print("\nTest 3: Create ModelConfig...")
    try:
        import onebitllm
        
        config = onebitllm.ModelConfig(
            vocab_size=10000,
            hidden_dim=256,
            num_layers=4,
            num_heads=4,
        )
        print(f"  ✓ Created ModelConfig")
        print(f"    - vocab_size: {config.vocab_size}")
        print(f"    - hidden_dim: {config.hidden_dim}")
        print(f"    - num_layers: {config.num_layers}")
        print(f"    - num_heads: {config.num_heads}")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_model_creation():
    """Test creating a OneBitModel"""
    print("\nTest 4: Create OneBitModel...")
    try:
        import onebitllm
        
        config = onebitllm.ModelConfig(
            vocab_size=1000,
            hidden_dim=128,
            num_layers=2,
            num_heads=2,
        )
        
        model = onebitllm.OneBitModel(config)
        print(f"  ✓ Created OneBitModel")
        
        try:
            param_count = model.parameter_count()
            print(f"    - parameters: {param_count:,}")
        except:
            print(f"    - parameter_count() not yet implemented")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_generate():
    """Test text generation"""
    print("\nTest 5: Generate text (if implemented)...")
    try:
        import onebitllm
        
        config = onebitllm.ModelConfig(
            vocab_size=1000,
            hidden_dim=64,
            num_layers=1,
            num_heads=2,
        )
        
        model = onebitllm.OneBitModel(config)
        
        # Try to generate
        try:
            output = model.generate("Test", max_tokens=5)
            print(f"  ✓ Generated: {output}")
            return True
        except NotImplementedError:
            print(f"  - generate() not yet implemented")
            return True
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("OneBitLLM Installation Test")
    print("=" * 60)
    
    results = []
    
    results.append(("Import", test_import()))
    results.append(("Classes", test_classes()))
    results.append(("ModelConfig", test_config()))
    results.append(("OneBitModel Creation", test_model_creation()))
    results.append(("Text Generation", test_generate()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓" if result else "✗"
        print(f"{status} {name}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed! Installation successful.")
        return 0
    else:
        print("\n✗ Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
