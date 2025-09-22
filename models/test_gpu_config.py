#!/usr/bin/env python3
"""
🔧 TEST GPU CONFIGURATION FIX
Verifica que la configuración GPU funciona correctamente
"""

print("🧪 Testing GPU configuration fix...")

# This should work without RuntimeError
import tensorflow as tf

# Configure GPU immediately after import (same as in notebook)
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU memory growth configured for {len(gpus)} GPU(s)")
        
        # Test that we can create a simple tensor (this initializes TF)
        test_tensor = tf.constant([1, 2, 3])
        print(f"✅ TensorFlow initialized successfully: {test_tensor}")
        
        # This should NOT cause RuntimeError now
        print("✅ No RuntimeError - fix successful!")
        
    else:
        print("⚠️ No GPU detected - running on CPU")
        print("✅ Configuration still works on CPU")
        
except RuntimeError as e:
    print(f"🚨 ERROR: {e}")
    print("❌ Fix not working - GPU config still has issues")
    
print("\n🎯 GPU Configuration Test Complete")
