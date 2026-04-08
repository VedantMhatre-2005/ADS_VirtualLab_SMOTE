#!/usr/bin/env python
"""Test GAN and SMOTE endpoints"""

import urllib.request
import json
import time

print("=" * 60)
print("Testing GAN Endpoint")
print("=" * 60)

try:
    req = urllib.request.Request(
        'http://localhost:5000/api/gan/Diabetes',
        data=b'',
        method='POST'
    )
    response = urllib.request.urlopen(req)
    data = json.loads(response.read())
    
    print("✓ GAN Request Successful")
    if data.get('success'):
        print(f"  - Original Minority: {data['originalMinority']}")
        print(f"  - Generated Samples: {data['generatedSamples']}")
        print(f"  - Balanced Dataset: {data['balancedDataset']}")
        print(f"  - Generator Loss: {data['generatorLoss']}")
        print(f"  - Discriminator Accuracy: {data['discriminatorAccuracy']:.1%}")
        print(f"  - Training Time: {data['trainingTime']:.2f}s")
        if data.get('explanation'):
            print(f"\nExplanation:\n{data['explanation'][:200]}...")
    else:
        print(f"✗ Error: {data.get('error')}")
except Exception as e:
    print(f"✗ Error: {str(e)}")

print("\n" + "=" * 60)
print("Testing SMOTE Endpoint")
print("=" * 60)

try:
    req = urllib.request.Request(
        'http://localhost:5000/api/smote/Diabetes',
        data=b'',
        method='POST'
    )
    response = urllib.request.urlopen(req)
    data = json.loads(response.read())
    
    print("✓ SMOTE Request Successful")
    if data.get('success'):
        print(f"  - Original Minority: {data['originalMinority']}")
        print(f"  - Generated Samples: {data['generatedSamples']}")
        print(f"  - Balanced Dataset: {data['balancedDataset']}")
        print(f"  - K-Neighbors: {data['neighbors']}")
        print(f"  - Training Time: {data['trainingTime']:.2f}s")
        if data.get('explanation'):
            print(f"\nExplanation:\n{data['explanation'][:200]}...")
    else:
        print(f"✗ Error: {data.get('error')}")
except Exception as e:
    print(f"✗ Error: {str(e)}")
