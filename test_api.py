#!/usr/bin/env python
"""Quick test of the Flask API"""

import urllib.request
import json

try:
    # Test datasets endpoint
    response = urllib.request.urlopen('http://localhost:5000/api/datasets')
    data = json.loads(response.read())
    
    print("✓ API Response - Success:", data.get('success'))
    print("✓ Number of datasets loaded:", len(data.get('datasets', [])))
    
    if data.get('datasets'):
        print("\nDatasets:")
        for ds in data['datasets']:
            print(f"  - {ds['name']}: {ds['total_samples']} samples (Minority: {ds['minority_class']}, Majority: {ds['majority_class']})")
    
except Exception as e:
    print("✗ Error:", str(e))
