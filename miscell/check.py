# check_null_bytes.py
import os

files_to_check = ['solution.py', 'GRU_model_submission.py', 'utils.py']

for filename in files_to_check:
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            content = f.read()
        
        if b'\x00' in content:
            null_count = content.count(b'\x00')
            print(f"❌ {filename} has {null_count} null bytes at positions:")
            for i, byte in enumerate(content):
                if byte == 0:
                    print(f"   Position {i}")
        else:
            print(f"✓ {filename} is clean")
    else:
        print(f"⚠ {filename} not found")