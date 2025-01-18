# classify_wrapper.py
import sys
import json
from main import classify_object

if __name__ == "__main__":
    # Read the .npy file path from the command line
    npy_file = sys.argv[1]

    # Call the classification function
    label = classify_object(npy_file)

    # Output the result as JSON
    print(json.dumps({"label": label}))
