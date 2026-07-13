import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import traceback
from streamlit.testing.v1 import AppTest

def verify_pages():
    pages_dir = Path("app/pages")
    all_pages = [Path("app/Home.py")] + list(pages_dir.glob("*.py"))
    
    results = {}
    
    for page in all_pages:
        try:
            at = AppTest.from_file(str(page))
            at.run(timeout=30)
            if at.exception:
                results[page.name] = {"status": "FAILED", "error": str(at.exception[0])}
            else:
                results[page.name] = {"status": "COMPLETE", "error": None}
        except Exception as e:
            results[page.name] = {"status": "FAILED", "error": traceback.format_exc()}
            
    # Print summary
    for page, res in results.items():
        print(f"[{res['status']}] {page}")
        if res['error']:
            print(f"   Error: {res['error']}")

if __name__ == "__main__":
    verify_pages()
