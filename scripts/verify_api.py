import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)

def test_api():
    print("Testing /")
    response = client.get("/")
    print(response.status_code, response.json())

    print("Testing /team/Brazil")
    response = client.get("/team/Brazil")
    print(response.status_code, response.json() if response.status_code == 200 else response.text)

    print("Testing /predict")
    response = client.post("/predict", json={"home_team": "Brazil", "away_team": "France"})
    print(response.status_code, response.json() if response.status_code == 200 else response.text)

    print("Testing /simulate")
    response = client.get("/simulate")
    print(response.status_code, response.json() if response.status_code == 200 else response.text)

    print("Testing /model/performance")
    response = client.get("/model/performance")
    print(response.status_code, response.json() if response.status_code == 200 else response.text)

if __name__ == "__main__":
    test_api()
