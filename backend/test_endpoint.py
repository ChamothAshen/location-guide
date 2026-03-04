import requests
import json

def test_next_stop_prediction():
    url = "http://127.0.0.1:8001/predict-next-stop"
    
    # Test cases with ONLY lat and lon
    test_cases = [
        {"name": "Water Gardens area", "lat": 7.9582, "lon": 80.7605},
        {"name": "Entrance area", "lat": 7.9569, "lon": 80.76},
        {"name": "Mirror Wall area", "lat": 7.9601, "lon": 80.7613},
        {"name": "Summit area", "lat": 7.963, "lon": 80.7607},
    ]
    
    for test in test_cases:
        payload = {
            "lat": test["lat"],
            "lon": test["lon"]
        }
        
        try:
            print(f"\n📡 Testing: {test['name']} ({test['lat']}, {test['lon']})")
            print("-" * 50)
            response = requests.post(url, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Current Location: {result['current_location']}")
                print(f"🎯 Predicted Next Stop: {result['predicted_next_stop']}")
                print(f"📊 Confidence: {result['confidence']*100:.0f}%")
                print(f"💬 {result['message']}")
                print(f"\nTop 3 Predictions:")
                for p in result['top_3_predictions']:
                    print(f"   - {p['poi']}: {p['probability']*100:.0f}%")
            else:
                print(f"❌ Error: {response.status_code}")
                print(response.text)
                
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            print("Note: Make sure the FastAPI server is running on port 8001.")

def test_predict_location():
    url = "http://127.0.0.1:8001/predict"
    payload = {"lat": 7.9582, "lon": 80.7605}
    
    try:
        print(f"\n📡 Testing /predict endpoint...")
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Location: {result['location_name']}")
            print(f"📝 Description: {result['description'][:80]}...")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Connection failed: {e}")

if __name__ == "__main__":
    test_predict_location()
    test_next_stop_prediction()
