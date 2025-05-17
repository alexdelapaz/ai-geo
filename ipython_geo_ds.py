 1/1: import requests
 1/2: requests.get('orbnode.com:8000/datasets/places-list-return')
 2/1: requests.get('orbnode.com:8000/datasets/places-list-return')
 2/2: requests.get('http://orbnode.com:8000/datasets/places-list-return')
 2/3: import requests
 2/4: requests.get('orbnode.com:8000/datasets/places-list-return')
 2/5: requests.get('http://orbnode.com:8000/datasets/places-list-return')
 2/6: requests.get('http://orbnode.com:8000/datasets/places-list-return').json()
 2/7: type(requests.get('http://orbnode.com:8000/datasets/places-list-return').json())
 2/8: places = requests.get('http://orbnode.com:8000/datasets/places-list-return').json()
 2/9: places
2/10: places.length
2/11: places.length()
2/12: clear
2/13: len(places)
2/14: places.keys()
2/15: places.values()
2/16: type(places.values())
2/17: type(list(places.values()))
2/18: places_list = (list(places.values()))
2/19: places_list
2/20: [place_name for place_name in places_list]
2/21: places_list
2/22:
for x in place_list:
    print(x)
    break
2/23:
for x in places_list:
    print(x)
    break
2/24:
for x in places_list:
    print(type(x))
    break
2/25:
for x in places_list:
    type(x)
    break
2/26:
for x in places_list:
    print(type(x))
    break
2/27:
for x,y in places_list:
    print(x,y)
    break
2/28:
for x,y,z in places_list:
    print(x,y)
    break
2/29:
for x,y in places_list:
    print(type(x),y)
    break
2/30: places_list
2/31: x,y = places_list[0]
2/32: x
2/33: y
2/34: places_list[0]
2/35: places_list[0].keys()
2/36: places_list[0].values()
2/37: places_list[0].keys()
2/38: places_list[0].values()
2/39: places_list.values()
2/40: places.values() for places in places_list
2/41: [places.values() for places in places_list]
2/42: [places['place_name'] for places in places_list]
2/43: place_names =
2/44: place_names = [places['place_name'] for places in places_list]
2/45: place_names
2/46: 'coffee' in places_names
2/47: 'coffee' in place_names
2/48: from scipy.spatial.distance import cosine
2/49: pip install scipy
2/50: from scipy.spatial.distance import cosine
2/51: (165000 - (2.25*14,580)) X (0.1/12)
2/52: (165000 - (2.25*14,580)) * (0.1/12)
2/53: (165000 - (2.25*14580)) * (0.1/12)
2/54: place_names
2/55: from sklearn.feature_extraction.text import TfidfVectorizer
2/56: from sklearn.feature_extraction.text import TfidfVectorizer
2/57: pip install sklearn
2/58: from sklearn.feature_extraction.text import TfidfVectorizer
 3/1: from sklearn.feature_extraction.text import TfidfVectorizer
 3/2: pip install scikit-learn
 3/3: pip install sklearn
 3/4: pip install scipy
 3/5:
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
 3/6: places = requests.get('http://orbnode.com:8000/datasets/places-list-return').json()
 3/7: import requests
 3/8: places = requests.get('http://orbnode.com:8000/datasets/places-list-return').json()
 3/9: place_names = [places['place_name'] for places in places_list]
3/10: [places['place_name'] for places in places_list]
3/11: places_list = (list(places.values()))
3/12: place_names = [places['place_name'] for places in places_list]
3/13: place_names
3/14:
def compare(str1,str2):
    vectorizer = TfidfVectorizer().fit_transform([str1, str2])
    vectors = vectorizer.toarray()
    cosine_sim = 1 - cosine(vectors[0], vectors[1]) if len(vectors) > 1 else 0
3/15:
def compare(str1,str2):
    vectorizer = TfidfVectorizer().fit_transform([str1, str2])
    vectors = vectorizer.toarray()
    cosine_sim = 1 - cosine(vectors[0], vectors[1]) if len(vectors) > 1 else 0
3/16:
def compare(str1,str2):
    vectorizer = TfidfVectorizer().fit_transform([str1, str2])
    vectors = vectorizer.toarray()
    cosine_sim = 1 - cosine(vectors[0], vectors[1]) if len(vectors) > 1 else 0
3/17:
def compare(str1,str2):
    vectorizer = TfidfVectorizer().fit_transform([str1, str2])
    vectors = vectorizer.toarray()
    cosine_sim = 1 - cosine(vectors[0], vectors[1]) if len(vectors) > 1 else 0
3/18:
def compare(str1,str2):
    vectorizer = TfidfVectorizer().fit_transform([str1, str2])
    vectors = vectorizer.toarray()
    cosine_sim = 1 - cosine(vectors[0], vectors[1]) if len(vectors) > 1 else 0
3/19:
def compare(str1,str2):
    vectorizer = TfidfVectorizer().fit_transform([str1, str2])
    vectors = vectorizer.toarray()
    cosine_sim = 1 - cosine(vectors[0], vectors[1]) if len(vectors) > 1 else 0
    return "Cosine Similarity": round(cosine_sim, 4)
3/20:
def compare(str1,str2):
    vectorizer = TfidfVectorizer().fit_transform([str1, str2])
    vectors = vectorizer.toarray()
    cosine_sim = 1 - cosine(vectors[0], vectors[1]) if len(vectors) > 1 else 0
    return {"Cosine Similarity": round(cosine_sim, 4)}
3/21:
for el in place_names:
    compare("yellow coffee house", el)
3/22: place_names
3/23:
for el in place_names:
    compare("yellow coffee house", el)
3/24:
import numpy as np

def safe_cosine_similarity(str1, str2):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([str1, str2]).toarray()
    
    # Calculate cosine similarity only if both vectors are non-zero
    if np.linalg.norm(vectors[0]) != 0 and np.linalg.norm(vectors[1]) != 0:
        return 1 - cosine(vectors[0], vectors[1])
    else:
        return 0.0
3/25:
for el in place_names:
    safe_cosine_similarity("yellow coffee house", el)
3/26:
for el in place_names:
    safe_cosine_similarity("coffee house", el)
3/27:
for el in place_names:
    safe_cosine_similarity("coffee", el)
3/28:
for el in place_names:
    print(el)
3/29:
for el in place_names:
    print(safe_cosine_similarity("coffee", el))
3/30:
for el in place_names:
    print(safe_cosine_similarity("yellow coffee", el))
3/31:
for el in place_names:
    print(safe_cosine_similarity("yellow tea", el))
3/32:
for el in place_names:
    print(safe_cosine_similarity("yellow jasmine", el))
3/33:
for el in place_names:
    print(safe_cosine_similarity("yellow espresso", el))
3/34:
for el in place_names:
    print(safe_cosine_similarity("yellow espresso & pour over", el))
3/35: places = requests.get('http://orbnode.com:8000/datasets/places-list-return').json()
3/36: place_names = [places['place_name'] for places in places_list]
3/37:
for el in place_names:
    print(safe_cosine_similarity("yellow espresso & pour over", el))
3/38: place_names
3/39: places = requests.get('http://orbnode.com:8000/datasets/places-list-return').json()
3/40: places_list = (list(places.values()))
3/41: place_names = [places['place_name'] for places in places_list]
3/42:
for el in place_names:
    print(safe_cosine_similarity("yellow espresso & pour over", el))
3/43: places = requests.get('http://orbnode.com:8000/datasets/places-list-return').json()
3/44: places
3/45: cd orclear
3/46: clear
3/47: places
 4/1: import requests
 4/2: places = requests.get('http://orbnode.com:8000/datasets/places-list-return').json()
 4/3: places
 4/4: places_names = [place['place_name'] for place in places]
 4/5: places_names = [place['place_name'] place for place in places]
 4/6: places_names = [place['place_name'] for place in places]
 4/7: places_names = [placs for place in places]
 4/8: places_names = [places for place in places]
 4/9: places_names
4/10: place_names = [place['place_name'] for place in places.values()]
4/11: place_names
4/12: 'coffee' in place_names
4/13: 'TEST' in place_names
4/14:
def calculate_cer(reference, hypothesis):
    """
    Calculate Character Error Rate (CER) between two strings.
    
    CER = (Insertions + Deletions + Substitutions) / Length of Reference String
    
    Parameters:
    reference (str): The ground truth string
    hypothesis (str): The string to compare against the reference
    
    Returns:
    float: Character Error Rate between 0 and 1
    """
    if len(reference) == 0:
        raise ValueError("Reference string cannot be empty")
        
    # Calculate Levenshtein distance (edit distance)
    distance = edit_distance(reference, hypothesis)
    
    # Calculate CER
    cer = distance / len(reference)
    
    return cer
 5/1:
def calculate_cer(reference, hypothesis):
    """
    Calculate Character Error Rate (CER) between two strings.
    
    CER = (Insertions + Deletions + Substitutions) / Length of Reference String
    
    Parameters:
    reference (str): The ground truth string
    hypothesis (str): The string to compare against the reference
    
    Returns:
    float: Character Error Rate between 0 and 1
    """
    if len(reference) == 0:
        raise ValueError("Reference string cannot be empty")
        
    # Calculate Levenshtein distance (edit distance)
    distance = edit_distance(reference, hypothesis)
    
    # Calculate CER
    cer = distance / len(reference)
    
    return cer
 5/2: places_names = [place['place_name'] for place in places]
 5/3: places = requests.get('http://orbnode.com:8000/datasets/places-list-return').json()
 5/4: import requests
 5/5: places = requests.get('http://orbnode.com:8000/datasets/places-list-return').json()
 5/6: places
 5/7: place_names
 5/8: places_names = [place['place_name'] for place in places]
 5/9: place_names = [place['place_name'] for place in places.values()]
5/10:
for name in place_names:
    print('coffee',place,calculate_cer('coffee',name)))
5/11:
for name in place_names:
    print('coffee',place,calculate_cer('coffee',name))
5/12:
for name in place_names:
    print('coffee',name,calculate_cer('coffee',name))
5/13:
from sklearn.metrics import edit_distance
import numpy as np
5/14: from torchmetrics.text import CharErrorRate
5/15: pip install torchmetrics
5/16: CharErrorRate('s','s')
5/17: from torchmetrics.text import CharErrorRate
5/18: CharErrorRate('s','s')
5/19: CharErrorRate().update(['coffee'],['yellow coffee'])
5/20: CharErrorRate().update(['coffee'],['yellow coffee']).plot()
5/21: CharErrorRate().update(['coffee'],['yellow coffee']).plot()
5/22: CharErrorRate().update(['coffee'],['yellow coffee']).plot()()
5/23: CharErrorRate().update(['coffee'],['yellow coffee']).plot()
5/24:
cer = CharErrorRate()
predictions = ["hello world", "this is bar"]
references = ["hello world", "this is foo"]
cer.update(predictions, references)
result = cer.compute()
print(result)
5/25: CharErrorRate().update(['coffee'],['yellow coffee']).compute()
5/26: CharErrorRate().update(['coffee'],['yellow coffee'])
5/27: CharErrorRate().update(['coffee'],['yellow coffee']).compute()
5/28: cer = CharErrorRate().update(['coffee'],['yellow coffee'])
5/29: cer.compute()
5/30: cer
5/31: cer = CharErrorRate().update(['coffee'],['yellow coffee'])
5/32: cer
5/33: cer = CharErrorRate().update(['coffee'],['yellow coffee'])
5/34: cer
5/35: cer = CharErrorRate(['coffee'],['yellow coffee'])
5/36: cer = CharErrorRate()
5/37: cer(['d',['d']))
5/38: cer(['d',['d'])
5/39: cer(['d',['d'])
5/40: cer(['d'],['d'])
5/41: cer(['d'],['dddd'])
5/42:
for name in place_names:
    print('coffee',name,cer('coffee',name))
5/43:
for name in place_names:
    print(name,cer('cup of yummay',name))
5/44:
for name in place_names:
    print(name,cer('c',name))
5/45: cer(['Yellow'],['yellow'])
5/46: cer(['Yellow'],['ellow'])
5/47: cer(['Yellow'],['Yellow'])
5/48: cer(['Yellow'],['Yllow'])
5/49: cer(['Yellow'],['Ye llow'])
5/50: cer(['Yellow'],['Yellow Cafe'])
5/51: cer(['Yellow'],['Yellow Cafee'])
5/52: cer(['Yellow'],['Yellow Cafeee'])
5/53: cer(['Yellow'],['Yellow '])
5/54: cer(['Yellow'],['Yellow'])
5/55: cer(['Yellow'],['Yellow '])
5/56: cer(['Yellow'],['Yellow  '])
5/57: cer(['Yellow'],['Yellow   '])
5/58: cer(['Yellow'],['Yellow1'])
5/59: cer(['Yellow'],['Yellow '])
5/60: cer(['Yellow'],['Yellow1 '])
5/61: cer(['Yellow'],['Yellow 1 '])
5/62: cer(['Yellow'],['Yellow 1'])
5/63: cer(['Y'],['Ye'])
5/64: cer(['Yyy'],['Yyye'])
5/65: cer(['Yy'],['Yye'])
5/66: cer(['Yttttttttttttttty'],['Ytttttttttttttttttye'])
5/67: cer(['Yttttttttttttttty'],['Yttttttttttttttttye'])
5/68: cer(['Yttttttttttttttty'],['Yttttttttttttttttyeeee'])
5/69: cer(['Yttttttttttttttty'],['Yttttttttttttttttyeeee'])
5/70:
for name in place_names:
    print(name,cer('Coffee Bar',name))
5/71:
for name in place_names:
    print(name,cer('Coffee Bar',name)<0.5)
    )
5/72:
for name in place_names:
    print(name,cer('Coffee Bar',name)<0.5)))
5/73:
for name in place_names:
    print(name,cer('Coffee Bar',name)<0.5)
5/74:
for name in place_names:
    if cer('Coffee Bar',name)<0.5):
        print(name)
5/75:
for name in place_names:
    if cer('Coffee Bar',name)<0.5:
        print(name)
5/76:
for name in place_names:
    if cer('Coffee Bar',name)<0.5:
        print("Coffee Bar", name)
 6/1:
import requests

def get_address_suggestions(query, country_code=None, limit=5):
    """Fetch address suggestions from OpenStreetMap Nominatim API."""
    base_url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": query,
        "format": "json",
        "addressdetails": 1,
        "limit": limit,
    }

    if country_code:
        params["countrycodes"] = country_code

    headers = {
        "User-Agent": "MyApp/1.0 (myemail@example.com)"
    }

    response = requests.get(base_url, params=params, headers=headers)
    if response.status_code == 200:
        results = response.json()
        suggestions = [
            {
                "display_name": result["display_name"],
                "lat": result["lat"],
                "lon": result["lon"],
                "address": result["address"]
            }
            for result in results
        ]
        return suggestions
    else:
        raise Exception(f"Error: {response.status_code} {response.text}")


# Example usage
if __name__ == "__main__":
    user_input = input("Enter an address: ")
    suggestions = get_address_suggestions(user_input)
    for i, suggestion in enumerate(suggestions, start=1):
        print(f"{i}. {suggestion['display_name']}")

    # You can integrate this into a form autofill by parsing the address details.
 6/2:
import requests

def get_address_suggestions(query, country_code=None, limit=5):
    """Fetch address suggestions from OpenStreetMap Nominatim API."""
    base_url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": query,
        "format": "json",
        "addressdetails": 1,
        "limit": limit,
    }

    if country_code:
        params["countrycodes"] = country_code

    headers = {
        "User-Agent": "MyApp/1.0 (myemail@example.com)"
    }

    response = requests.get(base_url, params=params, headers=headers)
    if response.status_code == 200:
        results = response.json()
        suggestions = [
            {
                "display_name": result["display_name"],
                "lat": result["lat"],
                "lon": result["lon"],
                "address": result["address"]
            }
            for result in results
        ]
        return suggestions
    else:
        raise Exception(f"Error: {response.status_code} {response.text}")


# Example usage
if __name__ == "__main__":
    user_input = input("Enter an address: ")
    suggestions = get_address_suggestions(user_input)
    for i, suggestion in enumerate(suggestions, start=1):
        print(f"{i}. {suggestion['display_name']}")

    # You can integrate this into a form autofill by parsing the address details.
 6/3:
import requests

def get_address_suggestions(query, country_code=None, limit=5):
    """Fetch address suggestions from OpenStreetMap Nominatim API."""
    base_url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": query,
        "format": "json",
        "addressdetails": 1,
        "limit": limit,
    }

    if country_code:
        params["countrycodes"] = country_code

    headers = {
        "User-Agent": "MyApp/1.0 (myemail@example.com)"
    }

    response = requests.get(base_url, params=params, headers=headers)
    if response.status_code == 200:
        results = response.json()
        suggestions = [
            {
                "display_name": result["display_name"],
                "lat": result["lat"],
                "lon": result["lon"],
                "address": result["address"]
            }
            for result in results
        ]
        return suggestions
    else:
        raise Exception(f"Error: {response.status_code} {response.text}")


# Example usage
if __name__ == "__main__":
    user_input = input("Enter an address: ")
    suggestions = get_address_suggestions(user_input, country_code='us')
    for i, suggestion in enumerate(suggestions, start=1):
        print(f"{i}. {suggestion['display_name']}")


    # You can integrate this into a form autofill by parsing the address details.
 6/4:
import requests

def get_address_suggestions(query, country_code=None, limit=5):
    """Fetch address suggestions from OpenStreetMap Nominatim API."""
    base_url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": query,
        "format": "json",
        "addressdetails": 1,
        "limit": limit,
    }

    if country_code:
        params["countrycodes"] = country_code

    headers = {
        "User-Agent": "MyApp/1.0 (myemail@example.com)"
    }

    response = requests.get(base_url, params=params, headers=headers)
    if response.status_code == 200:
        results = response.json()
        suggestions = [
            {
                "display_name": result["display_name"],
                "lat": result["lat"],
                "lon": result["lon"],
                "address": result["address"]
            }
            for result in results
        ]
        return suggestions
    else:
        raise Exception(f"Error: {response.status_code} {response.text}")


# Example usage
if __name__ == "__main__":
    user_input = input("Enter an address: ")
    suggestions = get_address_suggestions(user_input, country_code='us')
    for i, suggestion in enumerate(suggestions, start=1):
        print(f"{i}. {suggestion['display_name']}")


    # You can integrate this into a form autofill by parsing the address details.   ...:     user_input = input("Enter an address: ")
   ...:     suggestions = get_address_suggestions(user_input, country_code='us')
   ...:     for i, suggestion in enumerate(suggestions, start=1):
   ...:         print(f"{i}. {suggestion['display_name']}")
 6/6:
   ...:     user_input = input("Enter an address: ")
   ...:     suggestions = get_address_suggestions(user_input, country_code='us')
   ...:     for i, suggestion in enumerate(suggestions, start=1):
   ...:         print(f"{i}. {suggestion['display_name']}")
 6/7: user_input = input("Enter an address: ")
 6/8: suggestions = get_address_suggestions(user_input)
 6/9: suggestions
6/10:
    for i, suggestion in enumerate(suggestions, start=1):
        print(f"{i}. {suggestion['display_name']}")
6/11: suggestions = get_address_suggestions(user_input, country_code='us')
6/12:
    for i, suggestion in enumerate(suggestions, start=1):
        print(f"{i}. {suggestion['display_name']}")
6/13: suggestions
6/14: suggestions = get_address_suggestions(user_input, country_code='us', limit=1000)
6/15: suggestions
6/16: suggestions = get_address_suggestions(user_input, country_code='US')
6/17: suggestions
6/18: user_input = input("Enter an address: ")
6/19: suggestions = get_address_suggestions(user_input, country_code='US')
6/20: suggestions
6/21:
    for i, suggestion in enumerate(suggestions, start=1):
        print(f"{i}. {suggestion['display_name']}")
6/22: suggestions = get_address_suggestions(user_input, country_code='US', limit=20)
6/23:
    for i, suggestion in enumerate(suggestions, start=1):
        print(f"{i}. {suggestion['display_name']}")
6/24: user_input = input("Enter an address: ")
6/25: suggestions = get_address_suggestions(user_input, country_code='US', limit=20)
6/26:
    for i, suggestion in enumerate(suggestions, start=1):
        print(f"{i}. {suggestion['display_name']}")
6/27: suggestions[11]
6/28: suggestions[10]
6/29: suggestions[0]
6/30: suggestions[10]
6/31: suggestions[9]
6/32: suggestions[9]
6/33: suggestions[9]['display_name']
6/34: user_input = input("Enter an address: ")
6/35:
    for i, suggestion in enumerate(suggestions, start=1):
        print(f"{i}. {suggestion['display_name']}")
6/36: suggestions = get_address_suggestions(user_input, country_code='US', limit=20)
6/37:
    for i, suggestion in enumerate(suggestions, start=1):
        print(f"{i}. {suggestion['display_name']}")
6/38: user_input = input("Enter an address: ")
6/39: suggestions = get_address_suggestions(user_input, country_code='US', limit=20)
6/40:
    for i, suggestion in enumerate(suggestions, start=1):
        print(f"{i}. {suggestion['display_name']}")
6/41: user_input = input("Enter an address: ")
6/42: suggestions = get_address_suggestions(user_input, country_code='US', limit=20)
6/43:
    for i, suggestion in enumerate(suggestions, start=1):
        print(f"{i}. {suggestion['display_name']}")
6/44: user_input = input("Enter an address: ")
6/45: suggestions = get_address_suggestions(user_input, country_code='US', limit=20)
6/46:
    for i, suggestion in enumerate(suggestions, start=1):
        print(f"{i}. {suggestion['display_name']}")
6/47:
    for i, suggestion in enumerate(suggestions, start=1):
        print(f"{i}. {suggestion['display_name']}")
6/48: suggestions = get_address_suggestions(addr, country_code='US', limit=20)
6/49: addr='300 M St'
6/50: suggestions = get_address_suggestions(addr, country_code='US', limit=20)
6/51:
    for i, suggestion in enumerate(suggestions, start=1):
        print(f"{i}. {suggestion['display_name']}")
6/52:
def get_address_suggestions(query, country_code='us', limit=5, address_details=1, viewbox=None):
    """Fetch address suggestions from OpenStreetMap Nominatim API with optional viewbox for priority."""
    base_url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": query,
        "format": "json",
        "addressdetails": address_details,
        "limit": limit,
        "countrycodes": country_code,
        "bounded": 1 if viewbox else 0
    }

    if viewbox:
        params["viewbox"] = viewbox

    headers = {
        "User-Agent": "MyApp/1.0 (myemail@example.com)"
    }

    response = requests.get(base_url, params=params, headers=headers)
    if response.status_code == 200:
        results = response.json()
        suggestions = [
            {
                "display_name": result["display_name"],
                "lat": result["lat"],
                "lon": result["lon"],
                "address": result.get("address", {})
            }
            for result in results
        ]
        return suggestions
    else:
        raise Exception(f"Error: {response.status_code} {response.text}")
6/53:     dc_viewbox = "-77.119759,38.995548,-76.909393,38.803574"
6/54:     suggestions = get_address_suggestions(user_input, viewbox=dc_viewbox)
6/55:
    for i, suggestion in enumerate(suggestions, start=1):
        print(f"{i}. {suggestion['display_name']}")
6/56:     suggestions = get_address_suggestions(addr, viewbox=dc_viewbox)
6/57: addr
6/58:
    for i, suggestion in enumerate(suggestions, start=1):
        print(f"{i}. {suggestion['display_name']}")
6/59:     suggestions = get_address_suggestions(addr, viewbox=dc_viewbox)
6/60: clear
6/61: addr='300 M street ne'
6/62:     suggestions = get_address_suggestions(addr, viewbox=dc_viewbox)
6/63:
    for i, suggestion in enumerate(suggestions, start=1):
        print(f"{i}. {suggestion['display_name']}")
6/64: addr='300 M st'
6/65:
    for i, suggestion in enumerate(suggestions, start=1):
        print(f"{i}. {suggestion['display_name']}")
6/66:     suggestions = get_address_suggestions(addr, viewbox=dc_viewbox)
6/67:
    for i, suggestion in enumerate(suggestions, start=1):
        print(f"{i}. {suggestion['display_name']}")
6/68:

def get_address_suggestions(query, country_code='us', limit=5, address_details=1, viewbox=None, dedupe=True):
    """Fetch address suggestions from OpenStreetMap Nominatim API with optional viewbox for priority."""
    base_url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": query,
        "format": "json",
        "addressdetails": address_details,
        "limit": limit,
        "countrycodes": country_code,
        "bounded": 1 if viewbox else 0
    }

    if viewbox:
        params["viewbox"] = viewbox

    headers = {
        "User-Agent": "MyApp/1.0 (myemail@example.com)"
    }

    response = requests.get(base_url, params=params, headers=headers)
    if response.status_code == 200:
        results = response.json()
        seen = set()
        suggestions = []
        for result in results:
            if dedupe:
                if result["display_name"] in seen:
                    continue
                seen.add(result["display_name"])
            suggestions.append({
                "display_name": result["display_name"],
                "lat": result["lat"],
                "lon": result["lon"],
                "address": result.get("address", {})
            })
        return suggestions
    else:
        raise Exception(f"Error: {response.status_code} {response.text}")
6/69:     suggestions = get_address_suggestions(addr, viewbox=dc_viewbox)
6/70:
    for i, suggestion in enumerate(suggestions, start=1):
        print(f"{i}. {suggestion['display_name']}")
6/71: addr='300 M street ne'
6/72:     suggestions = get_address_suggestions(addr, viewbox=dc_viewbox)
6/73:
    for i, suggestion in enumerate(suggestions, start=1):
        print(f"{i}. {suggestion['display_name']}")
6/74:
def get_address_suggestions(query, country_code='us', limit=5, address_details=1, viewbox=None, dedupe=True, fuzzy_search=True):
    """Fetch address suggestions from OpenStreetMap Nominatim API with optional viewbox for priority and fuzzy search."""
    base_url = "https://nominatim.openstreetmap.org/search"

    # Adjust the query for broader search by removing exact constraints if fuzzy_search is enabled
    if fuzzy_search:
        query = query.replace(',', '').replace('.', '').replace('-', '')

    params = {
        "q": query,
        "format": "json",
        "addressdetails": address_details,
        "limit": limit,
        "countrycodes": country_code,
        "bounded": 1 if viewbox else 0,
        "dedupe": 1
    }

    if viewbox:
        params["viewbox"] = viewbox

    headers = {
        "User-Agent": "MyApp/1.0 (myemail@example.com)"
    }

    response = requests.get(base_url, params=params, headers=headers)
    if response.status_code == 200:
        results = response.json()
        seen = set()
        suggestions = []
        for result in results:
            if dedupe:
                if result["display_name"] in seen:
                    continue
                seen.add(result["display_name"])
            suggestions.append({
                "display_name": result["display_name"],
                "lat": result["lat"],
                "lon": result["lon"],
                "address": result.get("address", {})
            })
        return suggestions
    else:
        raise Exception(f"Error: {response.status_code} {response.text}")
6/75:     suggestions = get_address_suggestions(addr, viewbox=dc_viewbox)
6/76:
    for i, suggestion in enumerate(suggestions, start=1):
        print(f"{i}. {suggestion['display_name']}")
6/77: addr='300 M st'
6/78:     suggestions = get_address_suggestions(addr, viewbox=dc_viewbox)
6/79:
    for i, suggestion in enumerate(suggestions, start=1):
        print(f"{i}. {suggestion['display_name']}")
6/80:
async function searchAddress(query, boundedBy = null) {
    let url = `https://nominatim.openstreetmap.org/search?format=json&addressdetails=1&q=${encodeURIComponent(query)}&limit=5`;
    
    if (boundedBy) {
        url += `&viewbox=${boundedBy.join(',')}&bounded=1`;
    }
    
    const response = await fetch(url);
    const data = await response.json();
    return data;
}

// Example usage
searchAddress('300 m').then(console.log);

// With bounding box
searchAddress('300 m', [-0.5, 51.2, 0.5, 51.8]).then(console.log);
6/81:
import requests

def search_address(query, bounded_by=None):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": query,
        "format": "json",
        "addressdetails": 1,
        "limit": 5  # Adjust as needed
    }
    if bounded_by:
        params["viewbox"] = f"{bounded_by[0]},{bounded_by[1]},{bounded_by[2]},{bounded_by[3]}"
        params["bounded"] = 1

    response = requests.get(url, params=params)
    return response.json()

# Example usage
addresses = search_address("300 m")
bounded_addresses = search_address("300 m", bounded_by=(-0.5, 51.2, 0.5, 51.8))

print(addresses)
print(bounded_addresses)
6/82:
async function searchAddress(query, boundedBy = null) {
    let url = `https://nominatim.openstreetmap.org/search?format=json&addressdetails=1&q=${encodeURIComponent(query)}&limit=5`;
    
    if (boundedBy) {
        url += `&viewbox=${boundedBy.join(',')}&bounded=1`;
    }
    
    const response = await fetch(url);
    const data = await response.json();
    return data;
}

// Example usage
searchAddress('300 m').then(console.log);

// With bounding box
searchAddress('300 m', [-0.5, 51.2, 0.5, 51.8]).then(console.log);
6/83:
import requests

def search_address(query, bounded_by=None):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": query,
        "format": "json",
        "addressdetails": 1,
        "limit": 5  # Adjust as needed
    }
    if bounded_by:
        params["viewbox"] = f"{bounded_by[0]},{bounded_by[1]},{bounded_by[2]},{bounded_by[3]}"
        params["bounded"] = 1

    response = requests.get(url, params=params)
    return response.json()

# Example usage
addresses = search_address("300 m")
bounded_addresses = search_address("300 m", bounded_by=(-0.5, 51.2, 0.5, 51.8))

print(addresses)
print(bounded_addresses)
6/84:
import requests

def search_address(query, bounded_by=None):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": query,
        "format": "json",
        "addressdetails": 1,
        "limit": 5  # Adjust as needed
    }
    if bounded_by:
        params["viewbox"] = f"{bounded_by[0]},{bounded_by[1]},{bounded_by[2]},{bounded_by[3]}"
        params["bounded"] = 1

    response = requests.get(url, params=params)
    return response.json()

# Example usage
addresses = search_address("300 m")
bounded_addresses = search_address("300 m", bounded_by=(-0.5, 51.2, 0.5, 51.8))

print(addresses)
print(bounded_addresses)
6/85:
import requests

def search_address(query, bounded_by=None):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": query,
        "format": "json",
        "addressdetails": 1,
        "limit": 5
    }
    if bounded_by:
        params["viewbox"] = f"{bounded_by[0]},{bounded_by[1]},{bounded_by[2]},{bounded_by[3]}"
        params["bounded"] = 1

    headers = {
        "User-Agent": "MyApp/1.0 (your_email@example.com)"  # Replace with your details
    }

    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()  # Raise an exception for HTTP errors
    return response.json()

# Example usage
addresses = search_address("300 m")
bounded_addresses = search_address("300 m", bounded_by=(-0.5, 51.2, 0.5, 51.8))

print(addresses)
print(bounded_addresses)
6/86:
addresses = search_address("300 m")
bounded_addresses = search_address("300 m", bounded_by=(-0.5, 51.2, 0.5, 51.8))

print(addresses)
print(bounded_addresses)
6/87:     dc_viewbox = "-77.119759,38.995548,-76.909393,38.803574"
6/88:
addresses = search_address("300 m")
bounded_addresses = search_address("300 m", bounded_by=([point for point in dc_viewbox.xplit(',')]))

print(addresses)
print(bounded_addresses)
6/89:
addresses = search_address("300 m")
bounded_addresses = search_address("300 m", bounded_by=([point for point in dc_viewbox.split(',')]))

print(addresses)
print(bounded_addresses)
6/90: print(bounded_addresses)
6/91:
for addr in bounded_addresses:
    print(addr)
    break
6/92:
for addr in bounded_addresses:
    print(addr['address'])
    break
6/93:
for addr in bounded_addresses:
    print(addr['address'])
6/94:
import requests

def search_address(query, bounded_by=None):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": query,
        "format": "json",
        "addressdetails": 1,
        "limit": 20
    }
    if bounded_by:
        params["viewbox"] = f"{bounded_by[0]},{bounded_by[1]},{bounded_by[2]},{bounded_by[3]}"
        params["bounded"] = 1

    headers = {
        "User-Agent": "MyApp/1.0 (your_email@example.com)"  # Replace with your details
    }

    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()  # Raise an exception for HTTP errors
    return response.json()
6/95:
addresses = search_address("300 m")
bounded_addresses = search_address("300 m", bounded_by=([point for point in dc_viewbox.split(',')]))

print(addresses)
print(bounded_addresses)
6/96:
for addr in bounded_addresses:
    print(addr['address'])
6/97:
addresses = search_address("300 m st")
bounded_addresses = search_address("300 m st", bounded_by=([point for point in dc_viewbox.split(',')]))
6/98:
for addr in bounded_addresses:
    print(addr['address'])
6/99:
addresses = search_address("300 m str")
bounded_addresses = search_address("300 m str", bounded_by=([point for point in dc_viewbox.split(',')]))
6/100:
for addr in bounded_addresses:
    print(addr['address'])
6/101:
for addr in addresses:
    print(addr['address'])
6/102:
for addr in bounded_addresses:
    print(addr['address'])
6/103: bounded_addresses = search_address("300 m street", bounded_by=([point for point in dc_viewbox.split(',')]))
6/104:
for addr in bounded_addresses:
    print(addr['address'])
6/105: bounded_addresses = search_address("300 m street ne", bounded_by=([point for point in dc_viewbox.split(',')]))
6/106:
for addr in bounded_addresses:
    print(addr['address'])
6/107: import requests
6/108: result = cer.compute()
6/109: requests.get('http://orbnode.com:8000/datasets/places-list-return').json()
6/110: places = place['place_name'] for place in requests.get('http://orbnode.com:8000/datasets/places-list-return').json().values()
6/111: places = [place['place_name'] for place in requests.get('http://orbnode.com:8000/datasets/places-list-return').json().values()]
6/112: places
6/113: len(places)
6/114: places_dict = place['place_name'] for place in requests.get('http://orbnode.com:8000/datasets/places-list-return').json()
6/115: places_dict = requests.get('http://orbnode.com:8000/datasets/places-list-return').json()
6/116: places_dict
6/117: places_dict.keys()
6/118: places_dict.keys()[0]
6/119: list(places_dict.keys())
6/120: list(places_dict.values())
6/121: places_dict[list(places_dict.values())[0]]
6/122: list(places_dict.values())[0]
6/123: list(places.values())[0]
6/124: list(places_dict.values())
6/125: list(places_dict.keys())
6/126: list(places_dict.keys())[0]
6/127: places_dict[list(places_dict.keys())[0]]
6/128: places_dict[list(places_dict.keys())[1]]
6/129: places_dict[list(places_dict.keys())[2]]
6/130: places_dict[list(places_dict.keys())[3]]
6/131: places_dict[list(places_dict.keys())[4]]
6/132: places_dict[list(places_dict.keys())[4]]
6/133: places_dict[list(places_dict.keys())[5]]
6/134: places
6/135: [list(places_dict.keys())[5]]places_dict[0]
6/136: places_dict[0]
6/137: places_dict
6/138: list(places_dict)
6/139: places_dict[list(places_dict.keys())[2]]
6/140: places_dict.keys()
6/141: places_dict.keys()[0]
6/142: list(places_dict.keys())[0]
6/143: list(places_dict.values())[0]
6/144: list(places_dict.values())[0]
6/145: list(places_dict.values())[0]
6/146: list(places_dict.values())[1]
6/147: list(places_dict.values())[2]
6/148: list(places_dict.values())[3]
6/149: list(places_dict.values())[4]
6/150: list(places_dict.values())[5]
6/151: list(places_dict.values())[6]
6/152: list(places_dict.values())[5]
6/153: list(places_dict.values())[5]
6/154: list(places_dict.values())[6]
6/155: places_dict = requests.get('http://orbnode.com:8000/datasets/places-list-return').json()
6/156: places_dict
6/157: docker ps -a
6/158: places_dict = requests.get('http://localhost:8000/datasets/places-list-return').json()
6/159: places_dict = requests.get('http://localhost:8000/datasets/places-list-return').json()
6/160: places_dict = requests.get('http://localhost:8000/datasets/places-list-return').json()
6/161: places_dict = requests.get('http://orbnode.com:8000/datasets/places-list-return').json()
6/162: places_dict
 7/1: import pandas as pd
 7/2: i for i in range(10)
 7/3: [i for i in range(10)]
 7/4: [{'user_name':i} for i in range(10)]
 7/5: [{'user_name':'User {}'.formayt(i)} for i in range(10)]
 7/6: [{'user_name':'User {}'.format(i)} for i in range(10)]
 7/7: db = [{'user_name':'User {}'.format(i)} for i in range(10)]
 7/8: item['user_name'] == 'User 5 'for item in dc
 7/9: [item['user_name'] == 'User 5 'for item in dc]
7/10: [item['user_name'] == 'User 5 'for item in db]
7/11: [item['user_name'] == 'User 5'for item in db]
7/12: db[[item['user_name'] == 'User 5'for item in db]]
7/13: df = pd.DataFrame(db)
7/14: df
7/15: df[user_name == 'User 5']
7/16: df['user_name' == 'User 5']
7/17: df[df['user_name'] == 'User 5']
7/18: df['user_name'] == 'User 5'
7/19: db = [{'user_name':'User {}'.format(i)} for i in range(10)]
7/20: df = pd.DataFrame(db)
7/21: import uuid
7/22: uuid.uuid4()
7/23: uuid.uuid5(uuid.NAMESPACE_DNS,'User 5')
7/24: uuid.uuid5(uuid.NAMESPACE_DNS,'User 6')
7/25: uuid.uuid5(uuid.NAMESPACE_DNS,'User 5')
7/26: db = [{'user_name':'User {}'.format(i), '{}'.format(uuid.uuid4())} for i in range(10)]
7/27: db = [{'user_name':'User {}'.format(i), '{}'.format(uuid.uuid4()} for i in range(10)]
7/28: db = [{'user_name':'User {}'.format(i), '{}'.format(uuid.uuid4())} for i in range(10)]
7/29: db = [{'user_name':'User {}'.format(i), 'post_id':'{}'.format(uuid.uuid4())} for i in range(10)]
7/30: db
7/31:
import base64
import datetime
7/32:
def encode_filename(username):
    timestamp = datetime.datetime.utcnow().isoformat()
    raw_data = f"{timestamp}|{username}".encode()
    encoded = base64.urlsafe_b64encode(raw_data).decode()
    return encoded
7/33: encode_filename('alex 1')
7/34: type(encode_filename('alex 1'))
7/35: encode_filename('alex 1')
7/36:
def decode_filename(encoded_str):
    decoded_data = base64.urlsafe_b64decode(encoded_str).decode()
    timestamp, username = decoded_data.split("|", 1)
    return timestamp, username
7/37: decode_filename(encode_filename('alex 1'))
7/38:
def encode_filename(username):
   timestamp = datetime.datetime.now(datetime.UTC).isoformat()
    raw_data = f"{timestamp}|{username}".encode()
    encoded = base64.urlsafe_b64encode(raw_data).decode()
    return encoded
7/39:
def encode_filename(username):
    timestamp = datetime.datetime.now(datetime.UTC).isoformat()
    raw_data = f"{timestamp}|{username}".encode()
    encoded = base64.urlsafe_b64encode(raw_data).decode()
    return encoded
7/40: decode_filename(encode_filename('alex 1'))
7/41: timestamp = datetime.datetime.now(datetime.UTC).isoformat()
7/42: datetime.datetime.now(datetime.UTC).isoformat()
7/43: datetime.datetime.now(datetime.UTC)
7/44: datetime.UTC
7/45: datetime.datetime.now(datetime.EST)
7/46: datetime
7/47: help(datetime)
7/48: datetime.datetime.now(datetime.EST)
7/49: datetime.datetime.now(datetime.UTC)
7/50:
def encode_filename(username):
    timestamp = datetime.datetime.now(datetime.UTC).isoformat()
    raw_data = f"{timestamp}|{username}".encode()
    encoded = base64.urlsafe_b64encode(raw_data).decode()
    return encoded
7/51:
def decode_filename(encoded_str):
    decoded_data = base64.urlsafe_b64decode(encoded_str).decode()
    timestamp, username = decoded_data.split("|", 1)
    return timestamp, username
7/52: decode_filename(encode_filename('alex 1'))
7/53: encode_filename('alex 1')
7/54: encode_filename('alex 2')
7/55: encode_filename('alex 1')
7/56: db = [{'user_name':'User {}'.format(i), 'post_id':'{}'.format(uuid.uuid4())} for i in range(10)]
7/57: db
7/58: db = [{'user_name':'User {}'.format(i), 'post_id':'{}'.format(encode_filename('alex 1'))} for i in range(10)]
7/59: db
7/60: df = pd.DataFrame(db)
7/61: df
7/62: df[0]
7/63: df.iloc[0]
7/64: df.iloc(0)
7/65: df.iloc[0]
7/66: df.iloc[10]
7/67: df.iloc[9]
7/68: df.iloc[9].post)id
7/69: df.iloc[9].post_id
7/70: df.iloc[7].post_id
7/71: db = [{'user_name':'User {}'.format(i), 'post_id':'{}'.format(encode_filename('User {}'.format(i)))} for i in range(10)]
7/72: df = pd.DataFrame(db)
7/73: df.iloc[7].post_id
7/74: df.iloc[9].post_id
7/75: for x,y in zip('MjAyNS0wMy0wMlQyMDo0NjowOS44NDcyMjgrMDA6MDB8VXNlciA3', 'MjAyNS0wMy0wMlQyMDo0NjowOS44NDcyNDQrMDA6MDB8VXNlciA5')
7/76: [x,y for x,y in zip('MjAyNS0wMy0wMlQyMDo0NjowOS44NDcyMjgrMDA6MDB8VXNlciA3', 'MjAyNS0wMy0wMlQyMDo0NjowOS44NDcyNDQrMDA6MDB8VXNlciA5')]
7/77: [x,y for x in zip('MjAyNS0wMy0wMlQyMDo0NjowOS44NDcyMjgrMDA6MDB8VXNlciA3', 'MjAyNS0wMy0wMlQyMDo0NjowOS44NDcyNDQrMDA6MDB8VXNlciA5')]
7/78: [x for x in zip('MjAyNS0wMy0wMlQyMDo0NjowOS44NDcyMjgrMDA6MDB8VXNlciA3', 'MjAyNS0wMy0wMlQyMDo0NjowOS44NDcyNDQrMDA6MDB8VXNlciA5')]
7/79: [x for x,y in zip('MjAyNS0wMy0wMlQyMDo0NjowOS44NDcyMjgrMDA6MDB8VXNlciA3', 'MjAyNS0wMy0wMlQyMDo0NjowOS44NDcyNDQrMDA6MDB8VXNlciA5')]
7/80: [x,y if x==y for x,y in zip('MjAyNS0wMy0wMlQyMDo0NjowOS44NDcyMjgrMDA6MDB8VXNlciA3', 'MjAyNS0wMy0wMlQyMDo0NjowOS44NDcyNDQrMDA6MDB8VXNlciA5')]
7/81: [if x==y x,y for x,y in zip('MjAyNS0wMy0wMlQyMDo0NjowOS44NDcyMjgrMDA6MDB8VXNlciA3', 'MjAyNS0wMy0wMlQyMDo0NjowOS44NDcyNDQrMDA6MDB8VXNlciA5')]
7/82: [x==y for x,y in zip('MjAyNS0wMy0wMlQyMDo0NjowOS44NDcyMjgrMDA6MDB8VXNlciA3', 'MjAyNS0wMy0wMlQyMDo0NjowOS44NDcyNDQrMDA6MDB8VXNlciA5')]
7/83: db = [{'user_name':'User {}'.format(i), 'post_id':'{}'.format(encode_filename('User {}'.format(i)))} for i in range(10)]
7/84: df = pd.DataFrame(db)
7/85: db
7/86: df['user_name == 'User 5'']
7/87: df['user_name' == 'User 5']
7/88: df['user_name']
7/89: df['user_name'=='User5']
7/90: df['user_name'=='User 5']
7/91: df['user_name' == 'User 5']
7/92: df[df['user_name'] == 'User 5']
7/93: df['user_name'] == 'User 5'
7/94: ['user_name']
7/95: df['user_name']
7/96: import requests
7/97: requests.get('http://orbnode.com:8000/datasets/places-list-return').json()
7/98: requests.get('http://orbnode.com:8000/datasets/places-list-return').json()[0]
7/99: list(requests.get('http://orbnode.com:8000/datasets/places-list-return').json())[0]
7/100: list(requests.get('http://orbnode.com:8000/datasets/places-list-return').json())
7/101: list(requests.get('http://orbnode.com:8000/datasets/places-list-return').json())
7/102: requests.get('http://orbnode.com:8000/datasets/places-list-return').json()
7/103: requests.get('http://orbnode.com:8000/datasets/places-list-return').json().values()
7/104: type(requests.get('http://orbnode.com:8000/datasets/places-list-return').json())
7/105: requests.get('http://orbnode.com:8000/datasets/places-list-return').json().values().to_list()
7/106: requests.get('http://orbnode.com:8000/datasets/places-list-return').json().values().toList()
7/107: type(requests.get('http://orbnode.com:8000/datasets/places-list-return').json())
7/108: requests.get('http://orbnode.com:8000/datasets/places-list-return').json().values()
7/109: df_p = pd.data(requests.get('http://orbnode.com:8000/datasets/places-list-return').json().values())
7/110: df_p = pd.DataFrame(requests.get('http://orbnode.com:8000/datasets/places-list-return').json().values())
7/111: df_p
7/112: df_p.place_name
7/113: list_p = list(requests.get('http://orbnode.com:8000/datasets/places-list-return').json().values())
7/114: list_p
7/115:
for item in list_p:
    item.update({'place_type':['coffee']})
7/116: list_p
7/117:
for item in range(len(list_p),2):
    list_p[item].update({'place_type':['coffee']})
7/118: list_p
7/119:
for item in range(len(list_p),2):
    list_p[item].update({'place_type':['tea']})
7/120: list_p
7/121:
for item in range(len(list_p),2):
    print(item)
7/122:
for item in range(0, len(list_p),2):
    print(item)
7/123:
for item in range(0, len(list_p),2):
    list_p[item].update({'place_type':['tea']})
7/124: list_p
7/125: list_p
7/126: df_p = pd.DataFrame(list_p)
7/127: df_p
7/128:
df_p['place_type' == 'tea]
'
7/129: df_p['place_type' == 'tea']
7/130: df_p['place_type'] == 'tea'
7/131: df[df_p['place_type'] == 'tea']
7/132: df_p[df_p['place_type'] == 'tea']
7/133: df_p['place_type']
7/134: df_p['tea' in df_p['place_type']]
7/135: 'tea' in df_p['place_type']
7/136: df_p['place_type']
7/137: df_p
7/138: df_p['place_type']
7/139: list_p
7/140: lambda: x+'tea' in list_p
7/141: check = lambda: x+'tea' in list_p
7/142: check('s')
7/143: check('s')
7/144: check(None)
7/145: check()
7/146: check = lambda x: x+'tea' in list_p
7/147: check()
7/148: check(None)
7/149: check('s')
7/150: check = lambda x: x in list_p
7/151: check('tea')
7/152: list_p
7/153: check = lambda x: x in item['place_type'] for item in list_p
7/154: check = lambda x: [x in item['place_type'] for item in list_p]
7/155: check'tea'
7/156: check('tea')
7/157: check('tea')df
7/158: df_p
7/159: df_p[df_p.apply(lambda:x search_place)in x]
7/160: search_place = 'tea'
7/161: df_p[df_p.apply(lambda:x search_place in x)]
7/162: df_p[df_p['place_type'].apply(lambda:x search_place in x)]
7/163: df_p[df_p['place_type'].apply(lambda x: search_place in x)]
7/164: search_place = 'coffee'
7/165: df_p[df_p['place_type'].apply(lambda x: search_place in x)]
7/166: search_place = 'tea'
7/167: search_place = 'coffee'
7/168: search_place = 'tea'
7/169: df_p[df_p['place_type'].apply(lambda x: search_place in x)]
7/170: ls
7/171: pwd
7/172: ls /
7/173: ls /mnt/
7/174: ls /orbnode-backend-fastapi/
7/175: history
7/176: df_p[df_p['place_type'].apply(lambda x: search_place in x)]
7/177: df_p = pd.DataFrame(list_p)
7/178:
for item in range(len(list_p),2):
    list_p[item].update({'place_type':['coffee']})
7/179: list_p = list(requests.get('http://orbnode.com:8000/datasets/places-list-return').json().values())
7/180: df = pd.DataFrame(db)
7/181: db = [{'user_name':'User {}'.format(i), 'post_id':'{}'.format(encode_filename('User {}'.format(i)))} for i in range(10)]
7/182: db
7/183: history
7/184: db
7/185: list_p
7/186: df_p[df_p['place_type'].apply(lambda x: search_place in x)]
7/187: df
7/188: df_p
7/189:
for item in range(len(list_p),2):
    list_p[item].update({'place_type':['coffee','tea']})
7/190: df
7/191: df_p
7/192: df_p = pd.DataFrame(list_p)
7/193:
for item in range(len(list_p),2):
    list_p[item].update({'place_type':['coffee']})
7/194:
for item in range(len(list_p),2):
    list_p[item].update({'place_type':['coffee','tea']})
7/195: df_p[df_p['place_type'].apply(lambda x: search_place in x)]
7/196: df_p
7/197:
for item in range(len(list_p),2):
    list_p[item].update({'place_type':['coffee','tea']})
7/198: list_p
7/199: list_p
7/200:
for item in range(len(list_p),2):
    list_p[item].update({'place_typef':['coffee','tea']})
7/201: list_p
7/202:
for item in range(len(list_p),2):
    list_p[item].update({'place_typedddf':['coffee','tea']})
7/203: list_p
7/204:
for item in range(len(list_p),2):
    list_p[item].update({'place_type':['coffee']})
7/205:
for item in range(len(list_p),2):
    list_p[item].update({'place_type':['coffeededdd']})
7/206: list_p
7/207:
for item in range(len(list_p),2):
    list_p[item].update({'place_type':['coffee']})
7/208: list_p
7/209: list_p[0]
7/210: list_p[-1]
7/211: list_p[-10]
7/212: list_p[-11]
7/213: list_p[-12]
7/214: list_p[0]
7/215:
for item in range(len(list_p),2):
    print(list_p[item])
7/216:
for item in range(len(list_p),2):
    print(item)
7/217: range(len(list_p),2)
7/218:
for item in range(len(list_p),2):
    print(item)
7/219:
for item in list(range(len(list_p),2)):
    print(item)
7/220: list(range(len(list_p),2))
7/221:
for item in range(0, len(list_p),2):
    list_p[item].update({'place_type':['coffee', 'tea']})
7/222: df_p = pd.DataFrame(list_p)
7/223: df_p[df_p['place_type'].apply(lambda x: search_place in x)]
7/224: search_place = 'coffee'
7/225: df_p[df_p['place_type'].apply(lambda x: search_place in x)]
7/226: search_place = 'tea'
7/227: df_p[df_p['place_type'].apply(lambda x: search_place in x)]
7/228: search_place = 'biscuits'
7/229: df_p[df_p['place_type'].apply(lambda x: search_place in x)]
7/230: search_place = 'tea'
7/231: search_place = 'biscuits'
7/232: search_place = 'tea'
7/233: df_p[df_p['place_type'].apply(lambda x: search_place in x)]
7/234: df_p[df_p['place_type'].apply(lambda x: search_place in x)].iloc[0]
7/235: df_p[df_p['place_type'].apply(lambda x: search_place in x)].iloc[0]/place_type
7/236: df_p[df_p['place_type'].apply(lambda x: search_place in x)].iloc[0].place_type
7/237: df
7/238: db
7/239: df_p
7/240: list_p
7/241: df_p
7/242: db = [{'user_name':'User {}'.format(i), 'post_id':'{}'.format(encode_filename('User {}'.format(i)))} for i in range(10)]
7/243: db
7/244: df
7/245: df_p
7/246: df
7/247: df_p[df_p['place_name'].apply(lambda x: search_place in x)]
7/248: df_p[df_p['place_name'].apply(lambda x: 'blu' in x)]
7/249: df_p[df_p['place_name'].apply(lambda x: 'Blu' in x)]
7/250: df_p
7/251: df_p[df_p['place_name'].apply(lambda x: 'Blu' in x)]
7/252: df
7/253:
for item in range(0, len(db),2):
    list_p[item].update({'place_name':'tea-ny'})
7/254: list_p
7/255:
for item in range(0, len(db),2):
    db[item].update({'place_name':'tea-ny'})
7/256: db
7/257: df = pd.DataFrame(db)
7/258: df
7/259: df[df['place_name']=='tea-ny']
7/260: df[df['place_name']=='tea-ny']['post_id']
7/261: df_p[df_p['place_name'].apply(lambda x: 'tea' in x)]
7/262: df[df['place_name']=='tea-ny']['post_id']
7/263: df_p[df_p['place_name'].apply(lambda x: 'tea' in x)]
7/264: df[df['place_name']=='tea-ny']['post_id']
7/265: df[df['place_name']=='tea-ny']['post_id'].iloc[0]
7/266: df[df['place_name']=='tea-ny']['post_id']
7/267: df[df['place_name']=='tea-ny']['post_id'].iloc[0]
7/268: df_p[df_p['place_name'].apply(lambda x: 'tea' in x)]
7/269: history
7/270: df_p[df_p['place_name'].apply(lambda x: 'tea' in x)]
7/271: df[df['place_name']=='tea-ny']['post_id'].iloc[0]
7/272: df[df['place_name']=='tea-ny']['post_id']
7/273: df
7/274: df[df['place_name']=='tea-ny']
7/275: df[df['place_name']=='tea-ny']['post_id']
7/276: df_p
7/277: df
7/278: 'post_id':'{}'.format(encode_filename('User {}'.format(i)))
7/279: 'post_id'+'{}'.format(encode_filename('User {}'.format(i)))
7/280: 'post_id'+'{}'.format(encode_filename('User {}'.format(42)))
7/281: decode_filename('post_id'+'{}'.format(encode_filename('User {}'.format(42))))
7/282: 'post_id'+'{}'.format(encode_filename('User {}'.format(42)))
7/283: '{}'.format(encode_filename('User {}'.format(42)))
7/284: decode_filename('{}'.format(encode_filename('User {}'.format(42))))
 8/1: list_p = list(requests.get('http://orbnode.com:8000/datasets/places-list-return').json().values())
 8/2: import requests
 8/3: list_p = list(requests.get('http://orbnode.com:8000/datasets/places-list-return').json().values())
 8/4: list_p
 8/5:
for item in range(0, len(list_p),2):
    list_p[item].update({'place_type':['coffee', 'tea']})
 8/6: list_p
 8/7:
for item in range(0, len(list_p),0):
    list_p[item].update({'place_type':['coffee', 'tea']})
 8/8:
for item in range(0, len(list_p),1):
    list_p[item].update({'place_type':['coffee', 'tea']})
 8/9: list_p
8/10:
for item in range(0, len(list_p)):
    list_p[item].update({'place_address':'0000'})
8/11: list_p
8/12: list_p = list(requests.get('http://orbnode.com:8000/datasets/places-list-return').json().values())
8/13: list_p
8/14: clear
8/15: list_p
8/16: df
8/17: df_p
8/18: df = pd.DataFRame(list_p)
8/19: import pandas as pd
8/20: df = pd.DataFRame(list_p)
8/21: df = pd.DataFrame(list_p)
8/22: df = pd.DataFRame(list_p)
8/23: df = pd.DataFrame(list_p)
8/24: df
8/25: clear
8/26: df.head()
8/27: clear
8/28: clear
8/29: df
8/30: df.items
8/31: df.head
8/32: df.head()
8/33: import geopandas as gpd
8/34:
gdf = gpd.GeoDataFram({
  "type": "Feature",
  "geometry": {
    "type": "Point",
    "coordinates": [2.2945, 48.8584]  // [longitude, latitude]
  },
  "properties": {
    "name": "Eiffel Tower"
  }
}
)
8/35:
gdf = gpd.GeoDataFrame.from_features(        {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [2.2945, 48.8584]
            },
            "properties": {
                "name": "Eiffel Tower"
            }
        }
        )
8/36:
gdf = gpd.GeoDataFrame.from_features(        {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [2.2945, 48.8584]
            },
            "properties": {
                "name": "Eiffel Tower"
            }
        })
8/37:
gdf = gpd.GeoDataFrame.from_features(    [
    {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [2.2945, 48.8584]
            },
            "properties": {
                "name": "Eiffel Tower"
            }
        }])
8/38: gdf
8/39: clear
8/40: gdf
8/41: list_p = list(requests.get('http://orbnode.com:8000/datasets/places-list-return').json().values())
8/42: list_p
8/43: list_p = list(requests.get('http://orbnode.com:8000/datasets/places-list-return').json().leys())
8/44: list_p = list(requests.get('http://orbnode.com:8000/datasets/places-list-return').json().keys())
8/45: list_p
8/46: list_p = list(requests.get('http://orbnode.com:8000/datasets/places-list-return').json().values())
8/47: list_p
8/48: list_p[0]
8/49: list(requests.get('http://orbnode.com:8000/datasets/places-list-return').json())
8/50: requests.get('http://orbnode.com:8000/datasets/places-list-return').json()
8/51: type(requests.get('http://orbnode.com:8000/datasets/places-list-return').json())
8/52: places = requests.get('http://orbnode.com:8000/datasets/places-list-return').json()
8/53: places = requests.get('http://orbnode.com:8000/datasets/places-list-return').json()
8/54: places
8/55: places[0]
8/56:
for key, value in places:
    print(key, value)
    break
8/57:
for key, value in places.items():
    print(key, value)
    break
8/58:
for key, value in places.items():
    print(key)
    print(value)
8/59: dataframe_list = []
8/60:
for key, value in places.items():
    dataframe_list.append(
    {
    'uuid':key,
    'place_name':value.place_name
    }
    )
    break
8/61: places
8/62:
for key, value in places.items():
    dataframe_list.append(
    {
    'uuid':key,
    'place_name':value
    }
    )
    break
8/63: dataframe_list
8/64:
for key, value in places.items():
    dataframe_list.append(
    {
    'uuid':key,
    'place_name':value['place_name']
    }
    )
    break
8/65: dataframe_list
8/66: dataframe_list = []
8/67:
for key, value in places.items():
    dataframe_list.append(
    {
    'uuid':key,
    'place_name':value['place_name']
    }
    )
    break
8/68: dataframe_list
8/69: dataframe_list = []
8/70:
for key, value in places.items():
    dataframe_list.append(
    {
    'uuid':key,
    'place_name':value['place_name'],
    'place_address':value['place_address'],
    'geometry':Point(value.get('place_longitude'), value.get('place_latitude'))
    }
    )
8/71: from shapely.geometry import Point
8/72:
for key, value in places.items():
    dataframe_list.append(
    {
    'uuid':key,
    'place_name':value['place_name'],
    'place_address':value['place_address'],
    'geometry':Point(value.get('place_longitude'), value.get('place_latitude'))
    }
    )
8/73: places.items()
8/74: places = requests.get('http://orbnode.com:8000/datasets/places-list-return').json()
8/75: dataframe_list = []
8/76:
for key, value in places.items():
    dataframe_list.append(
    {
    'uuid':key,
    'place_name':value['place_name'],
    'place_address':value['place_address'],
    'geometry':Point(value.get('place_longitude'), value.get('place_latitude'))
    }
    )
8/77: dataframe_list
8/78: import geopandas as gpd
8/79: gdf_places = gpd.GeoDataFrame(dataframe_list, geometry='geometry', crs='EPSG:4326')
8/80: gdf_places
8/81: places = requests.get('http://orbnode.com:8000/datasets/places-list-return').json()
8/82: places
 9/1: import geopandas as gpd
 9/2: from shapely.geometry import Point
 9/3: import requests
 9/4: places = requests.get('http://orbnode.com:8000/datasets/places-list-return').json()
 9/5: dataframe_list = []
 9/6:
for key, value in places.items():
    dataframe_list.append(
    {
    'uuid':key,
    'place_name':value['place_name'],
    'place_address':value['place_address'],
    'geometry':Point(value.get('place_longitude'), value.get('place_latitude'))
    }
    )
 9/7: dataframe_list
 9/8: gdf_places = gpd.GeoDataFrame(dataframe_list, geometry='geometry', crs='EPSG:4326')
 9/9: gdf_places
9/10: places = requests.get('http://orbnode.com:8000/datasets/places-list-return').json()
9/11:
for key, value in places.items():
    dataframe_list.append(
    {
    'uuid':key,
    'place_name':value['place_name'],
    'place_address':value['place_address'],
    'geometry':Point(value.get('place_longitude'), value.get('place_latitude'))
    }
    )
9/12: gdf_places
9/13: places = requests.get('http://orbnode.com:8000/datasets/places-list-return').json()
9/14: places
9/15:
for key, value in places.items():
    dataframe_list.append(
    {
    'uuid':key,
    'place_name':value['place_name'],
    'place_address':value['place_address'],
    'geometry':Point(value.get('place_longitude'), value.get('place_latitude'))
    }
    )
9/16: gdf_places = gpd.GeoDataFrame(dataframe_list, geometry='geometry', crs='EPSG:4326')
9/17: dataframe_list
9/18: dataframe_list = []
9/19: dataframe_list
9/20:
for key, value in places.items():
    dataframe_list.append(
    {
    'uuid':key,
    'place_name':value['place_name'],
    'place_address':value['place_address'],
    'geometry':Point(value.get('place_longitude'), value.get('place_latitude'))
    }
    )
9/21: gdf_places = gpd.GeoDataFrame(dataframe_list, geometry='geometry', crs='EPSG:4326')
9/22: gdf_places
9/23: places = requests.get('http://orbnode.com:8000/datasets/places-list-return').json()
9/24: places
9/25: !pwd
9/26: !ls /
9/27: !ls /dataframe_list
9/28: !ls /
9/29: !ls /home/
9/30: !ls
9/31: !lpwd
9/32: !pwd
9/33: !ls
9/34: !ls /dataframe_list
9/35: gdf_places
9/36: %history -f /datasets/ipython_geo_ds.py
9/37: %history -f /dipython_geo_ds.py
9/38: ls
9/39: ls /
9/40: %history -f /dipython_geo_ds.py
9/41: %history -f /dipython_geo_ds.py
9/42: ls
9/43: gdf_places
9/44: places = requests.get('http://orbnode.com:8000/datasets/places-list-return').json()
9/45: places
9/46: gdf_places.to_json()
9/47: places
9/48: gdf_places.to_json()
9/49: gdf_places
9/50: gdf_places[0]
9/51: gdf_places.iloc(0)
9/52: gdf_places.iloc[0]
9/53: gdf_places.iloc[0].geometry
9/54: gdf_places.iloc[0].geometry[0]
9/55: gdf_places.iloc[0].geometry
9/56: gdf_places.iloc[0]['.geometry']
9/57: gdf_places.iloc[0]['geometry']
9/58: gdf_places.iloc[0].geometry
9/59: gdf_places.iloc[0]
9/60: gdf_places.iloc[0].geometry
9/61: gdf_places.iloc[0].geometry.lat
9/62: places = requests.get('http://orbnode.com:8000/datasets/places-list-return').json()
9/63: places
9/64: gdf_places.to_json()
9/65: gdf_places
9/66: gdf.geometry.x
9/67: gdf_places.geometry.x
9/68: gdf_places.geometry.y
9/69: gdf_places = gpd.GeoDataFrame(dataframe_list, geometry='geometry', crs='EPSG:4326')
9/70: dataframe_list
9/71: places = requests.get('http://orbnode.com:8000/datasets/places-list-return').json()
9/72: places
9/73: !curl 'http://orbnode.com:8000/datasets/places-list-return'
9/74: !curl 'http://localhost:8000/datasets/places-list-return'
9/75: !curl 'http://localhost:8000/datasets/places-list-return'
9/76: !curl 'http://localhost:8000
9/77: !curl 'http://localhost:8000'
9/78:
curl -X 'GET' \
  'http://localhost:8000/datasets/places-list-return' \
  -H 'accept: application/json'
9/79:
!curl -X 'GET' \
  'http://localhost:8000/datasets/places-list-return' \
  -H 'accept: application/json'
9/80: !curl 'http://orbnode.com:8000/datasets/places-list-return'
9/81: !curl 'http://localhost.com:8000/datasets/places-list-return'
9/82: !curl 'http://localhost.com:8000/datasets/places-list-return'
9/83: !curl 'http://localhost.com:8000'
9/84: !curl 'http://localhost:8000'
9/85: !curl 'http://localhost:8000/datasets/places-list-return'
9/86:
http://host.docker.internal:8000/datasets/places-list-return
http://host.docker.internal:8000/datasets/places-list-return
9/87: http://host.docker.internal:8000/datasets/places-list-return
9/88: http://host.docker.internal:8000/datasets/places-list-return
9/89: !curl 'http://host.docker.internal:8000/datasets/places-list-return'
9/90: !curl 'orbnode-backend-fastapi-container:8000/datasets/places-list-return'
9/91: !curl 'http://172.17.0.4:8000/datasets/places-list-return'
9/92: places_dev = requests.get('http://localhost:8000/datasets/places-list-return').json()
9/93: places_dev = requests.get('http://172.17.04.0:8000/datasets/places-list-return').json()
9/94: places_dev = requests.get('http://172.17.0.04:8000/datasets/places-list-return').json()
9/95: places_dev
9/96: places_dev
9/97: places_dev = requests.get('http://172.17.0.04:8000/datasets/places-list-return').json()
9/98: places_dev
9/99: places = requests.get('http://orbnode.com:8000/datasets/places-list-return').json()
9/100: places
9/101: docker ps -a
10/1: %history -f /ipython_geo_ds.py
11/1: clear
11/2: %history -f /datasets/ipython_geo_ds.py
11/3: %history -f ./ipython_geo_ds.py
11/4: ls
11/5: %history -f ./ipython_geo_ds.py
11/6: pwd
11/7: ls
11/8: rm ipython_geo_ds.py
11/9: %history -f./ipython_geo_ds.py
11/10: rm ipython_geo_ds.py
11/11: ls
11/12: %history -f /ipython_geo_ds.py
11/13: ls
11/14: ls /
11/15: %history -f ./ipython_geo_ds.py
11/16: pwd
11/17: ls /
11/18: %history -f /ipython_geo_ds.py
11/19: ls /
11/20: rm /dipython_geo_ds.py
11/21: ls /
11/22: %history -f /ipython_geo_ds.py
11/23: ls
11/24: rm ipython_geo_ds.py
11/25: ls
11/26: ls /
11/27: %history -f ./ipython_geo_ds.py
11/28: ls
11/29: pwd
12/1: places = requests.get('http://orbnode.com:8000/datasets/places-list-return').json()
12/2:
!curl -X 'GET' \
  'http://localhost:8000/datasets/places-list-return' \
  -H 'accept: application/json'
12/3: !curl 'http://172.17.0.4:8000/datasets/places-list-return'
12/4:
!curl -X 'GET' \
  'http://172.17.0.4:8000/datasets/places-list-return' \
  -H 'accept: application/json'
12/5: gdf_places = gpd.GeoDataFrame(dataframe_list, geometry='geometry', crs='EPSG:4326')
12/6: import requests
12/7: import geopandas as gpd
12/8: dataframe_list = []
12/9: gdf_places = gpd.GeoDataFrame(dataframe_list, geometry='geometry', crs='EPSG:4326')
12/10: places = requests.get('http://orbnode.com:8000/datasets/places-list-return').json()
12/11: gdf_places = gpd.GeoDataFrame(dataframe_list, geometry='geometry', crs='EPSG:4326')
12/12:
for key, value in places.items():
    dataframe_list.append(
    {
    'uuid':key,
    'place_name':value['place_name'],
    'place_address':value['place_address'],
    'geometry':Point(value.get('place_longitude'), value.get('place_latitude'))
    }
    )
12/13: from shapely.geometry import Point
12/14:
for key, value in places.items():
    dataframe_list.append(
    {
    'uuid':key,
    'place_name':value['place_name'],
    'place_address':value['place_address'],
    'geometry':Point(value.get('place_longitude'), value.get('place_latitude'))
    }
    )
12/15: dataframe_list
12/16: requests.get('http://orbnode.com:8000/datasets/places-list-return').json()
12/17: requests.get('http://orbnode.com:8000/datasets/places-list-return')
12/18: gdf_places = gpd.GeoDataFrame(dataframe_list, geometry='geometry', crs='EPSG:4326')
12/19: gdf_places
   1: %history -g -f /ipython_geo_ds.py
   2: %history -g -f ipython_geo_ds.py
