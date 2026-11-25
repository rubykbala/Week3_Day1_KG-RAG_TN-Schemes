import json
import os
import requests
from bs4 import BeautifulSoup

# Target URL for the Tamil Nadu schemes listing page
URL = "https://www.tn.gov.in/scheme_list.php?dep_id=Mg=="

# Browser-like headers so government portals return the full HTML page
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

response = requests.get(URL, headers=headers, timeout=20)
print(f"HTTP Status: {response.status_code} | HTML Length: {len(response.text)}")
response.raise_for_status()

soup = BeautifulSoup(response.text, "html.parser")

for tag in soup(["script", "style", "nav", "footer"]):
    tag.decompose()

# Fall back to the whole soup object if <main> and <body> are missing
main_content = soup.find("main") or soup.body or soup

rows = []
seen_texts = set()

for idx, el in enumerate(main_content.find_all(["tr", "li", "article", "div", "p"]), start=1):
    text = " ".join(el.get_text(" ", strip=True).split())
    links = [a.get("href") for a in el.find_all("a", href=True)]

    if len(text) > 40 and text not in seen_texts:
        seen_texts.add(text)
        rows.append({
            "id": f"item_{idx}",
            "source_url": URL,
            "text": text,
            "links": links,
        })

os.makedirs("data", exist_ok=True)
output_path = "data/schemes_raw_v2.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(rows, f, indent=2, ensure_ascii=False)

print(f"Saved {len(rows)} unique items to {output_path}")