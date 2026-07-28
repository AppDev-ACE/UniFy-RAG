import re

CONTACTS = {
    "hostel": [{"label": "Dean, SWAS", "email": "deanswas@sastra.edu"}],
    "transport": [{"label": "Transport office", "email": "transport@sastra.ac.in"}],
    "admissions": [{"label": "Admissions", "email": "admissions@sastra.edu", "phone": "+91 4362 264101"}],
    "medical": [{"label": "Campus hospital", "note": "Contact the campus hospital or local emergency services immediately."},
                {"label": "Tele-MANAS", "phone": "14416"}],
    "ragging": [{"label": "Anti-Ragging Committee", "note": "Report this directly to the Anti-Ragging Committee or a trusted university official immediately."}],
    "general": [{"label": "Admissions", "email": "admissions@sastra.edu", "phone": "+91 4362 264101"},
                {"label": "SASTRA website", "url": "https://www.sastra.edu"}],
}

HIGH_RISK_PATTERNS = {
    "ragging": r"\b(ragging|ragged|harass(?:ment|ed)?|bully(?:ing|ied)?)\b",
    "medical": r"\b(suicid(?:e|al)|self[ -]?harm|panic attack|overdose|medical emergency|want to die)\b",
    "personal_record": r"\b(my|mine)\b.*\b(marks?|attendance|fee|fees|fine|disciplinary|discipline|balance)\b|\b(my marks|my attendance|my fee)\b",
    "staff_contradiction": r"\b(warden|faculty|professor|staff|dean)\b.*\b(said|told|says|tells|contradict)\b|\bcontradict\w*\b",
}

def mandatory_route(query: str):
    text = query.lower()
    for kind, pattern in HIGH_RISK_PATTERNS.items():
        if re.search(pattern, text):
            if kind == "personal_record":
                return "general", "I can’t access or decide individual student records. Please contact the relevant university office."
            if kind == "staff_contradiction":
                return "general", "Please follow the warden’s or faculty member’s instruction. They are the authoritative source for your case."
            if kind == "medical":
                return "medical", "This needs human support now. Please contact the campus hospital or local emergency services immediately."
            return kind, "Please report this directly to the Anti-Ragging Committee or a trusted university official immediately."
    return None

def abstention(category: str | None = None, message: str | None = None):
    category = category if category in CONTACTS else "general"
    return {
        "status": "abstained",
        "message": message or "I don't have a verified answer for that. Please contact the appropriate university office or check the SASTRA website.",
        "contacts": CONTACTS[category],
    }
