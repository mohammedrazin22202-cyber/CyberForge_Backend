import json
import os
import random
import re
import time
import threading
import unicodedata
from difflib import SequenceMatcher

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__, static_folder=None)
CORS(app, resources={r"/*": {"origins": "*"}})

# ---------------------------------------------------------------------------
# Rate limiting — in-memory, no Redis required for a personal portfolio site.
# /chat        : 30 req/min per IP  — generous for real users, blocks bursts
# /suggestions : 60 req/min per IP  — fires on every keystroke, needs headroom
# On breach: 429 Too Many Requests with Retry-After header.
# ---------------------------------------------------------------------------
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=[],       # no global limit — applied explicitly per route
    storage_uri='memory://',
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset.json')
RESPONSES_DIR = os.path.join(BASE_DIR, 'responses')
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'frontend'))
if not os.path.exists(FRONTEND_DIR):
    FRONTEND_DIR = os.path.join(BASE_DIR, 'frontend')

with open(DATASET_PATH, 'r', encoding='utf-8') as f:
    DATASET = json.load(f)

STOP_WORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'can', 'could', 'do', 'does',
    'for', 'from', 'give', 'have', 'how', 'i', 'in', 'is', 'it', 'me', 'my',
    'of', 'on', 'or', 'please', 'show', 'tell', 'the', 'this', 'to', 'u',
    'ur', 'what', 'when', 'where', 'who', 'why', 'with', 'you', 'your',
    # 'about' causes false containment matches (e.g. "tell me about python"
    # becomes a substring of many "tell me about X" key_sentences)
    'about',
    # Semantically empty task-verbs — alone they carry no intent,
    # so single-token queries like "can u help me" → fallback correctly.
    'help', 'know', 'see', 'get', 'find', 'look', 'need', 'want', 'like',
    # 'me' used as a keyword on some entries causes spurious matches
    'me',
}

TOKEN_SYNONYMS = {
    # resume
    'cv': 'resume',
    'biodata': 'resume',
    'curriculum': 'resume',
    'vitae': 'resume',
    # contact
    'mail': 'email',
    'gmail': 'email',
    'message': 'contact',
    'call': 'phone',
    'reach': 'contact',
    'connect': 'contact',
    'ping': 'contact',
    'dm': 'contact',
    # github
    'repo': 'github',
    'repository': 'github',
    'repositories': 'github',
    'codebase': 'github',
    'sourcecode': 'github',
    'code': 'github',
    # portfolio
    'site': 'portfolio',
    'website': 'portfolio',
    'webpage': 'portfolio',
    'page': 'portfolio',
    'web': 'portfolio',
    # technology
    'technologies': 'technology',
    'tech': 'technology',
    'stack': 'technology',
    'tool': 'technology',
    'tools': 'technology',
    'framework': 'technology',
    'frameworks': 'technology',
    'language': 'technology',
    'languages': 'technology',
    # project
    'projects': 'project',
    'work': 'project',
    'works': 'project',
    'build': 'project',
    'built': 'project',
    'app': 'project',
    'apps': 'project',
    'application': 'project',
    'applications': 'project',
    'creation': 'project',
    # skills
    'libraries': 'library',
    'abilities': 'skill',
    'ability': 'skill',
    'capability': 'skill',
    'capabilities': 'skill',
    'expertise': 'skill',
    'competency': 'skill',
    'proficiency': 'skill',
    'strengths': 'skill',
    'strength': 'skill',
    'knowledge': 'skill',
    # identity / intro
    'introduce': 'intro',
    'introduction': 'intro',
    'background': 'intro',
    'bio': 'intro',
    'profile': 'intro',
    'yourself': 'intro',
    # job / hiring
    'hire': 'job',
    'hiring': 'job',
    'recruit': 'job',
    'recruitment': 'job',
    'position': 'job',
    'role': 'job',
    'opportunity': 'job',
    'employment': 'job',
    'career': 'job',
    'internship': 'job',
    # learning
    'studying': 'learning',
    'learning': 'learning',
    'learned': 'learning',
    'study': 'learning',
    'explore': 'learning',
    'exploring': 'learning',
    'practicing': 'learning',
    'practice': 'learning',
    # data / ML
    'dataset': 'data',
    'datasets': 'data',
    'database': 'data',
    'ml': 'machine learning',
    'ai': 'artificial intelligence',
    'nlp': 'natural language processing',
    'dl': 'deep learning',
    'nn': 'neural network',
    # python
    'py': 'python',
    # motivation / personality
    'inspire': 'motivate',
    'inspired': 'motivate',
    'inspires': 'motivate',
    'inspiration': 'motivate',
    'motivated': 'motivate',
    'motivation': 'motivate',
    'passion': 'motivate',
    'passionate': 'motivate',
    'drive': 'motivate',
    # certifications
    'certification': 'certificate',
    'certifications': 'certificate',
    'certificates': 'certificate',
    'certified': 'certificate',
    'course': 'certificate',
    'courses': 'certificate',
    # achievements
    'achievement': 'achievements',
    'accomplishment': 'achievements',
    'accomplishments': 'achievements',
    'award': 'achievements',
    'awards': 'achievements',
}

# Maps regex patterns to a canonical phrase that already exists in key_sentences.
# This fires BEFORE tokenization, so paraphrases resolve to the same token set.
PARAPHRASE_MAP: list[tuple] = [
    # name-based identity — catches "who is razin", "who is mohammed razin",
    # "tell me about razin", "razin who", "razin background", etc.
    (re.compile(r'\b(who\s+is|who\'?s|about|tell.{0,10}about)\s+(h\.?\s*)?(mohammed\s+)?razin\b', re.I), 'who are you'),
    (re.compile(r'\brazin\b', re.I), 'who are you'),
    (re.compile(r'\bmohammed\s+razin\b', re.I), 'who are you'),
    # identity / intro
    (re.compile(r'\b(describe|explain|tell).{0,10}\b(yourself|you|who you are)\b'), 'who are you'),
    (re.compile(r'\b(your|ur)\s+(background|profile|bio|story|intro)\b'), 'who are you'),
    (re.compile(r'\bintroduce\s+(yourself|urself)\b'), 'introduce yourself'),
    # what do you do
    (re.compile(r'\bwhat\s+(can|do)\s+(you|u)\s+do\b'), 'what do you do'),
    (re.compile(r'\b(describe|explain).{0,10}\b(capabilities|what you do|your work|your role)\b'), 'what do you do'),
    (re.compile(r'\bwhat\s+(kind|type|sort)\s+of\s+(work|things)\b'), 'what do you do'),
    # skills
    (re.compile(r'\b(list|show|tell me|describe).{0,10}\b(skills|abilities|expertise|capabilities)\b'), 'what are your skills'),
    (re.compile(r'\bwhat\s+(can|are).{0,10}\b(you|ur)\s+(do|good at|capable)\b'), 'what are your skills'),
    (re.compile(r'\bhow\s+(good|skilled|experienced)\s+(are you|r u)\b'), 'what are your skills'),
    # technologies
    (re.compile(r'\b(which|what).{0,10}\b(tech|tools|frameworks|languages|stack)\b'), 'what technologies do you know'),
    (re.compile(r'\bwhat\s+do\s+you\s+(use|code|program)\s+in\b'), 'what technologies do you know'),
    (re.compile(r'\b(tech|language|framework)\s+(you|ur)\s+(know|use|prefer)\b'), 'what technologies do you know'),
    # projects
    (re.compile(r'\bwhat\s+(have|did)\s+you\s+(built|made|created|developed|worked on)\b'), 'show projects'),
    (re.compile(r'\b(show|list|display|see).{0,10}\b(your\s+)?(work|portfolio|apps|builds)\b'), 'show projects'),
    (re.compile(r'\bwhat\s+projects.{0,20}\b(built|made|worked|done)\b'), 'show projects'),
    # best / proud project
    (re.compile(r'\b(favorite|favourite|most proud|top|best).{0,10}\bproject\b'), 'what is your best project'),
    (re.compile(r'\bproject\s+(you.re most proud|that stands? out)\b'), 'what is your best project'),
    # VConnect
    (re.compile(r'\b(what|tell).{0,15}\bv\s*connect\b'), 'tell me about vconnect'),
    (re.compile(r'\bv\s*connect\s+(project|app|system|platform)\b'), 'tell me about vconnect'),
    # LogiSense
    (re.compile(r'\b(what|tell).{0,15}\blogi\s*sense\b'), 'tell me about logisense'),
    (re.compile(r'\blogi\s*sense\s+(project|app|system|platform)\b'), 'tell me about logisense'),
    # ML / model-specific
    (re.compile(r'\bclass\s+imbalance\b'), 'how do you handle class imbalance in your village welfare datasets'),
    (re.compile(r'\bbias.{0,2}variance\b'), 'explain the bias variance trade off'),
    (re.compile(r'\bdata\s+drift\b'), 'how would you detect data drift if your ml model starts performing poorly after 6 months'),
    (re.compile(r'\bfeature\s+selection\b'), 'describe your feature selection strategy'),
    (re.compile(r'\binference\s+latency\b'), 'how do you measure the inference latency of your ml models'),
    (re.compile(r'\bcold\s+start\s+problem\b'), 'explain the cold start problem if you were to add a recommendation engine to your portfolio'),
    # Python — catch "tell me about python", "tell me about your python use", bare "python", etc.
    (re.compile(r'\bdo\s+you\s+(code|program|work)\s+in\s+python\b'), 'why python'),
    (re.compile(r'\b(why|how|when).{0,10}\bpython\b'), 'why python'),
    (re.compile(r'\bpython\s+(experience|skills?|knowledge|expertise)\b'), 'why python'),
    (re.compile(r'\b(skills?|experience|knowledge|expertise)\s+in\s+python\b'), 'why python'),
    (re.compile(r'\b(tell|about|describe|explain).{0,20}\bpython\b'), 'why python'),
    (re.compile(r'^\s*python\s*$'), 'why python'),  # bare "python"
    # hiring / job — "i wanna hire you" must resolve to pitch, NOT google-trust entry
    (re.compile(r'\b(why should|reason to)\s+(we|i|us)\s+(hire|recruit|pick|choose)\s+you\b'), 'why should we hire you'),
    (re.compile(r'\bwhat\s+makes\s+you\s+(stand out|special|different|unique|better)\b'), 'why should we hire you'),
    (re.compile(r'\b(your value|value proposition|selling point)\b'), 'why should we hire you'),
    (re.compile(r'\b(i|we)\s+(wanna|want to|would like to|wish to)\s+hire\s+(you|u)\b'), 'why should we hire you'),
    (re.compile(r'\b(hire|recruit)\s+(you|u)\b'), 'why should we hire you'),
    (re.compile(r'\bwhy\s+should\s+(i|we|someone)\s+(choose|pick|trust|consider)\s+(you|u)\b'), 'why should we hire you'),
    # internships — "any internship openings" must route to internship entry
    (re.compile(r'\b(any|got|have|is there).{0,10}(internship|intern)\b'), 'are you open to internships'),
    (re.compile(r'\b(open|available).{0,10}(internship|intern)\b'), 'are you open to internships'),
    (re.compile(r'\binternship\b'), 'are you open to internships'),
    # availability / job search
    (re.compile(r'\b(are you|r u).{0,10}\b(looking|searching|open)\s+(for|to).{0,10}\b(job|work|role|position|opportunity)\b'), 'are you looking for a job'),
    (re.compile(r'\b(available|open)\s+(for|to)\s+(work|hire|internship|freelance)\b'), 'are you looking for a job'),
    (re.compile(r'\b(r u|are you)\s+(available|free|open)\b'), 'are you looking for a job'),
    # fav/best project — must route to best-project, not generic show-projects
    (re.compile(r'\b(fav|favorite|favourite|fave)\s+(project|build|app|work)\b'), 'what is your best project'),
    (re.compile(r'\b(coolest|sickest|dopest|best).{0,10}(project|build|thing)\b'), 'what is your best project'),
    # contact
    (re.compile(r'\bhow\s+(can|do)\s+i\s+(contact|reach|get in touch|message)\s+(you|u)\b'), 'contact information'),
    (re.compile(r'\b(your\s+)?(email|phone|number|contact)\s*(address|info|details)?\b'), 'contact information'),
    # resume
    (re.compile(r'\b(get|download|see|view|share).{0,10}\b(your\s+)?(resume|cv)\b'), 'download resume'),
    # learning
    (re.compile(r'\bwhat\s+(are you|r u)?\s*(currently\s+)?(studying|learning|exploring|working on)\b'), 'what are you currently learning'),
    (re.compile(r'\b(latest|current|recent)\s+(studies|learning|courses?|skill)\b'), 'what are you currently learning'),
    # motivation / inspiration
    (re.compile(r'\bwhat\s+(drives|motivates|inspires|keeps)\s+you\b'), 'what inspires you'),
    (re.compile(r'\bwhy\s+(tech|technology|coding|programming|software)\b'), 'why technology'),
    # career goal
    (re.compile(r'\b(where|what).{0,10}\b(career|professional|future)\s+(goal|plan|vision|aspiration)\b'), 'what is your career goal'),
    (re.compile(r'\bwhere\s+do\s+you\s+(see yourself|want to be)\b'), 'what is your career goal'),
    # data science / ML
    (re.compile(r'\bwhat\s+is\s+(data science|machine learning|deep learning|ai)\b'), 'what is data science'),
    (re.compile(r'\b(explain|describe|tell).{0,10}\b(machine learning|data science)\b'), 'what is machine learning'),
    # weaknesses / strengths
    (re.compile(r'\b(your\s+)?(greatest\s+)?(weakness|weaknesses)\b'), 'what are your weaknesses'),
    (re.compile(r'\b(your\s+)?(greatest\s+)?(strength|strengths)\b'), 'what are your strengths'),
    # Tamil transliteration (romanized Tamil — common in Chennai tech circles)
    (re.compile(r'\b(vanakkam|vannakam|vanakam)\b', re.I),            'hello'),
    (re.compile(r'\bnee\s+(yaar|yar|yare)\b', re.I),                  'who are you'),
    (re.compile(r'\bunna\s+(pathi|patri|pari)\b', re.I),              'tell me about yourself'),
    (re.compile(r'\b(unna|unga|ungalukku)\s+pathi\b', re.I),         'tell me about yourself'),
    (re.compile(r'\b(thiramaikal|thiranmai|thiranmaikal)\b', re.I),   'what are your skills'),
    (re.compile(r'\b(skills|skill)\s+(sollu|sollunga|sollu)\b', re.I),'what are your skills'),
    (re.compile(r'\b(projects?|project)\s+(sollu|kaatu|kaattu)\b', re.I), 'show projects'),
    (re.compile(r'\b(resume|cv)\s+(kudu|taa|thaa)\b', re.I),         'download resume'),
    (re.compile(r'\b(github|git)\s+(kaatu|kaattu|kaattungo)\b', re.I),'open github'),
    (re.compile(r'\b(linkedin)\s+(kaatu|kaattu|kaattungo)\b', re.I), 'open linkedin'),
    (re.compile(r'\b(vela|velai)\s+(vaippu|vaaipu)\b', re.I),        'are you looking for a job'),
    (re.compile(r'\b(hire|hire)\s+(pannu|pannuveengala|panlam)\b', re.I), 'why should we hire you'),
    (re.compile(r'\bpython\s+(theriyuma|therium|therinja)\b', re.I),  'why python'),
    (re.compile(r'\b(yaar|yar)\s+(nee|neenga)\b', re.I),             'who are you'),
    (re.compile(r'\bnee\s+(enna|yenna)\s+(pandra|panra|seiyra)\b', re.I), 'what do you do'),
]


# ---------------------------------------------------------------------------
# Multilingual phrase map
#
# Maps native-script phrases (Tamil, Hindi, Arabic) and their common
# transliterations to canonical English intents.
#
# Design principle: this is a *portfolio chatbot*, not a general translator.
# We cover the ~30 questions a recruiter or visitor would actually ask —
# greetings, identity, skills, projects, hiring, contact — in the three
# languages most relevant to Razin's Tamil-Nadu/India context.
#
# When a native-script match is found, the translated English phrase is
# returned directly to score_input, bypassing normalize_text's ASCII strip.
# A language tag is also returned so the response can include a brief
# acknowledgement in the user's language.
# ---------------------------------------------------------------------------

MULTILINGUAL_MAP: list[tuple[str, str, str]] = [
    # (regex pattern, english_canonical, lang_tag)

    # ── Tamil (Unicode script) ─────────────────────────────────────────────
    ('வணக்கம்',                    'hello',                      'ta'),
    ('ஹலோ',                        'hello',                      'ta'),
    ('நீ யார்',                    'who are you',                'ta'),
    ('நீங்கள் யார்',               'who are you',                'ta'),
    ('உங்களை பறி சொல்லுங்கள்',    'tell me about yourself',     'ta'),
    ('உன்னைப் பற்றி சொல்',        'tell me about yourself',     'ta'),
    ('உங்கள் பற்றி சொல்லுங்கள்',  'tell me about yourself',     'ta'),
    # Skills — திறன் and திறமை are both used (thiran / thiramai)
    ('உங்கள் திறன்கள்',            'what are your skills',       'ta'),
    ('திறன்கள் சொல்லுங்கள்',      'what are your skills',       'ta'),
    ('திறன்கள் என்ன',              'what are your skills',       'ta'),
    ('உங்கள் திறமைகள்',            'what are your skills',       'ta'),
    ('திறமைகள் என்ன',              'what are your skills',       'ta'),
    ('திறமை என்ன',                 'what are your skills',       'ta'),
    ('திறமை சொல்லுங்கள்',          'what are your skills',       'ta'),
    ('தொழில்நுட்பம்',              'what technologies do you know', 'ta'),
    ('என்ன தெரியும்',              'what technologies do you know', 'ta'),
    ('திட்டங்கள்',                 'show projects',              'ta'),
    ('திட்டங்கள் காட்டு',          'show projects',              'ta'),
    ('உங்கள் திட்டங்கள்',          'show projects',              'ta'),
    ('VConnect பற்றி சொல்',        'tell me about vconnect',     'ta'),
    ('VConnect என்ன',              'tell me about vconnect',     'ta'),
    # Resume — ரெஸ்யூம் and ரெஸ்யூமே are both common spellings
    ('விண்ணப்பம் தரவும்',          'download resume',            'ta'),
    ('ரெஸ்யூம் தா',                'download resume',            'ta'),
    ('ரெஸ்யூமே தரவும்',            'download resume',            'ta'),
    ('ரெஸ்யூமே',                   'download resume',            'ta'),
    ('GitHub காட்டு',              'open github',                'ta'),
    ('LinkedIn காட்டு',            'open linkedin',              'ta'),
    ('வேலை தேடுகிறீர்களா',         'are you looking for a job',  'ta'),
    ('வேலை வாய்ப்பு',              'are you looking for a job',  'ta'),
    ('ஏன் உங்களை தேர்வு செய்ய வேண்டும்', 'why should we hire you', 'ta'),
    ('Python ஏன்',                 'why python',                 'ta'),
    ('தொடர்பு கொள்ள',              'contact information',        'ta'),
    ('தொடர்பு',                    'contact information',        'ta'),

    # ── Hindi (Devanagari) ────────────────────────────────────────────────
    ('नमस्ते',                     'hello',                      'hi'),
    ('हेलो',                       'hello',                      'hi'),
    ('नमस्कार',                    'hello',                      'hi'),
    ('आप कौन हैं',                 'who are you',                'hi'),
    ('तुम कौन हो',                 'who are you',                'hi'),
    ('अपने बारे में बताएं',         'tell me about yourself',     'hi'),
    ('आपके बारे में बताएं',         'tell me about yourself',     'hi'),
    ('खुद के बारे में बताओ',        'tell me about yourself',     'hi'),
    ('आपके कौशल',                  'what are your skills',       'hi'),
    ('कौशल क्या हैं',               'what are your skills',       'hi'),
    ('आप क्या जानते हैं',           'what technologies do you know', 'hi'),
    ('तकनीक क्या है',               'what technologies do you know', 'hi'),
    ('प्रोजेक्ट दिखाओ',             'show projects',              'hi'),
    ('आपके प्रोजेक्ट',              'show projects',              'hi'),
    ('VConnect के बारे में',        'tell me about vconnect',     'hi'),
    ('रिज्यूमे दो',                 'download resume',            'hi'),
    ('रेज्यूमे डाउनलोड',            'download resume',            'hi'),
    ('GitHub दिखाओ',               'open github',                'hi'),
    ('नौकरी ढूंढ रहे हैं',          'are you looking for a job',  'hi'),
    ('नौकरी की तलाश',               'are you looking for a job',  'hi'),
    ('आपको क्यों रखें',             'why should we hire you',     'hi'),
    ('Python क्यों',               'why python',                 'hi'),
    ('संपर्क करें',                 'contact information',        'hi'),
    ('संपर्क जानकारी',              'contact information',        'hi'),

    # ── Arabic ────────────────────────────────────────────────────────────
    ('مرحبا',                      'hello',                      'ar'),
    ('أهلا',                       'hello',                      'ar'),
    ('السلام عليكم',               'hello',                      'ar'),
    ('من أنت',                     'who are you',                'ar'),
    ('أخبرني عن نفسك',             'tell me about yourself',     'ar'),
    ('مهاراتك',                    'what are your skills',       'ar'),
    ('ما هي مهاراتك',              'what are your skills',       'ar'),
    ('ما هي تقنياتك',              'what technologies do you know', 'ar'),
    ('أظهر المشاريع',              'show projects',              'ar'),
    ('مشاريعك',                    'show projects',              'ar'),
    ('عن VConnect',               'tell me about vconnect',     'ar'),
    ('حمل السيرة الذاتية',         'download resume',            'ar'),
    ('السيرة الذاتية',             'download resume',            'ar'),
    ('أظهر GitHub',               'open github',                'ar'),
    ('هل تبحث عن عمل',            'are you looking for a job',  'ar'),
    ('لماذا نوظفك',               'why should we hire you',     'ar'),
    ('معلومات التواصل',            'contact information',        'ar'),
    ('تواصل معي',                  'contact information',        'ar'),

    # ── Romanized Tamil (ASCII — typed by Tamil speakers who don't switch IME) ──
    # Patterns are word-boundary anchored so they don't fire on unrelated words.
    (r'\bvanakkam\b',              'hello',                      'ta'),
    (r'\bvanakam\b',               'hello',                      'ta'),
    (r'\bnamaskar(am)?\b',         'hello',                      'ta'),
    (r'\bneenga\s+yaar(u)?\b',     'who are you',                'ta'),
    (r'\bnee\s+yaar(u)?\b',        'who are you',                'ta'),
    (r'\bungal(a)?\s+pathi\b',     'tell me about yourself',     'ta'),
    (r'\bungal(a)?\s+(skills?|thiramaikal|thirankal|thimai)\b', 'what are your skills', 'ta'),
    (r'\bthiramaikal\b',           'what are your skills',       'ta'),
    (r'\bthirankal\b',             'what are your skills',       'ta'),
    (r'\b(ungal(a)?|unna)\s+project(s|kal)?\b', 'show projects', 'ta'),
    (r'\bproject(s|kal)?\s+kaat(tu|unge|inga)\b', 'show projects', 'ta'),
    (r'\bresume\s+(kudu|taa?|tharavu|thaa)\b', 'download resume', 'ta'),
    (r'\bgithub\s+kaat(tu|unge|inga)\b', 'open github',          'ta'),
    (r'\blinkedin\s+kaat(tu|unge|inga)\b', 'open linkedin',      'ta'),
    (r'\bvconnect\s+(pathi|patri)\b', 'tell me about vconnect',  'ta'),
    (r'\bthogumbu\b',              'contact information',        'ta'),
    (r'\bthodarbu\b',              'contact information',        'ta'),
    (r'\bvela(i)?\s+(thedu|thedi)\b', 'are you looking for a job', 'ta'),
    (r'\b(yen|yean)\s+ungalai\s+(therrvu|tharvu)\b', 'why should we hire you', 'ta'),

    # ── Romanized Hindi (ASCII — typed by Hindi speakers in Hinglish) ──────
    (r'\bnamaste\b',               'hello',                      'hi'),
    (r'\bnamaskar\b',              'hello',                      'hi'),
    (r'\bkya\s+haal\b',            'hello',                      'hi'),
    (r'\baap\s+kaun\s+(ho|hain)\b', 'who are you',               'hi'),
    (r'\btum\s+kaun\s+ho\b',       'who are you',                'hi'),
    (r'\bapne\s+baare\s+mein\b',   'tell me about yourself',     'hi'),
    (r'\baap(ke)?\s+(skills?|kaushal)\b', 'what are your skills', 'hi'),
    (r'\bkya\s+(skills?|kaushal)\b', 'what are your skills',     'hi'),
    (r'\btumhara\s+kya\s+kaam\b',  'what do you do',             'hi'),
    (r'\baap\s+kya\s+karte\b',     'what do you do',             'hi'),
    (r'\bproject(s)?\s+(dikhao|dikhaiye|dekhao)\b', 'show projects', 'hi'),
    (r'\bvconnect\s+(ke\s+baare|kya\s+hai)\b', 'tell me about vconnect', 'hi'),
    (r'\bresume\s+(do|dijiye|bhejo|de\s+do)\b', 'download resume', 'hi'),
    (r'\bgithub\s+(dikhao|dekho)\b', 'open github',              'hi'),
    (r'\bnaukri\s+(dhundh|chahiye|chahie)\b', 'are you looking for a job', 'hi'),
    (r'\b(aapko|tumhe)\s+(kyun|kyon)\s+(rakhe|hire)\b', 'why should we hire you', 'hi'),
    (r'\bmujhe\s+(hire|rakho|select)\b', 'why should we hire you', 'hi'),
    (r'\bsampark\b',               'contact information',        'hi'),
    (r'\bcontact\s+(karo|karein|kijiye)\b', 'contact information', 'hi'),
]

# Acknowledgement prefixes in each language — shown before the English response
# so the user knows their language was understood.
LANG_ACK: dict[str, str] = {
    'ta': '>> உங்கள் கேள்வி புரிந்தது. ஆங்கிலத்தில் பதில்:\n\n',
    'hi': '>> आपका प्रश्न समझ आया। अंग्रेज़ी में उत्तर:\n\n',
    'ar': '>> تم فهم سؤالك. الإجابة بالإنجليزية:\n\n',
}

# Pre-compile: build list of (compiled_re, english_canonical, lang_tag).
# Entries whose pattern starts with \b or ^ are already regex strings;
# plain text entries (script phrases) are escaped for literal matching.
# Sort longest-literal-first so specific phrases beat shorter partials.
def _is_regex(p: str) -> bool:
    return p.startswith((r'\b', r'\B', '^', '(', '(?'))

_MULTILINGUAL_COMPILED: list[tuple] = sorted(
    [
        (
            re.compile(phrase if _is_regex(phrase) else re.escape(phrase),
                       re.IGNORECASE | re.UNICODE),
            english,
            lang,
        )
        for phrase, english, lang in MULTILINGUAL_MAP
    ],
    key=lambda t: -len(t[0].pattern),   # longer patterns first
)


def translate_multilingual(raw_input: str) -> tuple[str, str] | None:
    """If raw_input contains a known multilingual phrase, return (english_canonical, lang_tag).

    Returns None if no match — caller falls through to normal English pipeline.

    Handles two input types:
    - Native script (Tamil/Hindi/Arabic Unicode): matched on the raw string.
    - Romanized transliteration (ASCII): matched via regex word-boundary patterns
      in the _MULTILINGUAL_COMPILED list whose pattern uses \\b anchors.
    """
    has_non_ascii = any(ord(c) > 127 for c in raw_input)

    for pattern, english, lang in _MULTILINGUAL_COMPILED:
        # For pure-ASCII inputs, only attempt patterns that use word boundaries
        # (romanized section) — skip literal Unicode phrase patterns entirely.
        if not has_non_ascii and r'\b' not in pattern.pattern:
            continue
        if pattern.search(raw_input):
            return english, lang
    return None


def apply_paraphrase_map(text: str) -> str:
    """Replace paraphrased input with its canonical equivalent before scoring."""
    for pattern, canonical in PARAPHRASE_MAP:
        if pattern.search(text):
            return canonical
    return text


WEAK_DUPLICATE_WORDS = {
    # Original set
    'action', 'balance', 'change', 'choice', 'design', 'flow', 'future',
    'goal', 'growth', 'help', 'history', 'logic', 'power', 'quality',
    'safety', 'speed', 'style', 'time', 'value', 'vision',
    # High-frequency dataset pollutants (appear in 8+ entries each)
    'performance', 'progress', 'impact', 'connection', 'plan', 'guard',
    'support', 'success', 'storage', 'background', 'status', 'existence',
    'data', 'hello', 'system', 'process', 'result', 'output', 'input',
    'method', 'approach', 'solution', 'problem', 'feature', 'function',
    'structure', 'model', 'layer', 'point', 'level', 'step', 'stage',
    'type', 'mode', 'state', 'form', 'base', 'core', 'main', 'key',
    'part', 'set', 'list', 'item', 'group', 'section', 'area', 'field',
    'way', 'case', 'use', 'user', 'option', 'version', 'update', 'control',
}

# ---------------------------------------------------------------------------
# IDF (Inverse Document Frequency) weighting
# A token appearing in many dataset entries carries little discriminative
# signal — e.g. 'project' is in 292 entries, 'vconnect' in only 1.
# idf(token) = log(N / df), normalised to [0, 1].
# Near 0  → ubiquitous, weak signal (e.g. 'data', 'project', 'logic')
# Near 1  → rare, strong signal     (e.g. 'vconnect', 'logisense', 'xgboost')
# Built after prepare_dataset() so tokenize() is already available.
# ---------------------------------------------------------------------------
import math as _math

TOKEN_IDF: dict[str, float] = {}   # populated by _build_token_idf() at startup


def _build_token_idf() -> dict[str, float]:
    N = len(DATASET)
    df: dict[str, int] = {}
    for entry in DATASET:
        seen: set[str] = set()
        for phrase in entry.get('keywords', []) + entry.get('key_sentences', []):
            for tok in tokenize(phrase):
                if tok not in seen:
                    df[tok] = df.get(tok, 0) + 1
                    seen.add(tok)
    result: dict[str, float] = {}
    for tok, freq in df.items():
        raw = _math.log(N / freq)                    # 0 when freq == N
        result[tok] = min(raw / _math.log(N), 1.0)  # normalise to [0, 1]
    return result


def token_weight(tok: str) -> float:
    """IDF weight: 0 = seen everywhere (weak), 1 = very rare (strong)."""
    return TOKEN_IDF.get(tok, 0.5)


ACTION_MAP = {
    '6.1.Show projects.txt': [
        {'label': 'View Projects', 'url': '/#projects'},
    ],
    '6.2.Open GitHub.txt': [
        {'label': 'Open GitHub', 'url': 'https://github.com/mohammedrazin22202-cyber', 'external': True},
    ],
    '6.3.Open LinkedIn.txt': [
        {'label': 'Open LinkedIn', 'url': 'https://www.linkedin.com/in/razin88307', 'external': True},
    ],
    '6.4.Download resume.txt': [
        {'label': 'Download Resume', 'url': '/My_Resume.pdf', 'download': 'Mohammed_Razin_Data_Analyst.pdf'},
    ],
    '6.5.Contact information.txt': [
        {'label': 'Email', 'url': 'mailto:mohammedrazin22202@gmail.com?subject=Reaching%20out%20from%20your%20Portfolio'},
        {'label': 'Call', 'url': 'tel:+919342234674'},
        {'label': 'LinkedIn', 'url': 'https://www.linkedin.com/in/razin88307', 'external': True},
    ],
    '6.6.Show certifications.txt': [
        {'label': 'View Certifications', 'url': '/#achievements'},
    ],
    '6.7.Show skills.txt': [
        {'label': 'View Skills', 'url': '/#skills'},
    ],
    '6.8.Open portfolio.txt': [
        {'label': 'Open Portfolio', 'url': '/'},
    ],
    '6.9.Show achievements.txt': [
        {'label': 'View Achievements', 'url': '/#achievements'},
    ],
    '6.10.Show experience.txt': [
        {'label': 'View Workflow', 'url': '/#workflow'},
    ],
    '2.4.Tell me about LogiSense.txt': [
        {'label': 'Open LogiSense', 'url': 'https://logisense-rc4l.onrender.com/', 'external': True},
    ],
}


# Pre-compiled regular expressions for text normalization
RE_NON_ALPHANUM = re.compile(r'[^a-zA-Z0-9]+')
RE_SPACES = re.compile(r'\s+')
RE_VCONNECT = re.compile(r'\bv\s+connect\b')
RE_LOGISENSE = re.compile(r'\blogi\s+sense\b')
RE_LINKEDIN = re.compile(r'\blinked\s+in\b')
RE_GITHUB = re.compile(r'\bgit\s+hub\b')

def normalize_text(value: str) -> str:
    text = unicodedata.normalize('NFKC', str(value or ''))
    text = text.replace('&', ' and ')
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = RE_NON_ALPHANUM.sub(' ', text.lower())
    text = RE_SPACES.sub(' ', text).strip()
    text = RE_VCONNECT.sub('vconnect', text)
    text = RE_LOGISENSE.sub('logisense', text)
    text = RE_LINKEDIN.sub('linkedin', text)
    text = RE_GITHUB.sub('github', text)
    return RE_SPACES.sub(' ', text).strip()


def canonical_token(token: str) -> str:
    token = TOKEN_SYNONYMS.get(token, token)
    if len(token) > 4 and token.endswith('ies'):
        token = token[:-3] + 'y'
    elif len(token) > 3 and token.endswith('s') and not token.endswith('ss'):
        token = token[:-1]
    return TOKEN_SYNONYMS.get(token, token)


def tokenize(value: str) -> list[str]:
    tokens = [canonical_token(token) for token in normalize_text(value).split()]
    meaningful = [token for token in tokens if token not in STOP_WORDS]
    return meaningful or tokens


def prepare_dataset() -> None:
    global TOKEN_IDF
    for entry in DATASET:
        entry['_key_sentences'] = [
            (sentence, normalize_text(sentence), set(tokenize(sentence)))
            for sentence in entry.get('key_sentences', [])
            if normalize_text(sentence)
        ]
        entry['_keywords'] = [
            (keyword, normalize_text(keyword), set(tokenize(keyword)))
            for keyword in entry.get('keywords', [])
            if normalize_text(keyword)
        ]
    # Build IDF table after all entries are tokenised
    TOKEN_IDF = _build_token_idf()
    
    # Precompute total IDF weight sum for key sentences to avoid repeat token weighting
    for entry in DATASET:
        entry['_key_sentence_weights'] = {
            norm: sum(token_weight(t) for t in tokens)
            for _, norm, tokens in entry['_key_sentences']
        }
    
    # Precompute total IDF weight sum for keywords
    for entry in DATASET:
        entry['_keyword_weights'] = {
            norm: sum(token_weight(t) for t in tokens)
            for _, norm, tokens in entry['_keywords']
        }


prepare_dataset()


def build_suggestions() -> list[str]:
    preferred = [
        'who are you',
        'what are your skills',
        'what technologies do you know',
        'show projects',
        'tell me about vconnect',
        'tell me about logisense',
        'download resume',
        'open github',
        'open linkedin',
        'contact information',
        'why should we hire you',
    ]

    # Patterns that mark mechanically-generated key_sentence compounds.
    # These are dataset artifacts from templated generation ("tell me about X",
    # "introduce X", "explain X") and make terrible autocomplete suggestions.
    _JUNK_PREFIX = re.compile(
        r'^(tell me about|introduce|explain|more about|what is|'
        r'can you explain|describe|overview of|details of|information about|'
        r'tell me more about|wat is|wats)\s+',
        re.I
    )
    # Trailing noise appended to otherwise-clean suggestions during generation.
    # Only applies when there are ≥3 words before the suffix so "contact information"
    # (2 words) is preserved but "what are your skills information" (5 words) is not.
    _JUNK_SUFFIX = re.compile(
        r'^(\S+\s+){2,}\S+\s+(overview|details|summary|description|information)$', re.I
    )

    def is_junk(s: str) -> bool:
        # Mechanically-prefixed compound with a multi-word remainder
        m = _JUNK_PREFIX.match(s)
        if m and len(s[m.end():].strip().split()) >= 2:
            return True
        # Trailing noise word on a clean phrase
        if _JUNK_SUFFIX.search(s):
            return True
        # Too short to be useful
        if len(s.split()) < 2:
            return True
        return False

    def idf_score(sentence: str) -> float:
        """Sum of canonical IDF weights, normalised by token count."""
        toks = tokenize(normalize_text(sentence))
        if not toks:
            return 0.0
        return sum(token_weight(t) for t in toks) / len(toks)

    seen_norms: set[str] = set()
    suggestions: list[str] = []

    # Preferred always come first, never filtered
    for item in preferred:
        seen_norms.add(normalize_text(item))
        suggestions.append(item)

    # From each dataset entry keep the top-3 clean key_sentences by IDF score
    # (most distinctive phrasings surface first, junk stays out)
    TOP_PER_ENTRY = 3
    for entry in DATASET:
        candidates = []
        for sentence in entry.get('key_sentences', []):
            if is_junk(sentence):
                continue
            norm = normalize_text(sentence)
            if not norm or norm in seen_norms or len(norm) <= 2:
                continue
            candidates.append((idf_score(sentence), sentence, norm))

        # Sort best-first, take top N
        candidates.sort(reverse=True)
        for _, sentence, norm in candidates[:TOP_PER_ENTRY]:
            seen_norms.add(norm)
            suggestions.append(sentence)

    return suggestions


SUGGESTIONS = build_suggestions()

# ---------------------------------------------------------------------------
# Conversation memory
# Stores per-session context so follow-up questions resolve correctly.
# e.g. "tell me about vconnect" then "what tech did you use for it?" works.
#
# Structure per session_id:
#   last_file      — file_name of the last matched response
#   last_category  — numeric prefix of that file (e.g. "2" for VConnect)
#   last_tokens    — meaningful tokens from the last user query
#   last_ts        — unix timestamp of last activity (for TTL cleanup)
#   turn_count     — how many turns this session has had
#
# Sessions older than SESSION_TTL seconds are purged by a background thread.
# ---------------------------------------------------------------------------

SESSION_TTL = 1800   # 30 minutes idle → session forgotten
SESSIONS: dict[str, dict] = {}
_SESSIONS_LOCK = threading.Lock()


def _file_category(file_name: str) -> str:
    """Return the numeric category prefix of a file, e.g. '2' for '2.1.Tell me about VConnect.txt'."""
    return file_name.split('.')[0] if file_name else ''


def get_session(session_id: str) -> dict:
    """Return the session context dict (empty dict if session is new or expired)."""
    if not session_id:
        return {}
    with _SESSIONS_LOCK:
        return dict(SESSIONS.get(session_id, {}))


def update_session(session_id: str, file_name: str, user_tokens: set) -> None:
    """Persist the result of a successful match into the session."""
    if not session_id:
        return
    with _SESSIONS_LOCK:
        existing = SESSIONS.get(session_id, {})
        SESSIONS[session_id] = {
            'last_file':     file_name,
            'last_category': _file_category(file_name),
            'last_tokens':   set(user_tokens),
            'last_ts':       time.time(),
            'turn_count':    existing.get('turn_count', 0) + 1,
            # Preserve rotation state so variants/hooks cycle correctly
            'variants':      existing.get('variants', {}),
            'hooks':         existing.get('hooks', {}),
        }


def update_session_variation(session_id: str, variants: dict, hooks: dict) -> None:
    """Persist variant and hook rotation choices back into the live session."""
    if not session_id:
        return
    with _SESSIONS_LOCK:
        s = SESSIONS.get(session_id)
        if s is not None:
            s['variants'] = variants
            s['hooks']    = hooks


def _purge_old_sessions() -> None:
    """Background thread: remove sessions that have been idle for SESSION_TTL seconds."""
    while True:
        time.sleep(300)   # check every 5 minutes
        cutoff = time.time() - SESSION_TTL
        with _SESSIONS_LOCK:
            stale = [sid for sid, s in SESSIONS.items() if s.get('last_ts', 0) < cutoff]
            for sid in stale:
                del SESSIONS[sid]


threading.Thread(target=_purge_old_sessions, daemon=True).start()

# Pronoun / vague follow-up words that signal the user is referring back to
# the previous topic rather than starting a fresh query.
CONTEXT_PRONOUNS = {
    'it', 'its', "it's", 'that', 'this', 'those', 'these', 'they', 'them',
    'there', 'the', 'same', 'more', 'another', 'further', 'again', 'also',
}

# Topic-expanding phrases: if the user says one of these AND has no strong
# own tokens, treat the last topic as the anchor.
CONTEXT_FOLLOW_UP_PATTERNS = re.compile(
    r'\b(tell me more|more (about|on|info)|elaborate|expand|go on|continue|'
    r'and (what|how|why|when)|what (else|about)|any (more|other)|'
    r'how (does|did|do) (it|that|this)|why (is|was|did) (it|that|this)|'
    r'(give|show) me (more|details|examples?))\b'
)

# Category labels for human-readable context hints in the response
CATEGORY_LABELS = {
    '1': 'identity',
    '2': 'projects',
    '3': 'Python & data',
    '4': 'AI & security',
    '5': 'career & hiring',
    '6': 'links & actions',
    '7': 'personal',
    '8': 'easter eggs',
    '9': 'system design',
    '10': 'ML & models',
    '11': 'security',
    '12': 'engineering',
    '13': 'soft skills',
}

INTENT_RULES = [
    (re.compile(r'^/help$|^/commands$|^\?+$|\b(help|commands|subroutines)\b'), '8.8.Help and Commands.txt', 'help intent'),
    (re.compile(r'\b(resume|cv)\b'), '6.4.Download resume.txt', 'resume action'),
    (re.compile(r'\b(github|source code|code repository|repo)\b'), '6.2.Open GitHub.txt', 'github action'),
    (re.compile(r'\b(linkedin)\b'), '6.3.Open LinkedIn.txt', 'linkedin action'),
    (re.compile(r'\b(contact|email|phone|call|hire me|reach)\b'), '6.5.Contact information.txt', 'contact action'),
    (re.compile(r'\b(open|view|visit|launch)\s+(portfolio|website|site)\b'), '6.8.Open portfolio.txt', 'portfolio action'),
    (re.compile(r'\b(best|proud)\s+project\b'), '2.7.What is your best project.txt', 'best project intent'),
    (re.compile(r'\b(show|view|list|open).*\b(project|projects)\b|\b(project|projects).*\b(built|made|created)\b'), '6.1.Show projects.txt', 'projects action'),
    (re.compile(r'\b(show|view|list|display).*\b(skill|skills)\b'), '6.7.Show skills.txt', 'skills action'),
    # Python + skills combo must be caught BEFORE the generic skills rule
    (re.compile(r'\bpython\b.*\b(skill|skills|knowledge|experience|expertise)\b|\b(skill|skills|knowledge|experience|expertise)\b.*\bpython\b'), '3.1.Why Python.txt', 'python skills intent'),
    (re.compile(r'\b(skill|skills|strengths)\b(?!.*\bpython\b)'), '1.6.What are your skills.txt', 'skills intent'),
    (re.compile(r'\b(tech stack|technology stack|technologies|tools)\b'), '1.5.What technologies do you know.txt', 'tech intent'),
    (re.compile(r'^(what is |tell me about |explain |describe )?vconnect$'), '2.1.Tell me about VConnect.txt', 'vconnect intent'),
    (re.compile(r'^(what is |tell me about |explain |describe )?logisense$'), '2.4.Tell me about LogiSense.txt', 'logisense intent'),
    (re.compile(r'\b(do you know|know|use|learned|learning).*\bpython\b'), '3.1.Why Python.txt', 'python intent'),
]


def update_best(best: dict, score: float, entry: dict, matched: tuple[str, str]) -> None:
    if score > best['score']:
        best.update({
            'score': score,
            'file_name': entry['file_name'],
            'matched': [matched],
        })


def _idf_overlap(common: set[str], phrase_tokens: set[str], user_tokens: set[str], w_phrase: float = None, w_user: float = None) -> float:
    """
    Weighted overlap score using IDF.
    Instead of counting tokens equally, each token contributes its IDF weight.
    A match on 'vconnect' (idf~1.0) is worth far more than a match on 'project' (idf~0.1).

    Returns a value in [0, 1]: weighted_common / max(weighted_phrase, weighted_user).
    """
    if not common:
        return 0.0
    w_common   = sum(token_weight(t) for t in common)
    w_phrase   = w_phrase if w_phrase is not None else (sum(token_weight(t) for t in phrase_tokens) or 1e-9)
    w_user     = w_user if w_user is not None else (sum(token_weight(t) for t in user_tokens) or 1e-9)
    overlap    = w_common / w_phrase          # how much of the phrase is covered
    coverage   = w_common / w_user            # how much of the user query is covered
    return max(overlap, coverage)


def score_entry(entry: dict, user_norm: str, user_tokens: set[str]) -> dict:
    best = {'score': 0, 'file_name': entry['file_name'], 'matched': []}

    for original, phrase_norm, phrase_tokens in entry['_key_sentences']:
        w_phrase = entry['_key_sentence_weights'].get(phrase_norm)
        if phrase_norm == user_norm:
            update_best(best, 100, entry, ('sentence', original))
            continue

        # Containment: only fire when the user input has ≥2 meaningful tokens
        # so single-word or stop-word-only queries can't hit long key_sentences.
        # Also require the IDF-weighted overlap to be ≥ 0.3 to prevent generic
        # prefix phrases like "tell me about python" matching every
        # "tell me about X" sentence indiscriminately.
        if len(phrase_norm) >= 4 and len(user_tokens) >= 2 and (phrase_norm in user_norm or user_norm in phrase_norm):
            common_cont = phrase_tokens & user_tokens
            idf_cont = _idf_overlap(common_cont, phrase_tokens, user_tokens, w_phrase=w_phrase) if common_cont else 0.0
            if idf_cont >= 0.3:
                containment_score = 88 if len(phrase_tokens) > 1 else 74
                update_best(best, containment_score, entry, ('sentence', original))

        common = phrase_tokens & user_tokens
        if phrase_tokens and common:
            # Use IDF-weighted overlap instead of raw token count ratio.
            # This prevents generic tokens like 'project', 'data', 'logic'
            # from inflating scores for the wrong entry.
            idf_score = _idf_overlap(common, phrase_tokens, user_tokens, w_phrase=w_phrase)
            # Require a meaningful weighted signal (>= 0.35 weighted overlap)
            # OR at least 2 common tokens where IDF weight of each is decent
            high_value_common = [t for t in common if token_weight(t) >= 0.3]
            if idf_score >= 0.35 or (len(high_value_common) >= 2 and idf_score >= 0.20):
                token_score = 58 + (idf_score * 34)
                update_best(best, token_score, entry, ('sentence', original))

        # --- Semantic fuzzy fallback (replaces raw SequenceMatcher) ---
        # SequenceMatcher compares character sequences, not meaning, so
        # "tell me about your Python use" and "tell me about python" score
        # high while "do you code in Python?" scores low despite being the
        # right intent.  Instead we use a looser IDF token-overlap check:
        # if the user shares at least one high-value token with the phrase,
        # or ≥2 tokens with decent IDF weight, we score it proportionally.
        # Only run on meaningful (non-stop-word) user tokens so queries like
        # "can u help me" (all stop words) don't match via incidental overlap.
        meaningful_user_tokens = user_tokens - STOP_WORDS
        if phrase_tokens and meaningful_user_tokens:
            sem_common = phrase_tokens & meaningful_user_tokens
            if sem_common:
                sem_idf = _idf_overlap(sem_common, phrase_tokens, meaningful_user_tokens, w_phrase=w_phrase)
                high_val = [t for t in sem_common if token_weight(t) >= 0.4]
                # Accept: one very distinctive token, or two medium-weight tokens
                if high_val or (len(sem_common) >= 2 and sem_idf >= 0.15):
                    sem_score = 50 + (sem_idf * 30)
                    update_best(best, sem_score, entry, ('semantic fuzzy', original))

        # Typo guard: keep SequenceMatcher but only at very high similarity
        # (≥0.92) so it only fires for near-identical strings with small edits,
        # and score it below the IDF token path so it can never beat a good
        # semantic match.
        len_user = len(user_norm)
        len_phrase = len(phrase_norm)
        if len_user > 0 and len_phrase > 0 and (2.0 * min(len_user, len_phrase) / (len_user + len_phrase)) >= 0.92:
            ratio = SequenceMatcher(None, user_norm, phrase_norm).ratio()
            if ratio >= 0.92:
                update_best(best, 48 + (ratio * 20), entry, ('typo fuzzy', original))

    for original, keyword_norm, keyword_tokens in entry['_keywords']:
        w_kw = entry['_keyword_weights'].get(keyword_norm)
        if keyword_norm == user_norm:
            # Penalise exact keyword match proportionally to how common it is
            idf_penalty = int((1.0 - token_weight(keyword_norm)) * 20)
            update_best(best, 90 - idf_penalty, entry, ('keyword', original))
            continue

        if not keyword_tokens:
            continue

        # Skip single-token keywords that are near-ubiquitous in the dataset.
        # Threshold raised to 0.25 to prune more low-signal generic tokens.
        if len(keyword_tokens) == 1:
            tok = next(iter(keyword_tokens))
            if tok in WEAK_DUPLICATE_WORDS or token_weight(tok) < 0.25:
                continue

        common = keyword_tokens & user_tokens
        if not common:
            continue

        idf_score = _idf_overlap(common, keyword_tokens, user_tokens, w_phrase=w_kw)

        if keyword_tokens.issubset(user_tokens) and idf_score >= 0.25:
            kw_idf_penalty = int((1.0 - idf_score) * 20)
            update_best(best, 72 - kw_idf_penalty, entry, ('keyword', original))
        elif len(common) >= 2 and idf_score >= 0.20:
            kw_idf_penalty = int((1.0 - idf_score) * 20)
            update_best(best, 60 - kw_idf_penalty, entry, ('keyword', original))

    return best


def exact_sentence_match(user_norm: str) -> dict | None:
    for entry in DATASET:
        for original, phrase_norm, _ in entry['_key_sentences']:
            if phrase_norm == user_norm:
                return {
                    'file_name': entry['file_name'],
                    'score': 100,
                    'matched': [('sentence', original)],
                }
    return None


def score_input(user_input: str, context: dict | None = None) -> dict | None:
    # ------------------------------------------------------------------
    # Multilingual pre-pass — runs on the RAW input before normalize_text
    # strips non-ASCII. If the input is in Tamil, Hindi, or Arabic script,
    # translate it to a canonical English phrase and score that instead.
    # The lang_tag is stored in the result so the chat route can prepend
    # a brief acknowledgement in the user's language.
    # ------------------------------------------------------------------
    lang_tag: str | None = None
    ml_result = translate_multilingual(user_input)
    if ml_result:
        user_input, lang_tag = ml_result

    user_norm = normalize_text(user_input)
    if not user_norm:
        return None

    # 1. Exact sentence match (original input)
    exact = exact_sentence_match(user_norm)
    if exact:
        if lang_tag:
            exact['lang_tag'] = lang_tag
        return exact

    # 2. Intent rules (original input)
    for pattern, file_name, label in INTENT_RULES:
        if pattern.search(user_norm):
            return {
                'file_name': file_name,
                'score': 96,
                'matched': [('intent', label)],
            }

    # 3. Paraphrase normalisation — map semantically equivalent phrases to a
    #    canonical form so token overlap scoring works on meaning, not wording.
    canonical = apply_paraphrase_map(user_norm)
    if canonical != user_norm:
        exact2 = exact_sentence_match(normalize_text(canonical))
        if exact2:
            if lang_tag:
                exact2['lang_tag'] = lang_tag
            return exact2
        for pattern, file_name, label in INTENT_RULES:
            if pattern.search(normalize_text(canonical)):
                return {
                    'file_name': file_name,
                    'score': 94,
                    'matched': [('paraphrase+intent', label)],
                    'lang_tag': lang_tag,
                }
        user_tokens = set(tokenize(normalize_text(canonical)))
    else:
        user_tokens = set(tokenize(user_norm))

    # Detect follow-up intent early so the guard below can skip for context queries.
    _is_early_followup = bool(CONTEXT_FOLLOW_UP_PATTERNS.search(user_norm))
    _has_context       = bool((context or {}).get('last_category', ''))

    # Guard: single-token queries with low IDF return None immediately.
    # Bypass when a follow-up pattern is detected AND context exists —
    # the memory boost below will give it a valid score.
    if len(user_tokens) == 1:
        sole_token = next(iter(user_tokens))
        if (token_weight(sole_token) < 0.5 or sole_token in STOP_WORDS) \
                and not (_is_early_followup and _has_context):
            return None
    # Zero-token queries (all stop words) also need the same bypass
    if len(user_tokens) == 0 and not (_is_early_followup and _has_context):
        return None

    # -----------------------------------------------------------------------
    # 4. Conversation memory — context-aware scoring boost
    #
    # When the user's query is vague (pronouns, "tell me more", ≤1 own token)
    # AND we have a previous topic category, boost entries in that category.
    #
    # Boost tiers:
    #   +60  explicit follow-up phrase ("tell me more", "elaborate", etc.)
    #         Large enough to push a 0-scored entry past the 55 threshold.
    #   +20  query is mostly pronouns / vague (≤1 content token of its own)
    #   +8   weak follow-up — same category gets a soft nudge
    # -----------------------------------------------------------------------
    ctx_category = (context or {}).get('last_category', '')
    ctx_file     = (context or {}).get('last_file', '')

    own_tokens           = user_tokens - CONTEXT_PRONOUNS - STOP_WORDS
    is_explicit_followup = _is_early_followup
    is_vague_followup    = len(own_tokens) <= 1 and len(user_tokens) >= 1

    scored = [score_entry(entry, normalize_text(canonical), user_tokens) for entry in DATASET]

    if ctx_category and (is_explicit_followup or is_vague_followup):
        for item in scored:
            if _file_category(item['file_name']) == ctx_category:
                if is_explicit_followup:
                    boost = 60
                elif is_vague_followup:
                    boost = 20
                else:
                    boost = 8
                # Don't re-serve the exact same file — nudge within category
                if item['file_name'] == ctx_file and boost > 8:
                    boost = 5
                item['score'] = min(item['score'] + boost, 99)
                item['matched'].append(('context boost', f"cat:{ctx_category}"))
            elif is_explicit_followup:
                # For explicit follow-ups ("tell me more", "elaborate") the user
                # clearly wants to stay on topic. Hard-suppress off-category entries
                # so they can't beat the boosted in-category results.
                item['score'] = max(item['score'] - 40, 0)

    scored.sort(key=lambda item: item['score'], reverse=True)

    # Use a lower threshold when context boosting is active and the query is a
    # recognised follow-up — the boost already encodes confidence.
    min_score = 30 if (ctx_category and (is_explicit_followup or is_vague_followup)) else 55
    if not scored or scored[0]['score'] < min_score:
        return None

    scored[0]['score'] = int(round(min(scored[0]['score'], 100)))

    # Attach runner-up alternatives (score > 0, different file) for the hedge system.
    # Only entries that genuinely scored are included — not zero-score noise.
    alternatives = [
        item for item in scored[1:]
        if item['score'] > 0 and item['file_name'] != scored[0]['file_name']
    ][:3]
    scored[0]['alternatives'] = alternatives

    if lang_tag:
        scored[0]['lang_tag'] = lang_tag

    return scored[0]


def resolve_response_path(file_name: str) -> str | None:
    safe_name = os.path.basename(file_name or '')
    if not safe_name:
        return None

    candidates = [
        safe_name,
        safe_name if safe_name.endswith('.txt') else safe_name + '.txt',
        safe_name.replace('\u2019', '#U2019'),
        safe_name.replace("'", '#U2019'),
    ]

    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        path = os.path.join(RESPONSES_DIR, candidate)
        if os.path.exists(path):
            return path

    return None


def load_response(file_name: str) -> str | None:
    path = resolve_response_path(file_name)
    if path:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    return None


def get_fallback_response() -> str:
    fallback_path = os.path.join(RESPONSES_DIR, 'fallback.txt')
    if os.path.exists(fallback_path):
        with open(fallback_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    return (
        ">> QUERY UNRESOLVED. No matching data found in neural matrix.\n"
        ">> Try: skills, projects, VConnect, LogiSense, Python, resume, GitHub."
    )


def topic_label(file_name: str) -> str:
    """Convert a response filename into a short human-readable topic string.

    E.g. '2.1.Tell me about VConnect.txt' → 'Tell me about VConnect'
         '1.6.What are your skills.txt'   → 'What are your skills'
    """
    name = os.path.splitext(os.path.basename(file_name))[0]  # strip .txt
    # Strip leading numbering like '2.1.' or '13.4.'
    name = re.sub(r'^\d+\.\d+\.', '', name).strip()
    return name


def build_hedged_response(
    response_text: str,
    score: int,
    alternatives: list[dict],
    is_context_boosted: bool = False,
) -> str:
    """Wrap response_text with an appropriate confidence hedge.

    Score bands
    -----------
    75–100  High confidence — return as-is.
    55–74   Low confidence  — prefix soft hedge + list up to 2 alternatives.
    30–54   Context-boosted — lighter hedge (staying on topic).
    """
    if score >= 75:
        return response_text

    if is_context_boosted and score < 55:
        # Pure follow-up match — user wanted to stay on this topic
        prefix = ">> Continuing on the same topic:\n\n"
        return prefix + response_text

    # Low-confidence band (55–74): uncertain match
    prefix = (
        ">> I'm not 100% sure that's what you meant — here's my best match:\n\n"
    )
    suffix_parts = []

    # Only show alternatives that have a meaningful score gap from the winner
    # (within 30 points) so we don't suggest wildly irrelevant entries
    useful_alts = [
        a for a in alternatives
        if a['score'] > 0 and (score - a['score']) <= 30
    ][:2]

    if useful_alts:
        alt_lines = "\n".join(
            f"  • {topic_label(a['file_name'])}"
            for a in useful_alts
        )
        suffix_parts.append(
            f"\n\n>> Did you mean one of these instead?\n{alt_lines}"
        )

    suffix_parts.append(
        "\n>> Type your question differently for a better match."
    )

    return prefix + response_text + "".join(suffix_parts)


def actions_for(file_name: str) -> list[dict]:
    return ACTION_MAP.get(file_name, [])


# ---------------------------------------------------------------------------
# Response variation system
#
# Same question → same .txt file → same text every time feels robotic.
# Solution: per-session rotation tracking + inline opening-line variants +
# category-aware follow-up hooks appended to responses.
#
# Architecture:
#   RESPONSE_VARIANTS  — dict mapping file_name → list of alternate opening
#                        paragraphs that replace the first paragraph of the
#                        .txt response. Keeps the core content but changes
#                        the entry angle so repeat asks feel fresh.
#   FOLLOWUP_HOOKS     — dict mapping category → list of follow-up prompt
#                        lines. One is appended (rotated) to high-confidence
#                        responses so the conversation naturally continues.
#   Session tracking   — 'last_variant' and 'last_hook' stored in the session
#                        dict so consecutive asks of the same question cycle
#                        through variants rather than repeating.
# ---------------------------------------------------------------------------

RESPONSE_VARIANTS: dict[str, list[str]] = {
    '1.1.Who are you.txt': [
        "I'm Mohammed Razin H — Data Analyst and developer, Chennai-based.",
        "Name's Mohammed Razin H. I build data systems and I ship real products.",
        "Mohammed Razin H. MCA graduate, data analyst, builder — in that order.",
        "I'm Mohammed Razin H — a Data Analyst and developer based in Chennai.",
        "Name's Mohammed Razin H. I analyse data, build systems, and ship real products.",
        "Mohammed Razin H — MCA graduate, data analyst, developer. Chennai-based.",
    ],
    '1.3.Tell me about yourself.txt': [
        "Sure — I'll keep it honest and specific.",
        "Here's the honest version, without the buzzwords.",
        "Short version first, then I'll go deeper if you want.",
    ],
    '1.4.What do you do.txt': [
        "I turn raw, messy data into insights that actually drive decisions.",
        "I build data pipelines, dashboards, and systems that solve real problems.",
        "Depends on the day — data engineering, analytics, or building something new.",
    ],
    '2.1.Tell me about VConnect.txt': [
        "VConnect is my most technically substantial project — and the one that pushed me furthest as an engineer.",
        "If I had to pick one project that proves what I can do at scale, it's VConnect.",
        "VConnect started as a data challenge. It became a full data engineering system.",
    ],
    '3.1.Why Python.txt': [
        "Python chose me before I fully chose it — but once I went deep, the reasons became obvious.",
        "Honest answer: Python is just the right tool for the work I do.",
        "I've tried other languages. I keep coming back to Python — here's why.",
    ],
    '5.6.Why should we hire you.txt': [
        "Because I don't just know the tools — I've used them on real problems at real scale.",
        "Let me give you the evidence instead of the pitch.",
        "Most candidates can list tools. I can show you outcomes.",
    ],
    '7.6.Greetings User.txt': [
        ">> CYBERFORGE AI — ONLINE. Welcome, Operator. 👋",
        ">> Connection established. CYBERFORGE terminal active. 👋",
        ">> System ready. You've reached Mohammed Razin H's AI terminal. 👋",
    ],
    '1.6.What are your skills.txt': [
        "Skills are only meaningful with context, so here's mine:",
        "Here's what I actually know — and where I've applied it:",
        "Let me give you the real list, not the keyword-stuffed version:",
    ],
    '5.1.Are you looking for a job.txt': [
        "Yes — actively looking, specifically for roles where data actually matters.",
        "Yes. Open to the right opportunity — let me tell you what that looks like.",
        "Actively, yes. Here's what I'm looking for and what I bring:",
    ],
}

# Follow-up hooks by category.
# Each entry is a list of prompts — the system rotates through them per session.
FOLLOWUP_HOOKS: dict[str, list[str]] = {
    '1': [   # Identity / background
        "\n\n>> Want to know what I've actually built? Ask me about VConnect or LogiSense.",
        "\n\n>> Curious about my tech stack? Ask 'what technologies do you know'.",
        "\n\n>> Want the full picture? Ask me to walk you through my best project.",
    ],
    '2': [   # Projects
        "\n\n>> Want to go deeper? Ask me about the tech stack, the challenges, or the results.",
        "\n\n>> I have more projects — ask about LogiSense, my AI work, or what I'm building next.",
        "\n\n>> Ask me what problem this project actually solved — that's the interesting part.",
    ],
    '3': [   # Python / data science
        "\n\n>> Want to see Python in action? Ask about VConnect — 650K records, real pipelines.",
        "\n\n>> Ask me about a specific library — Pandas, NumPy, Scikit-learn, or what I'm learning now.",
        "\n\n>> Curious about my ML work? Ask about my AI projects or feature selection approach.",
    ],
    '4': [   # CyberForge / AI / cybersecurity
        "\n\n>> Want to know how I built this chatbot? Ask 'what is CyberForge AI'.",
        "\n\n>> Curious about the tech behind this terminal? I'm happy to walk you through it.",
        "\n\n>> Ask me about prompt engineering or what AI tools I actually use day-to-day.",
    ],
    '5': [   # Hiring / job / career
        "\n\n>> Want the evidence behind that? Ask me about VConnect or my hackathon wins.",
        "\n\n>> Curious about my availability or role preferences? Just ask.",
        "\n\n>> Ask me what makes me different from other candidates — I have a specific answer.",
    ],
    '7': [   # Personal / motivations
        "\n\n>> Want to talk about my work instead? Ask about my projects or skills.",
        "\n\n>> Curious what keeps me building? Ask what I'm currently learning.",
        "\n\n>> Ask me about the project I'm most proud of — the answer might surprise you.",
    ],
    '9': [   # System design / architecture
        "\n\n>> Want a concrete example? Ask how I applied this in VConnect or LogiSense.",
        "\n\n>> Ask me about another system design challenge — I have specific answers.",
        "\n\n>> Curious how I handle scale? Ask about the VConnect data pipeline.",
    ],
    '10': [  # ML / AI deep-dives
        "\n\n>> Want to go further? Ask about bias-variance trade-off, feature selection, or model validation.",
        "\n\n>> Ask me how I applied this on a real dataset — not a toy example.",
        "\n\n>> Curious about my full ML workflow? Ask about my latest AI project.",
    ],
    '13': [  # Behavioural / soft skills
        "\n\n>> Want a concrete example to back that up? Just ask.",
        "\n\n>> Ask me about another challenge I've faced — I have specific stories.",
        "\n\n>> Curious how I handle pressure or disagreement? Ask and I'll give you a real answer.",
    ],
}


def pick_variant(file_name: str, base_text: str, session: dict) -> str:
    """Replace the opening line/paragraph of base_text with a variant.

    Cycles through RESPONSE_VARIANTS for this file, skipping the one used
    last time (stored in session['variants'][file_name]).
    Falls back to base_text unchanged if no variants defined.
    """
    variants = RESPONSE_VARIANTS.get(file_name)
    if not variants:
        return base_text

    used = session.get('variants', {}).get(file_name)
    # Pick any variant that wasn't used last time
    choices = [v for v in variants if v != used] or variants
    chosen = random.choice(choices)

    # Store the choice in session (mutate in-place — caller holds reference)
    session.setdefault('variants', {})[file_name] = chosen

    # Replace only the first paragraph (up to first double newline)
    # so the core content stays intact.
    # If there's no double newline, try a single newline before falling back.
    # Never discard the response body — always append remaining content.
    parts = base_text.split('\n\n', 1)
    if len(parts) == 2:
        return chosen + '\n\n' + parts[1]
    # Try single-newline split as fallback
    parts = base_text.split('\n', 1)
    if len(parts) == 2:
        return chosen + '\n' + parts[1]
    # Truly single-line response — replace it entirely with the variant
    return chosen


def pick_hook(file_name: str, session: dict) -> str:
    """Return a follow-up hook line for this file's category, rotating per session."""
    category = _file_category(file_name)
    hooks = FOLLOWUP_HOOKS.get(category, [])
    if not hooks:
        return ''

    used = session.get('hooks', {}).get(file_name)
    choices = [h for h in hooks if h != used] or hooks
    chosen = random.choice(choices)

    session.setdefault('hooks', {})[file_name] = chosen
    return chosen


def apply_variation(file_name: str, base_text: str, score: int,
                    is_hedged: bool, session: dict) -> str:
    """Apply opening variant + follow-up hook to a response.

    Rules:
    - Variants always applied (freshness on repeat asks).
    - Hooks only on high-confidence (score >= 75) non-hedged responses —
      adding a "want to know more?" after an uncertain answer looks odd.
    - Action-only entries (cat 6 / 8) skip hooks — they already have buttons.
    """
    text = pick_variant(file_name, base_text, session)

    category = _file_category(file_name)
    skip_hook_cats = {'6', '8'}  # action / easter-egg categories
    if score >= 75 and not is_hedged and category not in skip_hook_cats:
        hook = pick_hook(file_name, session)
        if hook:
            text = text.rstrip() + hook

    return text


@app.errorhandler(429)
def rate_limit_handler(e):
    """Return a chatbot-style JSON response on rate limit breach."""
    return jsonify({
        'response': (
            ">> RATE LIMIT EXCEEDED. Slow down, operator.\n"
            ">> Maximum 30 messages per minute. Try again shortly."
        ),
        'score': 0,
        'confidence': 0,
        'is_hedged': False,
        'file': None,
        'matched': [],
        'actions': [],
    }), 429


@app.route('/chatbot')
def chatbot():
    return send_from_directory(FRONTEND_DIR, 'chatbot.html')


@app.route('/voice')
def voice():
    return send_from_directory(FRONTEND_DIR, 'voice.html')


@app.route('/')
def index():
    return send_from_directory(FRONTEND_DIR, 'index.html')


@app.route('/<path:filename>')
def portfolio_assets(filename):
    if filename in ('chat', 'status', 'suggestions'):
        return jsonify({'error': 'not found'}), 404

    return send_from_directory(FRONTEND_DIR, filename)


@app.route('/chat', methods=['POST'])
@limiter.limit('30 per minute')
def chat():
    data = request.get_json(silent=True) or {}
    user_input  = data.get('message', '').strip()
    session_id  = data.get('session_id', '').strip()   # frontend sends a UUID per page load

    # ------------------------------------------------------------------
    # Input length guard — truncate at 350 chars before any processing.
    # A real question never needs more than ~300 chars. Longer inputs are
    # either accidental pastes or deliberate stress / injection attempts.
    # We truncate (not reject) so partial-valid inputs still get a response.
    # ------------------------------------------------------------------
    MAX_INPUT = 350
    if len(user_input) > MAX_INPUT:
        user_input = user_input[:MAX_INPUT].strip()

    if not user_input:
        return jsonify({
            'response': '>> NULL INPUT DETECTED. Enter your query, human.',
            'score': 0,
            'confidence': 0,
            'file': None,
            'matched': [],
            'actions': [],
        })

    # Load conversation context for this session (empty dict if new/expired)
    context = get_session(session_id)

    result = score_input(user_input, context)

    if not result or result['score'] == 0:
        return jsonify({
            'response': get_fallback_response(),
            'score': 0,
            'confidence': 0,
            'file': None,
            'matched': [],
            'actions': [],
        })

    # Persist this turn into the session BEFORE returning
    user_tokens = set(tokenize(normalize_text(user_input)))
    update_session(session_id, result['file_name'], user_tokens)

    # Re-fetch the session after update so we can read/write variant state
    session = get_session(session_id)

    response_text = load_response(result['file_name'])
    if not response_text:
        response_text = (
            f">> RESPONSE FILE NOT FOUND: {result['file_name']}\n"
            ">> Place the matching response .txt file inside the responses folder."
        )

    score      = result['score']
    alts       = result.get('alternatives', [])
    matched    = result.get('matched', [])
    is_ctx     = any(m[0] == 'context boost' for m in matched)

    # Apply confidence hedging for low-certainty matches
    hedged_text = build_hedged_response(response_text, score, alts, is_context_boosted=is_ctx)
    is_hedged   = hedged_text != response_text

    # Apply opening variant + follow-up hook for response freshness
    final_text = apply_variation(result['file_name'], hedged_text, score, is_hedged, session)

    # Persist the variant/hook rotation choices back into the live session
    update_session_variation(session_id, session.get('variants', {}), session.get('hooks', {}))

    # If the user wrote in Tamil, Hindi, or Arabic, prepend a brief
    # acknowledgement in their language before the English response.
    lang_tag = result.get('lang_tag')
    if lang_tag and lang_tag in LANG_ACK:
        final_text = LANG_ACK[lang_tag] + final_text

    return jsonify({
        'response':   final_text,
        'score':      score,
        'confidence': score,
        'is_hedged':  is_hedged,
        'lang_tag':   lang_tag,
        'file':       result['file_name'],
        'matched':    matched[:10],
        'actions':    actions_for(result['file_name']),
    })


@app.route('/suggestions', methods=['GET'])
@limiter.limit('60 per minute')
def suggestions():
    query = normalize_text(request.args.get('q', ''))
    if not query:
        return jsonify({'suggestions': SUGGESTIONS})

    # Use the same canonical tokenization as the main scorer so synonyms
    # ("ml" → "machine learning", "ai" → "artificial intelligence") match.
    user_tokens = set(tokenize(query))
    meaningful  = user_tokens - STOP_WORDS

    scored = []
    for item in SUGGESTIONS:
        item_norm   = normalize_text(item)
        item_tokens = set(tokenize(item_norm))

        # Tier 1 — prefix match: user input is a leading substring of suggestion
        if item_norm.startswith(query):
            score = 4.0 + _idf_overlap(item_tokens & user_tokens, item_tokens, user_tokens)
        # Tier 2 — substring containment
        elif query in item_norm:
            score = 3.0 + _idf_overlap(item_tokens & user_tokens, item_tokens, user_tokens)
        # Tier 3 — IDF token overlap on meaningful (non-stop) tokens
        elif meaningful:
            common = item_tokens & meaningful
            if not common:
                continue
            score = _idf_overlap(common, item_tokens, meaningful)
            if score < 0.15:
                continue
        else:
            continue

        scored.append((score, item))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return jsonify({'suggestions': [item for _, item in scored[:12]]})


@app.route('/status', methods=['GET'])
def status():
    total = len(DATASET)
    loaded = sum(1 for entry in DATASET if resolve_response_path(entry['file_name']))
    return jsonify({'total_entries': total, 'responses_loaded': loaded, 'status': 'ONLINE'})


if __name__ == '__main__':
    if not os.path.exists(FRONTEND_DIR):
        print(f"\n>> Warning: Frontend folder not found at: {FRONTEND_DIR}")
        print(">> Local single-port serving will be unavailable unless frontend files are placed there.")

    port = int(os.environ.get('PORT', 6161))
    print("\n>> MEGATRON OS - CYBERFORGE AI SYSTEM INITIALIZING...")
    print(f">> Portfolio:  http://localhost:{port}/")
    print(f">> Chatbot UI: http://localhost:{port}/chatbot")
    print(f">> Voice Mode: http://localhost:{port}/voice")
    print(f">> Chat API:   http://localhost:{port}/chat")
    print(f">> Status API: http://localhost:{port}/status")
    print(f">> Suggest:    http://localhost:{port}/suggestions\n")
    app.run(debug=False, host='0.0.0.0', port=port)
