import nltk
import json
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import warnings
import textwrap
from sklearn.model_selection import train_test_split
import os

warnings.filterwarnings('ignore')

# --- 1. Intents Data ---
intents = {
   "intents": [
  {
    "tag": "bennett_admissions_process",
    "patterns": [
      "What is the admission process for Bennett University?",
      "How can I apply to Bennett?",
      "Tell me the steps for admission.",
      "What documents are needed for application?",
      "Is there an entrance exam?",
      "How does the selection process work?",
      "What is the application procedure?"
    ],
    "responses": [
      "The admission process is merit-based. The main steps are: 1. Fill the application form on the website (applications.bennett.edu.in/bu). 2. Submit/Update your scores (like Class XII, JEE, CUET, SAT) within 7 days of their declaration. 3. Candidates are shortlisted based on their scores. 4. Selected candidates receive a provisional admission offer letter via email. 5. Pay the required fee by the specified date to confirm your seat."
    ]
  },
  {
    "tag": "bennett_admission_deadline",
    "patterns": [
      "When is the last date to apply?",
      "What is the application deadline for 2025?",
      "Are admissions for 2025 open?",
      "Can I still apply?",
      "When do admissions close?"
    ],
    "responses": [
      "The application deadline for the 2025 academic year is September 30, 2025. It is recommended to apply early since seats may be allocated on a rolling / first-come basis; always confirm current deadlines on the official admissions page."
    ]
  },
  {
    "tag": "bennett_schools",
    "patterns": [
      "What are the different schools at Bennett University?",
      "Which departments do you have?",
      "Tell me about the academic schools.",
      "What fields of study are offered?"
    ],
    "responses": [
      "Bennett University comprises multiple schools, including the School of Engineering & Applied Sciences (and related engineering disciplines), School of Management, Times School of Media, School of Law, School of Design, School of Liberal Arts, and specialized schools for AI and other emerging areas. Each school hosts undergraduate, postgraduate, and research programs — see the Academics/Schools section on the site for program-wise details."
    ]
  },
  {
    "tag": "bennett_btech_courses",
    "patterns": [
      "What B.Tech specializations are available?",
      "Which engineering courses do you offer?",
      "Can I study AI at Bennett?",
      "Tell me about the B.Tech in Computer Science.",
      "What are the options in the School of Engineering?"
    ],
    "responses": [
      "The School of Engineering offers B.Tech degrees and related undergraduate programs such as Computer Science (CSE), Artificial Intelligence (AI), Electronics & Communication (ECE), Mechanical Engineering, Biotechnology, and Engineering Physics. Additional specializations and elective tracks include Data Science, Cyber Security, Robotics, Gaming, and Drones among others. Check the specific program pages for seat intake and curriculum."
    ]
  },
  {
    "tag": "bennett_management_courses",
    "patterns": [
      "Do you have an MBA program?",
      "What are the management courses?",
      "Tell me about the BBA program.",
      "What specializations are offered in the MBA?"
    ],
    "responses": [
      "Yes — Bennett’s School of Management offers undergraduate (BBA) and postgraduate (MBA) programs. MBA specializations and elective areas include Marketing, Finance, Human Resources, Business Analytics, Media Management, and Logistics & Supply Chain Management. Program details and admission requirements areavailable on the Management school pages."
    ]
  },
  {
    "tag": "bennett_law_courses",
    "patterns": [
      "What law programs are available?",
      "Do you offer an integrated law degree?",
      "Tell me about the School of Law.",
      "Is Bennett a good university for law?"
    ],
    "responses": [
      "The School of Law offers integrated undergraduate law programs such as B.A. LL.B. (Hons.) and BBA LL.B. (Hons.), as well as postgraduate (LL.M.) and Ph.D. research programs. The School is designed to meet Bar Council of India norms; check curriculum, faculty, and placement specifics on the School of Law page to see fit for your goals."
    ]
  },
  {
    "tag": "bennett_placements",
    "patterns": [
      "How are the placements at Bennett University?",
      "What is the average package?",
      "Which companies recruit from Bennett?",
      "Tell me about the placement record for B.Tech.",
      "What was the highest package offered?",
      "What is the placement for MBA?",
      "How are the placements for Law?",
      "What is the average package for BBA or MCA?"
    ],
    "responses": [
      "Bennett has a strong placement record managed by its Career Services Team. Key stats include: B.Tech (Highest: ₹1.37 Cr PA, Avg: ₹11.10 LPA), MBA (Highest: ₹33 LPA, Avg: ₹7.41 LPA), Law (Highest: ₹28.17 LPA, Avg: ₹6.7 LPA), BBA (Highest: ₹23.5 LPA, Avg: ₹5.16 LPA), MCA (Highest: ₹18 LPA, Avg: ₹5.7 LPA), and BA Mass Comm (Highest: ₹21 LPA, Avg: ₹3.91 LPA). Over 750 companies participate in placements."
    ]
  },
  {
    "tag": "bennett_recruiters",
    "patterns": [
      "Who are the top recruiters?",
      "Which major companies visit the campus for placements?",
      "Can you name some of the companies that hire from Bennett?"
    ],
    "responses": [
      "A large and diverse set of national and multinational companies recruit from Bennett across sectors (IT, product, consulting, finance, media, manufacturing, etc.). The placement/placement-partners page on the official site publishes an up-to-date list of visiting recruiters and role categories — please consult that page for the detailed and current list."
    ]
  },
  {
    "tag": "bennett_hostel_facilities",
    "patterns": [
      "Is there a hostel on campus?",
      "What are the hostel facilities like?",
      "Tell me about the accommodation.",
      "Are the hostels air-conditioned?",
      "What is the hostel fee?"
    ],
    "responses": [
      "Yes — Bennett provides modern on-campus hostels and collaborates with off-campus partners (e.g., Nalanda Living). Rooms are generally furnished and air-conditioned; standard facilities include mess/cafeteria, common areas, security, and utilities. Hostel fees vary by occupancy: on-campus triple sharing is approximately ₹1,70,000 per annum, while Nalanda (off-campus) triple is shown around ₹1,50,000 per annum. Single and twin options are higher (single reported around ₹3,24,000 PA and twin around ₹2,10,000 PA in fee tables). Always confirm the exact room type and current fees from the Fees/Hostel page."
    ]
  },
  {
    "tag": "bennett_student_clubs",
    "patterns": [
      "What student clubs are there?",
      "How is the campus life?",
      "Are there extracurricular activities?",
      "What is there to do outside of academics?",
      "Tell me about the student societies",
      "What fests or cultural events happen?",
      "What is campus life like?"
    ],
    "responses": [
      "Campus life at Bennett is very active. There are over 40 student-led clubs and societies covering interests like music, drama, dance, technology, entrepreneurship, and social causes. The university also hosts a calendar full of dynamic events, including cultural festivals, college fests, and competitions."
    ]
  },
  {
    "tag": "bennett_sports_facilities",
    "patterns": [
      "What sports facilities are there?",
      "Is there a gym or sports complex?",
      "Can I play cricket or football on campus?",
      "Tell me about the sports arenas",
      "Are there basketball courts?",
      "What sports can I play?"
    ],
    "responses": [
      "Yes, Bennett has extensive sports facilities on its 68-acre campus. This includes a large sports complex with 16+ state-of-the-art arenas, featuring sprawling cricket and football grounds, and dedicated courts for badminton and basketball, among other sports."
    ]
  },
  {
    
    "tag": "bennett_atm",
    "patterns": [
      "Is there an ATM on campus?",
      "Where can I find an ATM?",
      "ATM location"
    ],
    "responses": [
      "Yes, there is a 24x7 ATM on campus for your convenience."
    ]
  },
  {
    "tag": "bennett_food_options",
    "patterns": [
      "What food options are on campus?",
      "Are there hangout spots or eating joints?",
      "Can I get food like Domino's?",
      "What about dining?",
      "Where can I eat?",
      "Tell me about the cafeteria or mess"
    ],
    "responses": [
      "Yes, the campus has various hangout spots, eating joints, and dining options, including popular food chains like Domino's. This is in addition to the mess/cafeteria facilities for hostel residents."
    ]
  },
  {
    "tag": "bennett_wellness_center",
    "patterns": [
      "Is there a wellness center or medical facility?",
      "What is the wellness centre?",
      "Is there a doctor on campus?",
      "What if I get sick?"
    ],
    "responses": [
      "Yes, there is an on-campus Wellness Centre. It's a 20-bed integrated medical facility to handle student health needs."
    ]
  },
  {
    "tag": "bennett_convenience_store",
    "patterns": [
      "Is there a convenience store?",
      "Where can I buy daily need items?",
      "Is there a shop on campus?",
      "What about shopping?"
    ],
    "responses": [
      "Yes, there is a convenience store on campus to meet daily needs."
    ]
  },
  {
    "tag": "bennett_fee_structure",
    "patterns": [
      "How much is the fee for B.Tech?",
      "What is the fee structure for the MBA program?",
      "Can you give me the details of the tuition fees?",
      "Is there a registration fee?"
    ],
    "responses": [
      "Fees vary by program and year. For example, the first-year fee for B.Tech (CSE) is reported as ₹4,25,000, which includes a one-time registration charge of ₹45,000. The fee pages list program-wise tuition, registration, refundable deposits, and other charges (hostel, mess, security). For exact, program-specific breakdowns and the latest updates, consult the official Fee Structure page."
    ]
  },
  {
    "tag": "bennett_scholarships",
    "patterns": [
      "Does Bennett University offer scholarships?",
      "How can I get a scholarship?",
      "What are the eligibility criteria for scholarships?",
      "What is the merit-based scholarship?",
      "Is there a scholarship based on JEE or CUET score?",
      "How much scholarship can I get for B.Tech?"
    ],
    "responses": [
      "Yes, Bennett offers merit-based scholarships (up to 50%) for the first year, awarded on a first-come, first-served basis. Eligibility is based on 2025 competitive exam scores (like JEE Mains, CUET) or Class XII marks from 2024/2025. For example, for B.Tech, a 50% scholarship may be offered for a Class XII (Best of 3) score >= 96% or a JEE score >= 90 percentile. Scholarship seats are limited, and meeting the criteria doesn't guarantee an award. To continue the scholarship, a student must maintain a CGPA of 8 with no backlogs."
    ]
  },
  {
    "tag": "bennett_special_scholarships",
    "patterns": [
      "Is there a scholarship for a single girl child?",
      "What about a scholarship for siblings?",
      "Do alumni get any scholarship benefits?",
      "Is there a discount for defence personnel?"
    ],
    "responses": [
      "Yes, in addition to the main merit-based scholarships, Bennett University also offers 'Other (Special) Scholarships'. These include categories for Sibling, Single Girl Child, Alumni, and Defence Personnel. You should contact the admissions office for specific details and eligibility for these."
    ]
  },
  {
    "tag": "bennett_about",
    "patterns": [
      "Tell me about Bennett University",
      "What is Bennett University?",
      "Who founded Bennett University?",
      "What is the university's vision?",
      "How big is the campus?",
      "How many students are there?"
    ],
    "responses": [
      "Bennett University, established by The Times Group, is a private, research-driven university in Greater Noida. It's a 68-acre campus with over 11,500 students. It focuses on industry-aligned education across Engineering, AI, Management, Media, Law, Liberal Arts, and Design, with a faculty that is over 90% Ph.D. holders."
    ]
  },
  {
    "tag": "bennett_times_group_legacy",
    "patterns": [
      "What is the Times Group connection?",
      "Tell me about the Times Group legacy",
      "Is this university related to Times of India?"
    ],
    "responses": [
      "Yes, Bennett University was established by The Times Group, India's oldest and largest media house, which began in 1838. This connection provides the university with strong industry engagement and a focus on 'New Age' skills."
    ]
  },
  {
    "tag": "bennett_ai_courses",
    "patterns": [
      "What courses are in the School of Artificial Intelligence?",
      "Can I study AI at Bennett?",
      "Tell me about AI programs",
      "Do you have B.Tech in AI?",
      "Do you offer BCA in Artificial Intelligence?"
    ],
    "responses": [
      "Yes, Bennett has a dedicated School of Artificial Intelligence. It offers undergraduate programs like B.Tech (Artificial Intelligence), BCA (Artificial Intelligence), and B.Sc. (Artificial Intelligence), as well as M.Tech, M.Sc., and MCA programs focused on AI."
    ]
  },
  {
    "tag": "bennett_liberal_arts_courses",
    "patterns": [
      "What are the liberal arts programs?",
      "Tell me about the School of Liberal Arts",
      "Can I get a B.A. in Psychology?",
      "What B.A. Hons. degrees are offered?",
      "Do you have a degree in Economics or Sociology?"
    ],
    "responses": [
      "The School of Liberal Arts offers several B.A. (Hons.) degrees, including specializations in Psychology, Economics, English Literature, Sociology, Philosophy, Business Studies, and Political Science & International Relations."
    ]
  },
  {
    "tag": "bennett_media_courses",
    "patterns": [
      "What courses does the Times School of Media offer?",
      "Tell me about the media programs",
      "Can I study journalism?",
      "Do you have a B.A. in Mass Communication?",
      "What about film or TV programs?"
    ],
    "responses": [
      "The Times School of Media offers undergraduate programs like B.A. (Mass Communication) and B.A. (Film, TV & Web Series), as well as postgraduate and doctoral programs in media."
    ]
  },
  {
    "tag": "bennett_design_courses",
    "patterns": [
      "What design courses are available?",
      "Tell me about the School of Design",
      "Do you offer Fashion Design?",
      "What is B. Des?",
      "Can I study Game Design or VFX?"
    ],
    "responses": [
      "The School of Design offers B.Des (Hons.) programs in Fashion Design, Intelligent Textile Design, Product Design with AI, Game Design, Advanced Animation & VFX, and Communication Design."
    ]
  },
  {
    "tag": "bennett_research",
    "patterns": [
      "How is the research at Bennett?",
      "Tell me about research excellence",
      "How many labs or patents does Bennett have?",
      "What research grants has the university received?"
    ],
    "responses": [
      "Bennett has a strong focus on research, with 121+ high-end labs, over 6000 publications, and 188 published patents. The university has received over Rs 30 Cr. in research and consultancy grants."
    ]
  },
  {
    "tag": "bennett_innovation_startups",
    "patterns": [
      "Is there an innovation center?",
      "Does Bennett support startups?",
      "What is the Spark Cell?",
      "Tell me about entrepreneurship on campus"
    ],
    "responses": [
      "Yes, Bennett has an Innovation Centre that supports student startups through mentors, a Startup Portfolio, and the 'Spark Cell'. Over 60 startups have been launched, raising over ₹120 Cr+ in funds."
    ]
  },
  {
    "tag": "bennett_global_partnerships",
    "patterns": [
      "Does Bennett have international collaborations?",
      "What are the global programs?",
      "Can I study abroad?",
      "Tell me about the global pathway program",
      "Who are the academic partners?"
    ],
    "responses": [
      "Yes, Bennett has over 90 international collaborations with universities like Kent State University (USA), The University of BColumbia (Canada), and Florida International University (USA). They offer Global Pathway programs (like 2+2 or 3+1) allowing students to study partially at a partner university abroad."
    ]
  },
  {
    "tag": "bennett_contact",
    "patterns": [
      "How can I contact Bennett University?",
      "What is the university's address?",
      "Is there a contact number for admissions?",
      "Where is Bennett University located?"
    ],
    "responses": [
      "Bennett University is located at Plot Nos. 8–11, TechZone II, Greater Noida, Uttar Pradesh (Pin code 201310). For admissions assistance you can call the toll-free number 1800-103-8484; other contact details (departmental phone numbers and email IDs) are available on the Contact Us page of the official website."
    ]
  }
]
}
# --- 2. NLTK Downloads and Preprocessing ---
# Ensure NLTK data is available and downloaded into a project-local folder.
nltk_data_dir = os.path.join(os.path.dirname(__file__), 'nltk_data')
if nltk_data_dir not in nltk.data.path:
  nltk.data.path.insert(0, nltk_data_dir)

def _ensure_nltk_resource(resource_path, download_name=None):
  try:
    nltk.data.find(resource_path)
  except LookupError:
    name = download_name if download_name else resource_path.split('/')[-1]
    nltk.download(name, download_dir=nltk_data_dir, quiet=True)

_ensure_nltk_resource('tokenizers/punkt', 'punkt')
# Some environments reference a 'punkt_tab' tokenizer directory; ensure it's available too.
try:
  _ensure_nltk_resource('tokenizers/punkt_tab', 'punkt_tab')
except Exception:
  # If 'punkt_tab' isn't available via downloader in this environment, fall back to punkt
  pass
_ensure_nltk_resource('corpora/wordnet', 'wordnet')
_ensure_nltk_resource('corpora/omw-1.4', 'omw-1.4')

stemmer = WordNetLemmatizer()

words = []
classes = []
documents = []
ignore_words = ['?']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [stemmer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unique lemmatized words", words)

training = []
output_empty = [0] * len(classes)

# Training set, bag of words for each sentence
for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [stemmer.lemmatize(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

random.shuffle(training)


# --- 3. Data Augmentation ---

limit_per_tag = 2

def synonym_replacement(tokens, limit):
    augmented_sentences = []
    for i in range(len(tokens)):
        synonyms = []
        for syn in wordnet.synsets(tokens[i]):
            for lemma in syn.lemmas():
                synonyms.append(lemma.name().replace('_', ' '))

        synonyms = list(set(synonyms))

        if len(synonyms) > 0:
            num_augmentations = min(limit, len(synonyms))
            sampled_synonyms = random.sample(synonyms, num_augmentations)
            for synonym in sampled_synonyms:
                augmented_tokens = tokens[:i] + [synonym] + tokens[i + 1:]
                augmented_sentences.append(' '.join(augmented_tokens))
    return augmented_sentences


augmented_data = []

for i, doc in enumerate(training):
    bag, output_row = doc
    original_tokens = [words[j] for j in range(len(words)) if bag[j] == 1]

    augmented_sentences = synonym_replacement(original_tokens, limit_per_tag)

    for augmented_sentence in augmented_sentences:
        augmented_bag = [0] * len(words)

        augmented_pattern_words = nltk.word_tokenize(augmented_sentence)
        augmented_pattern_words = [stemmer.lemmatize(word.lower()) for word in augmented_pattern_words]

        for w_index, w in enumerate(words):
             if w in augmented_pattern_words:
                 augmented_bag[w_index] = 1

        augmented_data.append([augmented_bag, output_row])

combined_data = training + augmented_data
random.shuffle(combined_data)


def separate_data_by_tags(data):
    data_by_tags = {}
    for d in data:
        tag = tuple(d[1])
        if tag not in data_by_tags:
            data_by_tags[tag] = []
        data_by_tags[tag].append(d)
    return data_by_tags.values()


separated_data = separate_data_by_tags(combined_data)

training_data = []
testing_data = []

for tag_data in separated_data:
    if len(tag_data) >= 2:
        train_data, test_data = train_test_split(tag_data, test_size=0.2, random_state=42)
        training_data.extend(train_data)
        testing_data.extend(test_data)
    else:
        training_data.extend(tag_data)


random.shuffle(training_data)
random.shuffle(testing_data)

train_x = np.array([d[0] for d in training_data])
train_y = np.array([d[1] for d in training_data])
test_x = np.array([d[0] for d in testing_data])
test_y = np.array([d[1] for d in testing_data])


# --- 4. PyTorch Model Definition ---

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)


class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def accuracy(predictions, targets):
    predicted_labels = torch.argmax(predictions, dim=1)
    true_labels = torch.argmax(targets, dim=1)
    correct = (predicted_labels == true_labels).sum().item()
    total = targets.size(0)
    return correct / total


def test_model(model, test_loader, criterion):
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0

    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            total_accuracy += accuracy(outputs, targets) * inputs.size(0)

    average_loss = total_loss / len(test_loader.dataset) if len(test_loader.dataset) > 0 else 0.0
    average_accuracy = total_accuracy / len(test_loader.dataset) if len(test_loader.dataset) > 0 else 0.0
    return average_loss, average_accuracy


train_x = torch.tensor(train_x).float()
train_y = torch.tensor(train_y).float()
test_x = torch.tensor(test_x).float()
test_y = torch.tensor(test_y).float()

batch_size = 64
train_dataset = CustomDataset(train_x, train_y)
test_dataset = CustomDataset(test_x, test_y)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

input_size = len(train_x[0])
hidden_size = 8
output_size = len(train_y[0])


def train_model(num_epochs=50, model_path='model.pth', print_progress=True):
    """Train the model and save to model_path."""
    model = NeuralNetwork(input_size, hidden_size, output_size)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())

    if print_progress:
        print("\nStarting Training...\n")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        for inputs, targets in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_acc += accuracy(outputs, targets) * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_acc / len(train_loader.dataset)

        if print_progress:
            print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_acc:.4f}")

            test_loss, test_accuracy = test_model(model, test_loader, criterion)
            print(f"Epoch [{epoch+1}/{num_epochs}], Testing Loss: {test_loss:.4f} (Acc Focus), Testing Accuracy: {test_accuracy:.4f}\n")

    torch.save(model.state_dict(), model_path)
    if print_progress:
        print(f"Model saved to {model_path}")
    return model


def load_model(model_path, input_size, hidden_size, output_size):
    model = NeuralNetwork(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def preprocess_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def sentence_to_features(sentence_words, words):
    features = [1 if w in sentence_words else 0 for w in words]
    return torch.tensor(features).float().unsqueeze(0)


def _best_intent_by_overlap(sentence_words, intents_obj):
  """Return the intent tag with the highest token-overlap score (and the score).

  This is a simple heuristic used as a conversational fallback when the model
  is unsure or the user asks a short follow-up. It counts lemmatized token
  overlap between the user's sentence and each intent's patterns.
  """
  best_tag = None
  best_score = 0
  sent_set = set(sentence_words)
  for intent in intents_obj['intents']:
    # combine all pattern tokens for this intent
    intent_tokens = []
    for p in intent.get('patterns', []):
      intent_tokens.extend([stemmer.lemmatize(w.lower()) for w in nltk.word_tokenize(p)])
    score = len(sent_set.intersection(set(intent_tokens)))
    if score > best_score:
      best_score = score
      best_tag = intent['tag']
  return best_tag, best_score


def generate_response(sentence, model, words, classes, confidence_threshold=0.15, prev_tag=None):
  sentence_words = preprocess_sentence(sentence)
  sentence_words_in_vocab = [w for w in sentence_words if w in words]

  # If the sentence contains no known tokens, attempt to answer using
  # previous context (prev_tag) or return a polite fallback.
  if len(sentence_words_in_vocab) == 0:
    if prev_tag:
      # return another response from the previous tag to continue the thread
      for intent in intents['intents']:
        if intent['tag'] == prev_tag:
          return random.choice(intent['responses'])
    return "I'm sorry, but I don't understand. Can you please rephrase or provide more information?"

  features = sentence_to_features(sentence_words_in_vocab, words)

  with torch.no_grad():
    outputs = model(features)

  probabilities, predicted_class_index = torch.max(outputs, dim=1)
  confidence = probabilities.item()
  predicted_tag = classes[predicted_class_index.item()]

  # If confident enough, return the matching intent response
  if confidence > confidence_threshold:
    for intent in intents['intents']:
      if intent['tag'] == predicted_tag:
        return random.choice(intent['responses'])

  # Conversational fallback: short follow-ups like "and?", "what about fees?",
  # or pronoun-based queries should try to reuse previous tag if provided.
  lowered = sentence.lower()
  follow_up_tokens = ['and', 'also', 'what about', 'more', 'tell me more', 'details', 'how about', 'that', 'then']
  if prev_tag and any(ft in lowered for ft in follow_up_tokens):
    for intent in intents['intents']:
      if intent['tag'] == prev_tag:
        return random.choice(intent['responses'])

  # If model not confident, try a heuristic overlap search across intent patterns
  best_tag, best_score = _best_intent_by_overlap(sentence_words, intents)
  if best_tag and best_score > 0:
    for intent in intents['intents']:
      if intent['tag'] == best_tag:
        return random.choice(intent['responses'])

  # Last resort
  return "I'm sorry, but I'm not sure how to respond to that."


def ensure_model(model_path='model.pth', num_epochs=20, retrain=False, print_progress=False):
    """Return a loaded model (train one if missing or retrain=True).

    - If `model_path` exists and `retrain` is False, load and return it.
    - Else train a model (uses a reduced default epoch count) and return it.
    """
    if os.path.exists(model_path) and not retrain:
        model = load_model(model_path, input_size, hidden_size, output_size)
        if print_progress:
            print(f"Loaded model from {model_path}")
        return model

    # Train and return
    model = train_model(num_epochs=num_epochs, model_path=model_path, print_progress=print_progress)
    return model


if __name__ == '__main__':
    # CLI usage: train (if needed) and start interactive chat
    model_path = 'model.pth'
    model = ensure_model(model_path=model_path, num_epochs=50, retrain=False, print_progress=True)

    test_cases = [
        "What is the admission process for Bennett University?",
        "Tell me about the B.Tech in Computer Science.",
        "What are the hostel facilities like?",
        "What is the average student-to-faculty ratio at the university?",
    ]
    print('\nTesting Cases starting..........\n')

    for i in test_cases:
        print(f"USER: {i}")
        text = generate_response(i.lower(), model, words, classes)
        print("BOT: ")
        print(textwrap.fill(text, width=70))
        print('-------------------------------------')

    print('\n' + '='*20)
    print('START INTERACTIVE CHAT')
    print('='*20)

    print('Hello! I am a chatbot. How can I help you today? Type "quit" to exit.')
    while True:
        user_input = input('> ')
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        response = generate_response(user_input, model, words, classes)
        print(response)