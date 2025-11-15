from ai import ensure_model, preprocess_sentence, sentence_to_features, generate_response, words, classes, input_size, hidden_size, output_size

model = ensure_model(model_path='model.pth', num_epochs=1, retrain=False, print_progress=False)

samples = [
    "What is the admission process for Bennett University?",
    "Tell me about the B.Tech in Computer Science.",
    "What are the hostel facilities like?",
    "When is the last date to apply?",
    "Is there an entrance exam?",
]

for s in samples:
    print('---')
    print('INPUT:', s)
    tokens = preprocess_sentence(s.lower())
    print('TOKENS:', tokens)
    in_vocab = [t for t in tokens if t in words]
    print('TOKENS IN VOCAB:', in_vocab)
    feat = sentence_to_features(in_vocab, words)
    print('FEATURES SUM (number of matched words):', int(feat.sum().item()))
    with __import__('torch').no_grad():
        out = model(feat)
    print('MODEL OUTPUT (softmax probs):', out.tolist())
    probs, idx = __import__('torch').max(out, dim=1)
    print('PREDICTED CLASS INDEX:', int(idx.item()), 'CLASS:', classes[int(idx.item())], 'CONF:', float(probs.item()))
    resp = generate_response(s.lower(), model, words, classes)
    print('GENERATE_RESPONSE:', resp)

print('---')
print('VOCAB SIZE:', len(words))
print('CLASSES:', classes)
