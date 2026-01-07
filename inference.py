import onnxruntime as ort
import numpy as np

# 1. Load the ONNX Models
# Providers can be ['CUDAExecutionProvider', 'CPUExecutionProvider'] if you have GPU setup
enc_session = ort.InferenceSession("v3-encoder.onnx", providers=['CPUExecutionProvider'])
dec_session = ort.InferenceSession("v3-decoder.onnx", providers=['CPUExecutionProvider'])

idx2char={0: '<PAD>',
 1: '<SOS>',
 2: '<EOS>',
 3: '<UNK>',
 4: 'ค',
 5: 'อ',
 6: 'ม',
 7: 'พ',
 8: 'ิ',
 9: 'ว',
 10: 'เ',
 11: 'ต',
 12: 'ร',
 13: '์',
 14: 'ซ',
 15: 'ฟ',
 16: 'แ',
 17: 'ฮ',
 18: 'า',
 19: 'ด',
 20: 'น',
 21: 'ท',
 22: '็',
 23: 'บ',
 24: 'ไ',
 25: '้',
 26: 'ส',
 27: 'ป',
 28: 'ล',
 29: 'ช',
 30: 'ั',
 31: 'จ',
 32: 'โ',
 33: 'ห',
 34: 'ี',
 35: 'ย',
 36: 'ธ',
 37: 'ุ',
 38: 'ก',
 39: 'ง',
 40: '่',
 41: 'ู',
 42: '๊',
 43: 'ะ',
 44: ' ',
 45: 'ำ',
 46: 'ถ',
 47: 'ึ',
 48: 'ฉ',
 49: '1',
 50: 'ญ',
 51: '-',
 52: 'ฃ',
 53: '9',
 54: '3',
 55: 'ศ',
 56: 'ฌ',
 57: 'y',
 58: 'l',
 59: 'o',
 60: 'n',
 61: 'ข',
 62: 'ฝ'}

class Vocabulary:
    def __init__(self):
        self.char2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.idx2char = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.n_chars = 4

    def add_sentence(self, sentence):
        for char in str(sentence):
            if char not in self.char2idx:
                self.char2idx[char] = self.n_chars
                self.idx2char[self.n_chars] = char
                self.n_chars += 1

    def encode(self, sentence, max_len=None):
        indices = [self.char2idx.get(c, 3) for c in str(sentence)]
        indices = [1] + indices + [2] # Add SOS and EOS
        return indices

input_vocab=Vocabulary()
input_vocab.char2idx={'<PAD>': 0,
 '<SOS>': 1,
 '<EOS>': 2,
 '<UNK>': 3,
 'c': 4,
 'o': 5,
 'm': 6,
 'p': 7,
 'u': 8,
 't': 9,
 'e': 10,
 'r': 11,
 's': 12,
 'f': 13,
 'w': 14,
 'a': 15,
 'h': 16,
 'd': 17,
 'i': 18,
 'n': 19,
 'b': 20,
 'v': 21,
 'l': 22,
 'g': 23,
 'k': 24,
 'y': 25,
 'j': 26,
 'x': 27,
 'z': 28,
 ' ': 29,
 'q': 30,
 'é': 31,
 "'": 32,
 '-': 33,
 'ฺ': 34,
 'è': 35,
 '岸': 36,
 '田': 37,
 '文': 38,
 '雄': 39,
 '.': 40,
 '1': 41,
 '7': 42,
 '3': 43,
 '9': 44,
 '–': 45}


def transliterate_onnx(word, max_len=20):
    # ==========================================
    # Step 1: Preprocess Input (Encoder)
    # ==========================================
    # Encode string to indices [SOS, chars, EOS]
    # Note: Ensure your Vocabulary.encode method returns a list of integers
    src_indices = input_vocab.encode(word) 
    
    # Convert to NumPy: Shape [1, seq_len] (Batch size 1)
    src_input = np.array([src_indices], dtype=np.int64)
    
    # Run Encoder
    # ONNX Input name: 'src' (defined during export)
    # ONNX Output name: 'context'
    ort_inputs = {enc_session.get_inputs()[0].name: src_input}
    context = enc_session.run(None, ort_inputs)[0] # Shape: [1, 1, hidden_dim]

    # ==========================================
    # Step 2: Initialize Decoder Loop
    # ==========================================
    # Initial Hidden State is the Context from Encoder
    hidden = context 
    
    # Start Token <SOS> (Index 1 based on your training script)
    curr_token = np.array([1], dtype=np.int64) 
    
    decoded_indices = []
    
    # ==========================================
    # Step 3: Autoregressive Loop
    # ==========================================
    for _ in range(max_len):
        # Prepare inputs for Decoder
        # Inputs: input (token), hidden, context
        dec_inputs = {
            dec_session.get_inputs()[0].name: curr_token, # [1]
            dec_session.get_inputs()[1].name: hidden,     # [1, 1, 256]
            dec_session.get_inputs()[2].name: context     # [1, 1, 256]
        }
        
        # Run Decoder
        # Outputs: prediction [1, vocab_size], new_hidden [1, 1, 256]
        outputs = dec_session.run(None, dec_inputs)
        prediction = outputs[0]
        hidden = outputs[1] # Update hidden state for next step
        
        # Greedy Search: Pick the index with highest probability
        best_token_idx = np.argmax(prediction, axis=1)[0]
        
        # Check for <EOS> (Index 2)
        if best_token_idx == 2:
            break
            
        decoded_indices.append(best_token_idx)
        
        # Update current token for next iteration
        curr_token = np.array([best_token_idx], dtype=np.int64)

    # ==========================================
    # Step 4: Convert Indices to String
    # ==========================================
    result = ""
    for idx in decoded_indices:
        # Use your vocab's idx2char dictionary
        char = idx2char.get(idx, "")
        result += char
        
    return result

# ==========================================
# Test It
# ==========================================
test_words = ["hello", "computer", "bangkok", "padthai","cat"]

print(f"{'English':<15} | {'Thai (Predicted)':<15}")
print("-" * 35)

for word in test_words:
    thai_pred = transliterate_onnx(word)
    print(f"{word:<15} | {thai_pred:<15}")
