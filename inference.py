import onnxruntime as ort
import numpy as np

# 1. Load the ONNX Models
# Providers can be ['CUDAExecutionProvider', 'CPUExecutionProvider'] if you have GPU setup
enc_session = ort.InferenceSession("v4-encoder.onnx", providers=['CPUExecutionProvider'])
dec_session = ort.InferenceSession("v4-decoder.onnx", providers=['CPUExecutionProvider'])

idx2char={0: '<PAD>',
 1: '<SOS>',
 2: '<EOS>',
 3: '<UNK>',
 4: 'แ',
 5: 'อ',
 6: 'ค',
 7: 'ช',
 8: 'ั',
 9: '่',
 10: 'น',
 11: 'ท',
 12: 'ี',
 13: 'ฟ',
 14: 'ะ',
 15: 'ด',
 16: 'ป',
 17: 'เ',
 18: 'ต',
 19: 'ร',
 20: '์',
 21: 'ล',
 22: 'บ',
 23: '้',
 24: 'ม',
 25: 'ก',
 26: 'ฮ',
 27: 'า',
 28: 'ิ',
 29: 'ึ',
 30: 'พ',
 31: 'ส',
 32: 'ซ',
 33: 'โ',
 34: '็',
 35: 'ง',
 36: 'ุ',
 37: '๊',
 38: 'ู',
 39: 'ว',
 40: 'ไ',
 41: 'ย',
 42: 'จ',
 43: 'ห',
 44: 'ศ',
 45: 'ธ',
 46: 'ฤ',
 47: 'ษ',
 48: 'ฎ',
 49: '๋',
 50: ' ',
 51: 'ำ',
 52: 'ถ',
 53: 'ฉ',
 54: '1',
 55: 'ญ',
 56: '-',
 57: 'ฃ',
 58: '9',
 59: '3',
 60: 'ฌ',
 61: 'y',
 62: 'l',
 63: 'o',
 64: 'n',
 65: 'ข',
 66: 'ฝ'}

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
 'a': 4,
 'c': 5,
 't': 6,
 'i': 7,
 'o': 8,
 'n': 9,
 'v': 10,
 'e': 11,
 'd': 12,
 'p': 13,
 'r': 14,
 'l': 15,
 'b': 16,
 'u': 17,
 'm': 18,
 'h': 19,
 'g': 20,
 's': 21,
 'k': 22,
 'y': 23,
 'x': 24,
 'f': 25,
 'w': 26,
 'j': 27,
 'z': 28,
 'q': 29,
 '-': 30,
 ' ': 31,
 'é': 32,
 "'": 33,
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
test_words = ["hello", "computer", "bangkok", "pie","cat","unknow","sunset"]

print(f"{'English':<15} | {'Thai (Predicted)':<15}")
print("-" * 35)

for word in test_words:
    thai_pred = transliterate_onnx(word)
    print(f"{word:<15} | {thai_pred:<15}")
