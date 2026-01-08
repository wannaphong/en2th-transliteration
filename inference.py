import onnxruntime as ort
import numpy as np

# ==========================================
# 1. Configuration (Match your training)
# ==========================================
# REPLACE THIS with the JSON you exported from the training script!
# This is just a placeholder to make the code run.

char2idx={'<PAD>': 0,
 '<SOS>': 1,
 '<EOS>': 2,
 '<UNK>': 3,
 'a': 4,
 'r': 5,
 'e': 6,
 'b': 7,
 'c': 8,
 'd': 9,
 'f': 10,
 'g': 11,
 'h': 12,
 'i': 13,
 'j': 14,
 'k': 15,
 'l': 16,
 'm': 17,
 'n': 18,
 'o': 19,
 'p': 20,
 'q': 21,
 's': 22,
 't': 23,
 'u': 24,
 'v': 25,
 'x': 26,
 'y': 27,
 'z': 28,
 'w': 29,
 '-': 30,
 ' ': 31,
 'é': 32,
 "'": 33,
 'ฺ': 34,
 'è': 35,
 '.': 36,
 '1': 37,
 '7': 38,
 '3': 39,
 '9': 40,
 '–': 41,
 '"': 42,
 '+': 43,
 '/': 44,
 '4': 45,
 '8': 46,
 '6': 47,
 '2': 48,
 '0': 49}
idx2char={0: '<PAD>',
 1: '<SOS>',
 2: '<EOS>',
 3: '<UNK>',
 4: 'อ',
 5: 'า',
 6: 'ะ',
 7: 'บ',
 8: 'ี',
 9: 'ซ',
 10: 'ด',
 11: 'เ',
 12: 'ฟ',
 13: 'จ',
 14: 'ช',
 15: 'ไ',
 16: 'ค',
 17: 'แ',
 18: 'ล',
 19: '็',
 20: 'ม',
 21: 'น',
 22: 'โ',
 23: 'พ',
 24: 'ิ',
 25: 'ว',
 26: 'ร',
 27: '์',
 28: 'ส',
 29: 'ท',
 30: 'ย',
 31: 'ู',
 32: 'ก',
 33: 'ต',
 34: 'ป',
 35: '่',
 36: 'ั',
 37: 'ง',
 38: '้',
 39: 'ฮ',
 40: '๊',
 41: 'ถ',
 42: 'ห',
 43: 'ธ',
 44: 'ึ',
 45: 'ุ',
 46: 'ศ',
 47: 'ฤ',
 48: 'ษ',
 49: 'ฎ',
 50: '๋',
 51: ' ',
 52: 'ำ',
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
 66: 'ฝ',
 67: "'",
 68: 'ฺ',
 69: 'ื',
 70: 'ใ',
 71: '/',
 72: 'ฯ',
 73: 'ๆ',
 74: '`',
 75: 'ณ',
 76: '๎',
 77: ':',
 78: '*',
 79: 'ผ'}
class Vocabulary:
    def __init__(self):
        # Example mappings (YOU MUST USE YOUR REAL VOCAB HERE)
        self.char2idx = char2idx
        self.idx2char = idx2char
        # Load your actual vocab dicts here...
    
    def encode(self, sentence):
        return [1] + [self.char2idx.get(c, 3) for c in sentence] + [2]

# ==========================================
# 2. Load Transformer ONNX Models
# ==========================================
# Ensure you are using the Transformer exports (tf_encoder.onnx, tf_decoder.onnx)
import onnxruntime as ort
import numpy as np

# Load the FIXED models (exported with MAX_LEN=50)
enc_session = ort.InferenceSession("v10-tf_encoder_fixed.onnx", providers=['CPUExecutionProvider'])
dec_session = ort.InferenceSession("v10-tf_decoder_fixed.onnx", providers=['CPUExecutionProvider'])

MAX_LEN = 50  # MUST match the number used in export
vocab=Vocabulary()

def transliterate(word):
    # ==========================================
    # 1. PADDING LOGIC (Crucial Fix)
    # ==========================================
    # Encode: [SOS, ... chars ..., EOS]
    src_indices = vocab.encode(word.lower())
    
    # Calculate how much padding is needed
    pad_len = MAX_LEN - len(src_indices)
    
    if pad_len > 0:
        # Pad with 0s (assuming 0 is <PAD> in your vocab)
        src_indices += [0] * pad_len
    else:
        # Truncate if word is too long
        src_indices = src_indices[:MAX_LEN]
        
    # Shape becomes [1, 50] exactly
    src_input = np.array([src_indices], dtype=np.int64)

    # ==========================================
    # 2. Run Encoder
    # ==========================================
    ort_inputs = {enc_session.get_inputs()[0].name: src_input}
    enc_results = enc_session.run(None, ort_inputs)
    memory = enc_results[0] # Shape [1, 50, 128]

    # ==========================================
    # 3. Run Decoder (Also Needs Padding)
    # ==========================================
    # Initialize with [SOS, 0, 0, 0, ...]
    generated_tokens = [1] + [0] * (MAX_LEN - 1)
    
    current_idx = 0 # Pointer to the token we are currently generating
    result = ""
    
    for _ in range(25): # Loop up to max output length
        trg_input = np.array([generated_tokens], dtype=np.int64)
        
        dec_inputs = {
            dec_session.get_inputs()[0].name: trg_input,
            dec_session.get_inputs()[1].name: memory
        }
        
        # Run Decoder
        outputs = dec_session.run(None, dec_inputs)
        predictions = outputs[0] # [1, 50, vocab_size]
        
        # Get the logits for the CURRENT position
        # If we are at index 0 (<SOS>), we want the prediction for index 1
        logits = predictions[0, current_idx, :]
        
        best_token_idx = np.argmax(logits)
        
        if best_token_idx == 2: # EOS
            break
            
        # Write the prediction into the NEXT position in the buffer
        current_idx += 1
        if current_idx < MAX_LEN:
            generated_tokens[current_idx] = best_token_idx
            
        if best_token_idx > 3: # Skip special tokens
            result += vocab.idx2char.get(best_token_idx, "")
            
    return result

# Usage
# print(transliterate_fixed("bangkok", vocab))

test_words = ["hello", "computer", "bangkok", "pie","cat","unknow","sunset"]

print(f"{'English':<15} | {'Thai (Predicted)':<15}")
print("-" * 35)

for word in test_words:
    thai_pred = transliterate(word)
    print(f"{word:<15} | {thai_pred:<15}")
