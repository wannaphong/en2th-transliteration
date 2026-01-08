import onnxruntime as ort
import numpy as np

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

# 1. Configuration (Must match your training vocab)
# Replace this with the JSON you exported earlier if it's different
# This is just a placeholder example based on standard characters
class Vocabulary:
    def __init__(self):
        # You should load the actual JSON here in production
        self.char2idx = char2idx
        self.idx2char = idx2char
        # ... (You must populate this with your actual characters) ...

    def encode(self, sentence):
        # Helper to encode string to [SOS, indices, EOS]
        return [1] + [self.char2idx.get(c, 3) for c in sentence] + [2]

# 2. Load the Smart Models
# Note: Ensure you are loading the '_attn.onnx' versions
enc_session = ort.InferenceSession("v8-encoder_attn.onnx", providers=['CPUExecutionProvider'])
dec_session = ort.InferenceSession("v8-decoder_attn.onnx", providers=['CPUExecutionProvider'])

def transliterate_smart(word, max_len=25):
    # ==========================================
    # Step A: Encoder
    # ==========================================
    # 1. Prepare Input
    src_indices = [1] + [char2idx.get(c, 3) for c in word.lower()] + [2]
    src_input = np.array([src_indices], dtype=np.int64) # Shape: [1, seq_len]

    # 2. Run Encoder
    # Returns:
    # - encoder_outputs: [1, seq_len, hidden*2] (Used for Attention)
    # - hidden: [1, 1, hidden] (Used to init Decoder)
    ort_inputs = {enc_session.get_inputs()[0].name: src_input}
    enc_results = enc_session.run(None, ort_inputs)

    encoder_outputs = enc_results[0]
    hidden = enc_results[1]

    # ==========================================
    # Step B: Decoder Loop (with Attention)
    # ==========================================
    curr_token = np.array([1], dtype=np.int64) # <SOS>
    decoded_chars = []

    for _ in range(max_len):
        # Inputs: input, hidden, encoder_outputs
        dec_inputs = {
            dec_session.get_inputs()[0].name: curr_token,      # [1]
            dec_session.get_inputs()[1].name: hidden,          # [1, 1, hidden]
            dec_session.get_inputs()[2].name: encoder_outputs  # [1, seq_len, hidden*2]
        }

        # Run Decoder
        outputs = dec_session.run(None, dec_inputs)

        prediction_logits = outputs[0] # [1, vocab_size]
        hidden = outputs[1]            # Update hidden state

        # Greedy Search (Argmax)
        best_token_idx = np.argmax(prediction_logits)

        # Check EOS
        if best_token_idx == 2:
            break

        decoded_chars.append(idx2char.get(best_token_idx, ""))
        curr_token = np.array([best_token_idx], dtype=np.int64)

    return "".join(decoded_chars)

test_words = ["hello", "computer", "bangkok", "pie","cat","unknow","sunset"]

print(f"{'English':<15} | {'Thai (Predicted)':<15}")
print("-" * 35)

for word in test_words:
    thai_pred = transliterate_smart(word)
    print(f"{word:<15} | {thai_pred:<15}")
