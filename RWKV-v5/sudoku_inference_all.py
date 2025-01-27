import os
import pandas as pd

os.environ["RWKV_JIT_ON"] = "1"
os.environ["RWKV_CUDA_ON"] = "0"

from rwkv_model import RWKV
# from rwkv.model import RWKV # models trained with the original RWKV-LM repo
from rwkv.utils import PIPELINE, PIPELINE_ARGS
from rwkv.rwkv_tokenizer import TRIE_TOKENIZER

# L6-D256-x060
# L8-D320-x060
# L12-D768-x060
model = RWKV(model="out/L12-D768-x060/rwkv-0.pth", strategy="cuda fp16", verbose=False)
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")
pipeline.tokenizer = TRIE_TOKENIZER("sudoku_vocab.txt")
gen_args = PIPELINE_ARGS(top_k=1, alpha_frequency=0, alpha_presence=0, token_stop=[105])

# make sure your sudoku has exactly one solution. launch.py will verify this automatically.
input_str = '''<input>
0 0 8 1 6 7 0 2 0 
5 0 0 2 3 0 0 0 0 
7 6 0 0 5 4 8 0 1 
8 7 0 0 4 0 0 0 0 
0 2 0 0 0 0 0 0 0 
0 0 4 0 0 3 0 9 0 
0 0 0 0 0 0 3 7 0 
0 4 0 0 0 0 0 8 0 
3 1 0 8 0 6 9 0 4 
</input>

'''
import pandas as pd

# Read the CSV file
df = pd.read_csv("/Users/bosonphoton/rmkv/easy100.csv")

# Display the first few rows of the DataFrame
print(df)

all_samples = [] # array of 100 input strings


for i in range(100):
    print(f'{" Model input ":-^100}\n{all_samples[i]}\n{" Model output ":-^100}')
    pipeline.generate(all_samples[i], token_count=50000, args=gen_args, callback=lambda x: print(x, end="", flush=True))


