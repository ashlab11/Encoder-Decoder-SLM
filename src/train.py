from src.sft import sft
from src.seq_to_seq import seq_to_seq
from src.span import span_corruption

def train(output_dir: str, output_file_name: str):
    sft(output_dir, output_file_name)
    seq_to_seq(output_dir, output_file_name)
    span_corruption(output_dir, output_file_name)

if __name__ == "__main__":
    train(output_dir="models/base", output_file_name="trained")