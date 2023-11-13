import os
import lzma
from tqdm import tqdm

class OpenWebTextProcessor:
    """
    A class to process and extract text files from the OpenWebText Corpus.
    This class handles the extraction of text from compressed files, splitting
    the data into training and validation sets, and compiling a unique set of
    characters (vocabulary) found in the text.
    """
    
    def __init__(self, folder_path, output_file_train, output_file_val, vocab_file):
        """
        Initializes the OpenWebTextProcessor with directory paths for the data
        and output files.
        """
        self.folder_path = folder_path
        self.output_file_train = output_file_train
        self.output_file_val = output_file_val
        self.vocab_file = vocab_file
        self.vocab = set()
        print("OpenWebTextProcessor initialized.")

    def _xz_files_in_dir(self, directory):
        """
        Private method to list all .xz files in a specified directory.
        """
        files = []
        for filename in os.listdir(directory):
            if filename.endswith(".xz") and os.path.isfile(os.path.join(directory, filename)):
                files.append(filename)
        print(f"Found {len(files)} .xz files in {directory}.")
        return files

    def _process_files(self, files, output_file):
        """
        Private method to process a list of files, extracting their text.
        """
        print(f"Processing {len(files)} files for {output_file}...")
        with open(output_file, "w", encoding="utf-8") as outfile:
            for filename in tqdm(files, total=len(files), desc=f"Processing {output_file}"):
                file_path = os.path.join(self.folder_path, filename)
                try:
                    with lzma.open(file_path, "rt", encoding="utf-8") as infile:
                        text = infile.read()
                        outfile.write(text)
                        self.vocab.update(set(text))
                except IOError as e:
                    print(f"Error reading file {filename}: {e}")
        print(f"Processing complete. Data saved to {output_file}.")

    def split_files(self):
        """
        Splits the dataset into training and validation sets.
        """
        files = self._xz_files_in_dir(self.folder_path)
        split_index = int(len(files) * 0.9)  # 90% for training
        files_train = files[:split_index]
        files_val = files[split_index:]

        print("Starting file splitting for training and validation sets.")
        self._process_files(files_train, self.output_file_train)
        self._process_files(files_val, self.output_file_val)
        print("File splitting complete.")

    def write_vocab(self):
        """
        Writes the accumulated vocabulary to a file.
        """
        print(f"Writing vocabulary to {self.vocab_file}...")
        try:
            with open(self.vocab_file, "w", encoding="utf-8") as vfile:
                for char in sorted(self.vocab):
                    vfile.write(char + '\n')
            print(f"Vocabulary written to {self.vocab_file}.")
        except IOError as e:
            print(f"Error writing to vocab file: {e}")

    def run(self):
        """
        Executes the entire file processing flow.
        """
        print("Starting OpenWebText processing...")
        self.split_files()
        self.write_vocab()
        print("OpenWebText processing completed.")

# Usage example
processor = OpenWebTextProcessor(
    "./openwebtext", 
    "train_split.txt", 
    "val_split.txt", 
    "vocab.txt"
)
processor.run()
