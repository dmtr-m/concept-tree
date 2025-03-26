# %%
import os
from pathlib import Path

input_directory = Path("/home/kdemyokhin_1/concept-tree-course-work/articles_parsed/arxiv-txt-cs")
output_directory = Path("/home/kdemyokhin_1/concept-tree-course-work/articles_anaphora_resolved/arxiv-txt-cs")

# %%
import spacy
import coreferee
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import List, Tuple
import os
import sys
from tqdm import tqdm


class CoreferenceResolver:
    """Handles coreference resolution with SpaCy and coreferee."""
    
    def __init__(self):
        self.nlp = self._initialize_spacy_model()
    
    @staticmethod
    def _initialize_spacy_model() -> spacy.language.Language:
        """Initialize SpaCy model with disabled unnecessary components."""
        nlp = spacy.load(
            'en_core_web_trf',
            disable=['tagger', 'parser', 'attribute_ruler', 'lemmatizer']
        )
        nlp.add_pipe('coreferee')
        return nlp
    
    def resolve_text(self, text: str) -> str:
        """Resolve coreferences in the given text."""
        doc = self.nlp(text)
        resolved_tokens = []
        
        for token in doc:
            resolved = doc._.coref_chains.resolve(token)
            resolved_tokens.extend(resolved if resolved else [token])
        
        return ' '.join(t.text for t in resolved_tokens)


class FileProcessor:
    """Processes files with coreference resolution."""
    
    @staticmethod
    def process_single_file(
        input_path: Path,
        output_path: Path,
        resolver: CoreferenceResolver
    ) -> None:
        """Process a single file with coreference resolution."""
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            
            if not text:
                return
                
            resolved_text = resolver.resolve_text(text)
            
            os.makedirs(output_path.parent, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(resolved_text)
                
        except Exception as e:
            print(f"Error processing {input_path}: {str(e)}")
    
    @staticmethod
    def get_file_pairs(
        input_dir: Path,
        output_dir: Path,
        file_extension: str = "*.txt"
    ) -> List[Tuple[Path, Path]]:
        """Generate input-output file path pairs."""
        input_files = list(input_dir.rglob(file_extension))
        return [
            (in_file, output_dir / in_file.relative_to(input_dir))
            for in_file in input_files
        ]


class ParallelProcessor:
    """Handles parallel file processing."""
    
    def __init__(self):
        self.resolver = None
    
    def initialize_worker(self) -> None:
        """Initialize worker process with SpaCy model."""
        self.resolver = CoreferenceResolver()
    
    def process_file_wrapper(self, file_pair: Tuple[Path, Path]) -> None:
        """Wrapper for processing a single file in parallel."""
        input_path, output_path = file_pair
        FileProcessor.process_single_file(input_path, output_path, self.resolver)
    
    def process_in_parallel(
        self,
        file_pairs: List[Tuple[Path, Path]],
        num_processes: int = cpu_count()
    ) -> None:
        """Process files in parallel using multiprocessing."""
        print(f"Starting processing with {num_processes} processes...")
        
        with Pool(
            processes=num_processes,
            initializer=self.initialize_worker
        ) as pool:
            for i, _ in tqdm(enumerate(pool.imap_unordered(
                self.process_file_wrapper, file_pairs, chunksize=1
            )), len(file_pairs)):
                if i % 100 == 0:
                    print(f"Processed {i}/{len(file_pairs)} files...")
        
        print("Processing complete!")


def main(input_directory: str, output_directory: str, cpu_count) -> None:
    """Main execution function."""
    input_path = Path(input_directory)
    output_path = Path(output_directory)
    
    # Prepare file list
    file_pairs = FileProcessor.get_file_pairs(input_path, output_path)
    print(f"Found {len(file_pairs)} files to process")
    
    # Increase recursion limit for complex documents
    sys.setrecursionlimit(10000)
    
    # Process files
    processor = ParallelProcessor()
    processor.process_in_parallel(file_pairs, cpu_count)

# %%
import warnings
warnings.filterwarnings('ignore')


main(input_directory=input_directory, output_directory=output_directory, )


