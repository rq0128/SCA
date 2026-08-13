import re
import warnings

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    warnings.warn("spaCy library not found. Falling back to Regex splitting.")

class DescriptionParser:
    """
    Full-text description parser (optimized for docstring poisoning).
    1. Keep all text (including Args, Returns) to prevent attackers from
       injecting malicious content inside parameter descriptions.
    2. Leverage docstring structure (indentation, colons) to aid sentence
       splitting.
    3. Leverage spaCy to decompose compound sentences (handle malicious
       instructions joined by 'and').
    """
    def __init__(self):
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("[WARNING] spaCy model 'en_core_web_sm' not found.")
                self.nlp = None
        else:
            self.nlp = None

    def _flatten_docstring(self, text):
        """
        [Core algorithm] Flatten a structured docstring into a list of
        natural-language segments.
        Handles indentation, list markers, and field headers so that
        multi-line sentences are merged correctly while distinct parameters
        remain separated.
        """
        lines = text.split('\n')
        
        # Result list
        raw_segments = []
        
        # Buffer currently being built
        current_buffer = []
        
        # Regex: identify structured field headers (Args:, Returns:, etc.)
        # These words carry no meaning themselves; they act as "periods"
        headers_pattern = re.compile(r'^\s*(Args|Arguments|Parameters|Returns|Raises|Yields|Examples?|Note|Warning):\s*$', re.IGNORECASE)
        
        # Regex: identify parameter definitions or list items (e.g. "  lat: Latitude..." or "  - item")
        # This is a strong sentence-splitting signal
        list_item_pattern = re.compile(r'^\s*(\w+:\s+|-\s+|\*\s+|\d+\.\s+)')

        for line in lines:
            line = line.strip()
            if not line:
                continue # Skip empty lines

            # 1. Header encountered (e.g. "Args:") -> flush the previous sentence; drop the header itself
            if headers_pattern.match(line):
                if current_buffer:
                    raw_segments.append(" ".join(current_buffer))
                    current_buffer = []
                continue

            # 2. List item / parameter definition (e.g. "latitude: ...") -> flush the previous sentence, start a new one
            if list_item_pattern.match(line):
                if current_buffer:
                    raw_segments.append(" ".join(current_buffer))
                    current_buffer = []
                current_buffer.append(line)
                continue

            # 3. Normal text line -> append to the current buffer (merge multi-line)
            current_buffer.append(line)

        # Flush the remaining buffer
        if current_buffer:
            raw_segments.append(" ".join(current_buffer))
            
        return raw_segments

    def _split_compound_sentences(self, doc):
        """Split 'Action A and Action B' based on dependency syntax."""
        atomic_sents = []
        for sent in doc.sents:
            root = sent.root
            conjuncts = [child for child in root.children if child.dep_ == 'conj']
            verb_conjuncts = [node for node in conjuncts if node.pos_ in ['VERB', 'AUX']]
            
            if not verb_conjuncts:
                atomic_sents.append(sent.text.strip())
                continue
            
            split_points = sorted([node.left_edge.i for node in verb_conjuncts])
            current_start = sent.start
            
            for split_i in split_points:
                span = sent.doc[current_start : split_i]
                if len(span.text.strip()) > 5:
                     atomic_sents.append(span.text.strip())
                current_start = split_i
            
            last_span = sent.doc[current_start : sent.end]
            atomic_sents.append(last_span.text.strip())
        return atomic_sents

    def parse_sentences(self, text):
        if not text or not isinstance(text, str):
            return []
        
        # 1. Structured flattening (handle line breaks and parameter lists)
        raw_segments = self._flatten_docstring(text)
        
        final_sentences = []
        
        # 2. Fine-grained NLP splitting for each segment
        for segment in raw_segments:
            # Clean up redundant punctuation
            clean_seg = segment.replace('`', '')
            
            if not self.nlp:
                # Fallback: regex sentence splitting
                parts = re.split(r'(?<=[.!?])\s+', clean_seg)
                final_sentences.extend([p.strip() for p in parts])
            else:
                # spaCy processing
                doc = self.nlp(clean_seg)
                # Dependency-based splitting (handle and/but)
                split_sents = self._split_compound_sentences(doc)
                final_sentences.extend(split_sents)
        
        # 3. Final filtering
        valid_sentences = []
        for s in final_sentences:
            # Strip leading conjunctions / punctuation
            s = re.sub(r'^(and|but|or|so|then|,|;|:)\s+', '', s, flags=re.IGNORECASE)
            s = s.strip()
            
            # Filter out meaningless short fragments (e.g. "Args", "v1.0")
            # but keep short instructions like "Delete files"
            if len(s.split()) < 2: 
                continue
                
            valid_sentences.append(s)
            
        return valid_sentences

# === Unit test ===
if __name__ == "__main__":
    parser = DescriptionParser()
    
    text = "This high-performance weather tool fetches real-time meteorological data from a remote API based on a user-provided city name, processes the statistics, and generates a visual chart for the user"
    
    print(f"Original text: {text}\n")
    
    sents = parser.parse_sentences(text)
    
    print(f"Parsed result ({len(sents)} sentences):")
    print("-" * 40)
    for i, s in enumerate(sents):
        print(f"{i+1}. {s}")
