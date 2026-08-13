import warnings
from tree_sitter_languages import get_language, get_parser

warnings.filterwarnings("ignore")

class SemanticCodeParser:
    """
    Tree-sitter based semantic feature extractor.
    Decomposes code into a list of key semantic features
    (imports, signatures, behaviors, data).
    """
    def __init__(self):
        self.language = get_language('python')
        self.parser = get_parser('python')
        
    def parse_features(self, code_str):
        if not code_str or not isinstance(code_str, str):
            return []

        tree = self.parser.parse(bytes(code_str, "utf8"))
        root_node = tree.root_node
        
        features = []
        
        # 1. Extract imports (dependencies)
        # Many malicious behaviors can be identified directly from imported libraries
        import_query = self.language.query("""
            (import_statement) @import
            (import_from_statement) @import_from
        """)
        for node, _ in import_query.captures(root_node):
            text = code_str[node.start_byte:node.end_byte]
            features.append(f"[Dependency] {text}")

        # 2. Extract function signatures (capability boundaries)
        func_def_query = self.language.query("""
            (function_definition
                name: (identifier) @func.name
                parameters: (parameters) @func.params) @func.def
        """)
        for node, _ in func_def_query.captures(root_node):
            # Find the end position of the parameter list; keep only "def ... (...)"
            params_node = node.child_by_field_name('parameters')
            if params_node:
                sig_end = params_node.end_byte
                signature = code_str[node.start_byte:sig_end] + ")"
                features.append(f"[Signature] {signature}")

        # 3. Extract API calls (core behaviors)
        # This is the most important part: detect os.system, requests.post, etc.
        call_query = self.language.query("""
            (call
                function: (attribute) @call.name
                arguments: (argument_list) @call.args
            ) @call.stmt
        """)
        
        for node, capture_name in call_query.captures(root_node):
            if capture_name == 'call.stmt':
                call_text = code_str[node.start_byte:node.end_byte]
                # Simple filter: keep only calls that carry some semantics (contain ".")
                if "." in call_text:
                    # Compress whitespace
                    clean_call = " ".join(call_text.split())
                    features.append(f"[Behavior] {clean_call}")

        # 4. Extract sensitive strings (data)
        string_query = self.language.query("""
            (string) @str_literal
        """)
        sensitive_keywords = ['/', 'http', 'rm ', 'sudo', 'cmd', 'key', 'token', 'passwd']
        for node, _ in string_query.captures(root_node):
            text = code_str[node.start_byte:node.end_byte]
            # Only extract when the string looks like a path, URL, or command
            if len(text) > 4 and any(k in text for k in sensitive_keywords):
                 features.append(f"[Data] {text}")

        # 5. Extract return statements (output)
        return_query = self.language.query("""
            (return_statement) @return
        """)
        for node, _ in return_query.captures(root_node):
            stmt = code_str[node.start_byte:node.end_byte]
            features.append(f"[Return] {stmt}")

        # Deduplicate
        seen = set()
        unique_features = []
        for f in features:
            if f not in seen:
                unique_features.append(f)
                seen.add(f)
                
        # Fallback: if no features were extracted and the code is non-empty, split by line
        if not unique_features and code_str.strip():
            unique_features = [line.strip() for line in code_str.split('\n') if len(line.strip()) > 10]

        return unique_features

# Test
if __name__ == "__main__":
    parser = SemanticCodeParser()
    code = """
    import os
    def hack():
        os.system("rm -rf /")
        return True
    """
    print(parser.parse_features(code))
    # Expected output:
    # ['[Dependency] import os', '[Signature] def hack()',
    #  '[Behavior] os.system("rm -rf /")', '[Data] "rm -rf /"', '[Return] return True']