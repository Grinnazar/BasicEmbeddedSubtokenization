import customtkinter as ctk
import numpy as np
import sentencepiece as spm
from pathlib import Path
from typing import List, Tuple, Optional
import sys


# Parameters
MODEL_PREFIX = "pl_bpe_"  # Prefix for the model
EMBEDDINGS_FILENAME = "embeddings" # Name of the embeddings file
TOP_K_RESULTS = 15 # Number of nearest words to show for the result


# Embedding Operations Logic
class EmbeddingOperations:
    def __init__(self, sp_processor: spm.SentencePieceProcessor, embeddings_matrix: np.ndarray):
        self.sp = sp_processor
        self.emb_full = embeddings_matrix

        if self.sp is None or self.emb_full is None:
            raise ValueError("SentencePiece processor or embeddings matrix not loaded.")

        vocab_size = self.sp.get_piece_size()
        self.emb = self.emb_full[:vocab_size]

        self.emb_norms = np.linalg.norm(self.emb, axis=1)
        self.emb_norms[self.emb_norms < 1e-9] = 1e-9
    
    def find_nearest_words_euclidean(self,
                                     target_vector: np.ndarray,
                                     k: int = TOP_K_RESULTS,
                                     exclude_pieces: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        if target_vector is None:
            return []
        
        target_vector_sq_norm = np.sum(target_vector**2)

        emb_sq_norms = np.sum(self.emb**2, axis=1)
        
        # Obliczanie kwadratu dystansu
        sq_distances = emb_sq_norms - 2 * (self.emb @ target_vector) + target_vector_sq_norm
        sq_distances[sq_distances < 0] = 0
        
        distances = np.sqrt(sq_distances)

        num_to_fetch = k + (len(exclude_pieces) if exclude_pieces else 0) + 10
        num_to_fetch = min(num_to_fetch, self.sp.get_piece_size())
        
        # Sortowanie rosnąco (najmniejszy dystans = największe podobieństwo)
        sorted_indices = np.argsort(distances)
        best_indices = sorted_indices[:num_to_fetch]
        
        results = []
        exclude_pieces_set = set(exclude_pieces) if exclude_pieces else set()
        unk_token_id = self.sp.unk_id()
        pad_token_id = self.sp.pad_id()
        
        for idx_int in best_indices:
            idx = int(idx_int)
            if len(results) >= k: break
            
            piece_text = self.sp.id_to_piece(idx)
            
            if idx == unk_token_id or \
               (pad_token_id != -1 and idx == pad_token_id) or \
               piece_text in exclude_pieces_set:
                continue
                
            results.append((piece_text, float(distances[idx])))  # dystans
            
        return results

    def get_word_vector(self, word: str) -> Tuple[Optional[np.ndarray], List[str], str]:
        if not word.strip():
            return None, [], "Błąd: Pole słowa jest puste."
        pieces = self.sp.encode(word.lower().strip(), out_type=str)
        if not pieces:
            return None, [], f"Tokenizacja: [PUSTA]\nBłąd: Słowo '{word}' nie dało tokenów."
        vectors, valid_pieces_for_vector, tokenization_display_parts = [], [], []
        for piece in pieces:
            piece_id = self.sp.piece_to_id(piece)
            if piece_id != self.sp.unk_id() and 0 <= piece_id < self.emb.shape[0]:
                vectors.append(self.emb[piece_id])
                valid_pieces_for_vector.append(piece)
                tokenization_display_parts.append(f"{piece}")
            else:
                tokenization_display_parts.append(f"{piece}[OOV]")
        tokenization_str = " | ".join(tokenization_display_parts)
        if not vectors:
            return None, pieces, f"Tokenizacja: {tokenization_str}\nBłąd: Brak poprawnych osadzeń dla tokenów '{word}'."
        word_vector = np.sum(vectors, axis=0)
        return word_vector, valid_pieces_for_vector, f"Tokeny: {tokenization_str}"

    def find_nearest_words_to_vector(self,
                                     target_vector: np.ndarray,
                                     k: int = TOP_K_RESULTS,
                                     exclude_pieces: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        if target_vector is None:
            return []
        target_vector_norm = np.linalg.norm(target_vector)
        if target_vector_norm < 1e-9:
            return [("Wektor wynikowy bliski zeru", 0.0)]
        similarities = (self.emb @ target_vector) / (self.emb_norms * target_vector_norm + 1e-9)
        num_to_fetch = k + (len(exclude_pieces) if exclude_pieces else 0) + 10
        num_to_fetch = min(num_to_fetch, self.sp.get_piece_size())
        sorted_indices = np.argsort(similarities)
        best_indices = sorted_indices[-num_to_fetch:][::-1]
        results = []
        exclude_pieces_set = set(exclude_pieces) if exclude_pieces else set()
        unk_token_id = self.sp.unk_id()
        pad_token_id = self.sp.pad_id()
        for idx_int in best_indices:
            idx = int(idx_int)
            if len(results) >= k: break
            piece_text = self.sp.id_to_piece(idx)
            if idx == unk_token_id or \
               (pad_token_id != -1 and idx == pad_token_id) or \
               piece_text in exclude_pieces_set:
                continue
            results.append((piece_text, float(similarities[idx])))
        return results

# Main Application GUI
class WordArithmeticApp(ctk.CTk):
    def __init__(self, embed_ops: EmbeddingOperations):
        super().__init__()
        self.embed_ops = embed_ops
        self.title("Zaawansowana Arytmetyka Słów 📐🧮")
        self.geometry("750x650")
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        # Main frame
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(pady=10, padx=10, fill="both", expand=True)

        # Mode Selection
        mode_frame = ctk.CTkFrame(main_frame)
        mode_frame.pack(pady=10, padx=10, fill="x")
        ctk.CTkLabel(mode_frame, text="Tryb Operacji:", font=ctk.CTkFont(weight="bold", size=18)).pack(side="left", padx=5)
        self.mode_var = ctk.StringVar(value="Analogia")
        modes = ["Analogia", "Operacje Skalarne", "Najbliższe Sąsiedztwo", "Podobieństwo Słów"] # <-- Changed mode name
        self.mode_selector = ctk.CTkSegmentedButton(mode_frame, values=modes, variable=self.mode_var, command=self._on_mode_change)
        self.mode_selector.pack(side="left", padx=5, expand=True, fill="x")

        # Word A
        word_a_frame = ctk.CTkFrame(main_frame)
        word_a_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(word_a_frame, text="Słowo A:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.entry_word_a = ctk.CTkEntry(word_a_frame, width=200, placeholder_text="np. król")
        self.entry_word_a.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.label_token_a = ctk.CTkLabel(word_a_frame, text="Tokeny A: N/A", wraplength=450, justify="left")
        self.label_token_a.grid(row=0, column=2, padx=5, pady=5, sticky="w")
        word_a_frame.columnconfigure(1, weight=1)

        # Mode-Specific Inputs Frame
        self.mode_specific_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        self.mode_specific_frame.pack(pady=5, padx=10, fill="x")
        self._setup_analogy_frame()
        self._setup_scalar_frame()
        self._setup_neighbors_frame()
        self._setup_similarity_frame()

        # Execute Button
        self.execute_button = ctk.CTkButton(main_frame, text="Wykonaj", command=self._perform_action, height=40)
        self.execute_button.pack(pady=10, padx=10, fill="x")

        # Result Frame
        result_frame = ctk.CTkFrame(main_frame)
        result_frame.pack(pady=10, padx=10, fill="both", expand=True)
        ctk.CTkLabel(result_frame, text="Wynik:", font=ctk.CTkFont(weight="bold", size=24)).pack(anchor="w", pady=(0,5))
        self.result_textbox = ctk.CTkTextbox(result_frame, activate_scrollbars=True, font=ctk.CTkFont(size=16))
        self.result_textbox.pack(fill="both", expand=True)
        self.result_textbox.configure(state="disabled")

        self._on_mode_change() # Initial UI setup

    def _setup_analogy_frame(self):
        self.analogy_frame = ctk.CTkFrame(self.mode_specific_frame, fg_color="transparent")
        ctk.CTkLabel(self.analogy_frame, text="Operacja AB:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.operation_ab_var = ctk.StringVar(value="+")
        self.op_menu_ab = ctk.CTkOptionMenu(self.analogy_frame, variable=self.operation_ab_var, values=["+", "-"], width=70)
        self.op_menu_ab.grid(row=0, column=1, padx=5, pady=5)
        ctk.CTkLabel(self.analogy_frame, text="Słowo B:").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.entry_word_b = ctk.CTkEntry(self.analogy_frame, width=150, placeholder_text="np. mężczyzna")
        self.entry_word_b.grid(row=0, column=3, padx=5, pady=5, sticky="ew")
        self.label_token_b = ctk.CTkLabel(self.analogy_frame, text="Tokeny B: N/A", wraplength=250, justify="left")
        self.label_token_b.grid(row=0, column=4, padx=5, pady=5, sticky="w")
        ctk.CTkLabel(self.analogy_frame, text="Operacja BC:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.operation_bc_var = ctk.StringVar(value="+")
        self.op_menu_bc = ctk.CTkOptionMenu(self.analogy_frame, variable=self.operation_bc_var, values=["+", "-"], width=70)
        self.op_menu_bc.grid(row=1, column=1, padx=5, pady=5)
        ctk.CTkLabel(self.analogy_frame, text="Słowo C (Opc.):").grid(row=1, column=2, padx=5, pady=5, sticky="w")
        self.entry_word_c = ctk.CTkEntry(self.analogy_frame, width=150, placeholder_text="np. kobieta")
        self.entry_word_c.grid(row=1, column=3, padx=5, pady=5, sticky="ew")
        self.label_token_c = ctk.CTkLabel(self.analogy_frame, text="Tokeny C: N/A", wraplength=250, justify="left")
        self.label_token_c.grid(row=1, column=4, padx=5, pady=5, sticky="w")
        self.analogy_frame.columnconfigure(3, weight=1)

    def _setup_scalar_frame(self):
        self.scalar_frame = ctk.CTkFrame(self.mode_specific_frame, fg_color="transparent")
        ctk.CTkLabel(self.scalar_frame, text="Operacja Skalarna:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.scalar_op_var = ctk.StringVar(value="+")
        self.scalar_op_menu = ctk.CTkOptionMenu(self.scalar_frame, variable=self.scalar_op_var, values=["+", "-"], width=70)
        self.scalar_op_menu.grid(row=0, column=1, padx=5, pady=5)
        ctk.CTkLabel(self.scalar_frame, text="Wartość Skalarna:").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.entry_scalar_value = ctk.CTkEntry(self.scalar_frame, width=150, placeholder_text="np. 2 lub 0.5")
        self.entry_scalar_value.grid(row=0, column=3, padx=5, pady=5, sticky="ew")
        self.scalar_frame.columnconfigure(3, weight=1)

    def _setup_neighbors_frame(self):
        self.neighbors_frame = ctk.CTkFrame(self.mode_specific_frame, fg_color="transparent")
        ctk.CTkLabel(self.neighbors_frame, text="Znajdź K najbliższych sąsiadów dla Słowa A.").pack(padx=5, pady=5)

    def _setup_similarity_frame(self):
        self.similarity_frame = ctk.CTkFrame(self.mode_specific_frame, fg_color="transparent")
        ctk.CTkLabel(self.similarity_frame, text="Słowo B:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.entry_word_sim_b = ctk.CTkEntry(self.similarity_frame, width=200, placeholder_text="np. królowa")
        self.entry_word_sim_b.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.label_token_sim_b = ctk.CTkLabel(self.similarity_frame, text="Tokeny B: N/A", wraplength=450, justify="left")
        self.label_token_sim_b.grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.similarity_frame.columnconfigure(1, weight=1)

    def _on_mode_change(self, selected_mode: Optional[str] = None):
        if selected_mode is None:
            selected_mode = self.mode_var.get()

        # Hide all frames first
        self.analogy_frame.pack_forget()
        self.scalar_frame.pack_forget()
        self.neighbors_frame.pack_forget()
        self.similarity_frame.pack_forget()

        # Clear old token info if it exists
        self.label_token_b.configure(text="Tokeny B: N/A")
        self.label_token_c.configure(text="Tokeny C: N/A")
        self.label_token_sim_b.configure(text="Tokeny B: N/A")


        if selected_mode == "Analogia":
            self.analogy_frame.pack(fill="x", expand=True)
            self.execute_button.configure(text="Oblicz Analogię")
        elif selected_mode == "Operacje Skalarne":
            self.scalar_frame.pack(fill="x", expand=True)
            self.execute_button.configure(text="Zastosuj Operację Skalarną")
        elif selected_mode == "Najbliższe Sąsiedztwo":
            self.neighbors_frame.pack(fill="x", expand=True)
            self.execute_button.configure(text="Znajdź Najbliższe Słowa")
        elif selected_mode == "Podobieństwo Słów":
            self.similarity_frame.pack(fill="x", expand=True)
            self.execute_button.configure(text="Oblicz Podobieństwo")

        # Clear previous results
        self.result_textbox.configure(state="normal")
        self.result_textbox.delete("1.0", "end")
        self.result_textbox.configure(state="disabled")

    def _display_results(self, header: str, nearest_items: List[Tuple[str, float]], error_msg: Optional[str] = None):
        self.result_textbox.configure(state="normal")
        self.result_textbox.delete("1.0", "end")
        if error_msg:
            self.result_textbox.insert("end", f"BŁĄD: {error_msg}\n")
        self.result_textbox.insert("end", f"{header}:\n")
        if nearest_items:
            for item, score in nearest_items:
                self.result_textbox.insert("end", f"  - {item} (podobieństwo: {score:.4f})\n")
        elif not error_msg:
            self.result_textbox.insert("end", "Nie znaleziono pasujących słów/tokenów.\n")
        self.result_textbox.configure(state="disabled")

    def _perform_action(self):
        self.result_textbox.configure(state="normal")
        self.result_textbox.delete("1.0", "end")
        self.result_textbox.configure(state="disabled")

        mode = self.mode_var.get()
        word_a_str = self.entry_word_a.get()
        vec_a, pieces_a, token_a_info = self.embed_ops.get_word_vector(word_a_str)
        self.label_token_a.configure(text=token_a_info)

        if vec_a is None:
            self._display_results("Błąd przetwarzania Słowa A", [], token_a_info.split('\n', 1)[-1] if '\n' in token_a_info else token_a_info)
            return

        if mode == "Analogia":
            self._calculate_analogy(vec_a, pieces_a)
        elif mode == "Operacje Skalarne":
            self._calculate_scalar_op(vec_a, pieces_a)
        elif mode == "Najbliższe Sąsiedztwo":
            self._find_neighbors_for_word_a(vec_a, pieces_a)
        elif mode == "Podobieństwo Słów":
            self._calculate_similarity(vec_a)
    
    def _display_analogy_results(self, header: str, nearest_cos: List[Tuple[str, float]], nearest_euc: List[Tuple[str, float]]):
        self.result_textbox.configure(state="normal")
        self.result_textbox.delete("1.0", "end")
        
        self.result_textbox.insert("end", f"{header}\n\n")
        
        self.result_textbox.insert("end", "--- Najbliższe słowa (Podobieństwo KOSINUSOWE) ---\n")
        self.result_textbox.insert("end", "(Im bliżej 1.0000, tym lepsze)\n")
        if nearest_cos:
            for item, score in nearest_cos:
                self.result_textbox.insert("end", f"  - {item} (podobieństwo: {score:.4f})\n")
        else:
            self.result_textbox.insert("end", "Nie znaleziono pasujących słów/tokenów.\n")
            
        self.result_textbox.insert("end", "\n--- Najbliższe słowa (DYSTANS EUKLIDESOWY) ---\n")
        self.result_textbox.insert("end", "(Im bliżej 0.0000, tym lepsze)\n")
        if nearest_euc:
            for item, score in nearest_euc:
                # W przypadku euklidesowego, wyświetlamy "dystans"
                self.result_textbox.insert("end", f"  - {item} (dystans: {score:.4f})\n")
        else:
            self.result_textbox.insert("end", "Nie znaleziono pasujących słów/tokenów.\n")
            
        self.result_textbox.configure(state="disabled")

    def _calculate_analogy(self, vec_a: np.ndarray, pieces_a: List[str]):
        word_b_str = self.entry_word_b.get()
        op_ab = self.operation_ab_var.get()
        vec_b, pieces_b, token_b_info = self.embed_ops.get_word_vector(word_b_str)
        self.label_token_b.configure(text=token_b_info)
        if vec_b is None:
            self._display_results("Błąd przetwarzania Słowa B", [], token_b_info.split('\n', 1)[-1] if '\n' in token_b_info else token_b_info)
            return
        
        all_input_pieces = pieces_a + pieces_b
        result_vec = None
        if op_ab == "+": result_vec = vec_a + vec_b
        elif op_ab == "-": result_vec = vec_a - vec_b
        else:
            self._display_results("Błąd", [], "Nieznana operacja AB.")
            return

        word_c_str = self.entry_word_c.get()
        if word_c_str.strip():
            op_bc = self.operation_bc_var.get()
            vec_c, pieces_c, token_c_info = self.embed_ops.get_word_vector(word_c_str)
            self.label_token_c.configure(text=token_c_info)
            if vec_c is None:
                self._display_results("Błąd przetwarzania Słowa C (kontynuowano z A op B)", [], token_c_info.split('\n', 1)[-1] if '\n' in token_c_info else token_c_info)
                # Użyjemy tylko cosine w przypadku błędu na C
                nearest_cos = self.embed_ops.find_nearest_words_to_vector(result_vec, exclude_pieces=all_input_pieces)
                self._display_results(f"Najbliższe słowa (COS) dla '{self.entry_word_a.get()} {op_ab} {self.entry_word_b.get()}'", nearest_cos)
                return
            all_input_pieces.extend(pieces_c)
            if op_bc == "+": result_vec = result_vec + vec_c
            elif op_bc == "-": result_vec = result_vec - vec_c
        else:
            self.label_token_c.configure(text="Tokeny C: N/A (nie podano)")

        header = f"Wektor wynikowy: '{self.entry_word_a.get()}{op_ab}{word_b_str}"
        if word_c_str.strip(): header += f"{self.operation_bc_var.get()}{word_c_str}"
        header += "'"

        # --- NOWA LOGIKA W ANALOGII ---
        
        # 1. Podobieństwo Kosinusowe (im bliżej 1, tym lepsze)
        nearest_cos = self.embed_ops.find_nearest_words_to_vector(result_vec, exclude_pieces=all_input_pieces)
        
        # 2. Dystans Euklidesowy (im bliżej 0, tym lepsze)
        nearest_euc = self.embed_ops.find_nearest_words_euclidean(result_vec, exclude_pieces=all_input_pieces)
        
        self._display_analogy_results(header, nearest_cos, nearest_euc) # <-- NOWA METODA WYŚWIETLANIA

    def _calculate_scalar_op(self, vec_a: np.ndarray, pieces_a: List[str]):
        scalar_op = self.scalar_op_var.get()
        scalar_value_str = self.entry_scalar_value.get()
        try:
            scalar_value = float(scalar_value_str)
        except ValueError:
            self._display_results("Błąd", [], f"Niepoprawna wartość skalarna: '{scalar_value_str}'. Wprowadź liczbę.")
            return

        result_vec = None

        if scalar_op == "+": result_vec = vec_a + scalar_value
        elif scalar_op == "-": result_vec = vec_a - scalar_value
        else:
            self._display_results("Błąd", [], "Nieznana operacja skalarna.")
            return
        
        header = f"Najbliższe słowa dla '{self.entry_word_a.get()} {scalar_op} {scalar_value}'"
        nearest = self.embed_ops.find_nearest_words_to_vector(result_vec, exclude_pieces=pieces_a)
        self._display_results(header, nearest)

    def _find_neighbors_for_word_a(self, vec_a: np.ndarray, pieces_a: List[str]):
        header = f"Najbliższe słowa dla '{self.entry_word_a.get()}'"
        nearest = self.embed_ops.find_nearest_words_to_vector(vec_a, exclude_pieces=pieces_a)
        self._display_results(header, nearest)

    # ++ NEW METHOD ++
    def _calculate_similarity(self, vec_a: np.ndarray):
        word_a_str = self.entry_word_a.get()
        word_b_str = self.entry_word_sim_b.get()

        vec_b, _, token_b_info = self.embed_ops.get_word_vector(word_b_str)
        self.label_token_sim_b.configure(text=token_b_info)

        if vec_b is None:
            self._display_results(f"Błąd przetwarzania Słowa B: '{word_b_str}'", [], token_b_info.split('\n', 1)[-1] if '\n' in token_b_info else token_b_info)
            return

        # --- Calculations ---
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        # Cosine Similarity: Range [-1, 1]. Closer to 1 is more similar.
        if norm_a > 1e-9 and norm_b > 1e-9:
            cosine_sim = np.dot(vec_a, vec_b) / (norm_a * norm_b)
        else:
            cosine_sim = 0.0

        # Euclidean Distance (L2): Range [0, inf). Smaller is more similar.
        euclidean_dist = np.linalg.norm(vec_a - vec_b)
        

        # --- Display results ---
        self.result_textbox.configure(state="normal")
        self.result_textbox.delete("1.0", "end")
        
        header = f"Podobieństwo między '{word_a_str}' a '{word_b_str}':\n\n"
        self.result_textbox.insert("end", header)
        
        self.result_textbox.insert("end", f"  - Podobieństwo kosinusowe: {cosine_sim:.4f}\n")
        self.result_textbox.insert("end", "    (Zakres: -1 do 1. Im bliżej 1, tym bardziej podobne)\n\n")
        
        self.result_textbox.insert("end", f"  - Dystans Euklidesowy (L2): {euclidean_dist:.4f}\n")
        self.result_textbox.insert("end", "    (Im mniejsza wartość, tym bardziej podobne)\n\n")
        
        self.result_textbox.configure(state="disabled")

# Main
if True:
    model_file = Path(f"{MODEL_PREFIX}.model")
    embeddings_file = Path(f"{EMBEDDINGS_FILENAME}.npy")
    sp_processor = None
    embeddings_matrix = None
    error_messages = []

    if not model_file.exists():
        error_messages.append(f"Błąd: Nie znaleziono pliku modelu SentencePiece: {model_file}")
    else:
        try:
            sp_processor = spm.SentencePieceProcessor()
            sp_processor.load(str(model_file))
        except Exception as e:
            error_messages.append(f"Błąd ładowania modelu SentencePiece: {e}")
            sp_processor = None
    
    if not embeddings_file.exists():
        error_messages.append(f"Błąd: Nie znaleziono pliku osadzeń: {embeddings_file}")
    else:
        try:
            embeddings_matrix = np.load(embeddings_file)
        except Exception as e:
            error_messages.append(f"Błąd ładowania pliku osadzeń: {e}")
            embeddings_matrix = None
            
    if sp_processor and embeddings_matrix is not None:
        if sp_processor.get_piece_size() > embeddings_matrix.shape[0]:
            error_messages.append(f"Ostrzeżenie: Rozmiar słownika ({sp_processor.get_piece_size()}) > wierszy osadzeń ({embeddings_matrix.shape[0]}).")
        elif sp_processor.get_piece_size() < embeddings_matrix.shape[0]:
             error_messages.append(f"Ostrzeżenie: Rozmiar słownika ({sp_processor.get_piece_size()}) < wierszy osadzeń ({embeddings_matrix.shape[0]}).")

    if sp_processor is None or embeddings_matrix is None:
        root_error = ctk.CTk()
        root_error.title("Błąd Krytyczny Modelu")
        root_error.geometry("700x300")
        label_error = ctk.CTkLabel(root_error, text="Nie udało się zainicjalizować aplikacji z powodu błędów modelu:\n\n" + "\n".join(error_messages),
                                   wraplength=680, justify="left", text_color="red")
        label_error.pack(pady=20, padx=20, fill="both", expand=True)
        root_error.mainloop()
        sys.exit(1)

    try:
        embed_ops = EmbeddingOperations(sp_processor, embeddings_matrix)
        app = WordArithmeticApp(embed_ops)
        app.mainloop()
    except Exception as e:
        root_error = ctk.CTk()
        root_error.title("Błąd Krytyczny Aplikacji")
        root_error.geometry("600x250")
        label_error = ctk.CTkLabel(root_error, text=f"Nie udało się zainicjalizować EmbeddingOperations lub aplikacji:\n\n{e}",
                                   wraplength=580, justify="left", text_color="red")
        label_error.pack(pady=20, padx=20, fill="both", expand=True)
        root_error.mainloop()
        sys.exit(1)
