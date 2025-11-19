import re
from collections import Counter
import sys

def analyze_text_file(file_path: str, top_n: int = 25):
    """
    Analyzes a large text file without loading it all into memory.
    
    Args:
        file_path (str): The path to the text file.
        top_n (int): The number of most common words to display.
    """
    print(f"🔥 Starting analysis of '{file_path}'...")

    word_counts = Counter()
    total_word_count = 0

    try:
        # The 'with open' statement streams the file line by line
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            for line in file:
                # 1. Make the line lowercase to treat 'The' and 'the' as the same
                line = line.lower()
                
                # 2. Use regex to find all word-like sequences.
                # \b matches word boundaries, \w+ matches one or more letters/numbers.
                # This is great for handling multiple languages.
                words = re.findall(r'\b\w+\b', line)
                
                # 3. Update our counts
                if words:
                    word_counts.update(words)
                    total_word_count += len(words)

    except FileNotFoundError:
        print(f"💀 ERROR: The file was not found at '{file_path}'")
        print("Please check the path and try again.")
        sys.exit(1) # Exit the script if the file doesn't exist
    except Exception as e:
        print(f"😬 An unexpected error occurred: {e}")
        sys.exit(1)

    # --- Print the results ---
    
    print("\n✅ Analysis complete! Here's the breakdown:")
    print("-" * 40)
    
    # Total number of words
    print(f"📊 Total Words: {total_word_count:,}")
    
    # Number of unique words
    unique_word_count = len(word_counts)
    print(f"💎 Unique Words: {unique_word_count:,}")
    
    # The Top N most common words
    print(f"\n🏆 Top {top_n} Most Common Words:")
    print("-" * 40)
    for word, count in word_counts.most_common(top_n):
        print(f"{word:<20} | {count:>7,}") # Formatted for nice alignment

# --- Main execution block ---
if __name__ == "__main__":
    # <-- ❗️ IMPORTANT: Change this to the actual path of your file!
    # On Linux, your path might look like '/home/grinnazar/documents/my_big_book.txt'
    # Or if it's in the same folder, just 'my_big_book.txt'
    path_to_your_file = 'AllDatasets.txt' 
    
    analyze_text_file(file_path=path_to_your_file)
