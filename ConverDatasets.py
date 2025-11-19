import os

def combine_txt_files(root_folder, output_filename="big.txt"):
    if not os.path.isdir(root_folder):
        print(f"Error: Folder '{root_folder}' not found.")
        return

    output_filepath = os.path.join(root_folder, output_filename)
    files_processed_count = 0
    files_skipped_count = 0

    print(f"Starting to combine .txt files into '{output_filepath}'...")

    try:
        with open(output_filepath, 'w', encoding='utf-8') as outfile:
            # Walk through the directory tree
            for dirpath, dirnames, filenames in os.walk(root_folder):
                for filename in filenames:
                    if filename.lower().endswith('.txt'):
                        file_path = os.path.join(dirpath, filename)

                        # Ensure we don't try to read the output file itself
                        if os.path.abspath(file_path) == os.path.abspath(output_filepath):
                            print(f"Skipping the output file itself: '{file_path}'")
                            continue

                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as infile:
                                content = infile.read()
                                outfile.write(content)
                                # Add a newline between contents of different files for better separation
                                # You can remove this if you want a strict concatenation without any separators.
                                outfile.write('\n') 
                                print(f"Successfully processed and added: '{file_path}'")
                                files_processed_count += 1
                        except Exception as e:
                            print(f"Error reading file '{file_path}': {e}")
                            files_skipped_count += 1
        
        print(f"\nFinished!")
        print(f"Total .txt files processed and combined: {files_processed_count}")
        if files_skipped_count > 0:
            print(f"Total files skipped due to errors: {files_skipped_count}")
        print(f"Combined content saved to: '{output_filepath}'")

    except IOError as e:
        print(f"Error writing to output file '{output_filepath}': {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


folder_to_scan = "AllDatasets"
    
output_file_name = "AllDatasets.txt" 
    
combine_txt_files(folder_to_scan, output_file_name)

