def copy_first_x_lines(input_file, output_file, x_lines):
    """
    Reads the first X lines from an input file and appends them to an output file.
    """
    # Make sure X is a valid number
    try:
        num_lines = int(x_lines)
        if num_lines < 0:
            print("Can't copy a negative number of lines, fam. Setting to 0.")
            num_lines = 0
    except ValueError:
        print(f"'{x_lines}' isn't a number. Defaulting to 10 lines.")
        num_lines = 10

    try:
        # Open the source file to read
        with open(input_file, 'r', encoding='utf-8') as in_f:
            # Open the destination file to append
            # 'a' mode creates the file or appends to it if it exists
            with open(output_file, 'a', encoding='utf-8') as out_f:
                
                if num_lines == 0:
                    print("You asked for 0 lines. No lines appended. Done.")
                    return

                print(f"Appending first {num_lines} lines from '{input_file}' to '{output_file}'...")
                
                # Read line by line
                for i in range(num_lines):
                    line = in_f.readline()
                    
                    # If readline() returns an empty string, we hit the end of the file
                    if not line:
                        print(f"FYI: Hit the end of '{input_file}' after {i} lines.")
                        break # Stop looping if EOF
                    
                    # Write the line to the new file
                    out_f.write(line)
        
        print(f"GG. Appended (up to) {num_lines} lines.")

    except FileNotFoundError:
        print(f"Bruh, can't find this file: '{input_file}'")
    except IOError as e:
        print(f"Big yikes. Had an I/O error: {e}")
    except Exception as e:
        print(f"Something random went wrong: {e}")

def main():
    """
    Main function to run the script with parameters set in the file.
    """
    
    # --- SET YOUR PARAMETERS HERE ---
    input_name = "polish_pd_complete.txt"
    output_name = "FirstMillionBD.txt"
    num_lines_str = "1000000"  # Make sure this is a string
    # --- END OF PARAMETERS ---

    print(f"Running script with hardcoded params:")
    print(f"  Input file: {input_name}")
    print(f"  Output file: {output_name}")
    print(f"  Lines to append: {num_lines_str}\n")
    
    copy_first_x_lines(input_name, output_name, num_lines_str)


main()

