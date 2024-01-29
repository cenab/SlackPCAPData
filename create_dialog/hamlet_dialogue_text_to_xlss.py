import pandas as pd

# Specify the path to your text file
text_file_path = 'hamlet_dialogue.txt'  # Change this to the path of your text file

# Read the text file
with open(text_file_path, 'r') as file:
    lines = file.readlines()

# Initialize an empty list to hold the parsed data
data = []

# Loop through each line and split into character and dialogue
for line in lines:
    line = line.strip()  # Remove leading/trailing whitespace
    if line:  # Skip empty lines
        parts = line.split('. ', 1)
        if len(parts) == 2:
            character, dialogue = parts
        else:
            # If not, it's a continuation of the previous dialogue
            character, dialogue = data[-1][0], line  # Reuse last character's name
        data.append([character, dialogue])

# Convert the list of lists into a DataFrame
df = pd.DataFrame(data, columns=['Character', 'Dialogue'])

# Specify the path for your output Excel file
excel_file_path = 'play_dialogue_hamlet.xlsx'

# Write the dataframe to an Excel file, without the index
df.to_excel(excel_file_path, index=False)

print(f"Text file has been converted to Excel and saved as '{excel_file_path}'")
