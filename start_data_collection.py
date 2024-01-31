import pandas as pd
import random
import time

from android_pcap_data_collection_and_analysis.user_interaction.slack_control import open_general_chat_from_home_screen, post_message_to_the_chat, return_to_home_screen


def grab_hamlet_from_excel():
    excel_file_path = 'android_pcap_data_collection_and_analysis/create_dialog/play_dialogue_hamlet.xlsx'  # Change this to the path of your Excel file
    df = pd.read_excel(excel_file_path)
    return df

def main():
    # Define the channel name
    CHANNEL_GENERAL = "general"
    CHANNEL_RANDOM = "random"

    # Define the user interaction functions
    hamlet_lines = grab_hamlet_from_excel()

    # Initialize the previous sequence flag; assume False at start
    prev_is_text_sequence = False

    for index, row in hamlet_lines.iterrows():

        which_channel_to_post = random.randint(1, 10)
        random_wait_intervals = random.randint(1, 60)
        is_text_sequence = random.randint(1, 10) % 2 == 0

        if prev_is_text_sequence:
            return_to_home_screen()
            time.sleep(random_wait_intervals)

        if (is_text_sequence or index == 0):
            if which_channel_to_post % 2 == 0:
                stdout, stderr = open_general_chat_from_home_screen(CHANNEL_RANDOM)
            else:
                stdout, stderr = open_general_chat_from_home_screen(CHANNEL_GENERAL)

        character, dialogue = row
        print(f"Posting '{dialogue}' as '{character}'...")

        stdout, stderr = post_message_to_the_chat(f"{character}: {dialogue}")

        prev_is_text_sequence = is_text_sequence
    return 

if __name__ == "__main__":
    main()