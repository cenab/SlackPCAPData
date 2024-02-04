import pandas as pd
import random
import time
import logging


from user_interaction.slack_control import open_general_chat_from_home_screen, post_message_to_the_chat, return_to_home_screen


def grab_hamlet_from_excel():
    excel_file_path = 'create_dialog/play_dialogue_hamlet.xlsx'  # Change this to the path of your Excel file
    df = pd.read_excel(excel_file_path)
    return df

def main():
    # Setup basic logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Define the channel names
    CHANNEL_GENERAL = "general"
    CHANNEL_RANDOM = "random"

    # Define the user interaction functions
    hamlet_lines = grab_hamlet_from_excel()

    # Initialize the previous sequence flag; assume False at start
    prev_is_text_sequence = False

    for index, (character, dialogue) in enumerate(hamlet_lines.itertuples(index=False), 1):
        try:
            # Decision making for channel posting and wait intervals
            which_channel_to_post = random.choice([CHANNEL_RANDOM, CHANNEL_GENERAL])
            random_wait_intervals = random.randint(1, 60)
            is_text_sequence = random.choice([True, False])
            
            # Open the chat in the designated channel for the first iteration or if it's a text sequence
            if is_text_sequence or index == 1:
                time.sleep(random_wait_intervals)
                stdout, stderr = open_general_chat_from_home_screen(which_channel_to_post)
                if stderr:
                    logging.error(f"Error opening channel {which_channel_to_post}: {stderr}")
                    continue  # Skip this iteration

            time.sleep(2)
            logging.info(f"Posting '{dialogue}' as '{character}'...")
            stdout, stderr = post_message_to_the_chat(f"{character}: {dialogue}")
            if stderr:
                logging.error(f"Error posting message: {stderr}")
                continue  # Skip this iteration

            

            # Update the flag for text sequence
            prev_is_text_sequence = is_text_sequence

        except Exception as e:
            logging.exception(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()