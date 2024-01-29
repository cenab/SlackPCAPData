import subprocess

# Define your shell functions as multi-line strings
open_general_chat_from_home_screen = """
openGeneralChatFromHomeScreen () {
    local channel="$1"

    echo "Starting from home screen..."
    adb shell input keyevent KEYCODE_HOME

    echo "Waiting for home screen animation to complete..."
    sleep 1

    echo "Opening Slack..."
    adb shell input tap 160 1550

    echo "Waiting for Slack app to open..."
    sleep 2

    adb shell input tap 80 220
    sleep 2
    adb shell input tap 80 220
    sleep 2
    adb shell input tap 80 220
    sleep 2
    adb shell input tap 80 220

    sleep 1

    echo "Opening DMs channel..."
    adb shell input tap 330 2180

    sleep 1

    echo "Opening DMs search..."
    adb shell input tap 572 403

    sleep 1

    echo "Adding DM search input..."
    adb shell input text "$channel"

    sleep 1

    echo "Opening general channel..."
    adb shell input tap 260 423
}
"""

post_message_to_the_chat = """
postMessageToTheChat() {
    local message="${1// /%s}"  # Replaces all spaces with '%s'

    # Clicks on the text box
    adb shell input tap 550 2200

    # Writes your text
    adb shell input text "$message"

    # Clicks on send button
    adb shell input tap 1000 1450
}
"""

return_to_home_screen = """
returnToHomeScreen() {
    # Returns to home screen
    adb shell input keyevent KEYCODE_HOME
}
"""

# Function to run the shell script from Python
def run_shell_script(script, args=None):
    if args is not None:
        script += f'\n{script.split()[0]} "{args}"'
    process = subprocess.run(['bash', '-c', script], text=True, capture_output=True)
    return process.stdout, process.stderr

# Example usage
channel_name = "general"
message = "Hello from Python"

# Import and use the shell function
stdout, stderr = run_shell_script(open_general_chat_from_home_screen, channel_name)
print(stdout)

stdout, stderr = run_shell_script(post_message_to_the_chat, message)
print(stdout)

# stdout, stderr = run_shell_script(return_to_home_screen)
# print(stdout)
