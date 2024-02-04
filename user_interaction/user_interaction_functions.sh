#!/bin/bash

# Function to automate Slack interactions
openGeneralChatFromHomeScreen () {
    local channel="$1"
    formatted_channel=${channel// /%s}  # Replaces spaces with %s
    formatted_channel=${formatted_channel//\'/\\\'}

    echo "Starting from home screen...";
    adb shell input keyevent KEYCODE_HOME ;

    sleep 2;

    echo "Opening Slack...";
    adb shell am start -n com.Slack/slack.features.home.HomeActivity
    
    sleep 2;

    echo "Opening DMs channel...";
    adb shell input tap 330 2180;

    sleep 2;

    echo "Opening DMs search...";
    adb shell input tap 572 403;

    sleep 2;

    echo "Adding DM search input...";
    adb shell input text "'$formatted_channel'"


    sleep 2;

    echo "Opening general channel...";
    adb shell input tap 260 423;
}

postMessageToTheChat() {

    KEYBOARD_STATUS=$(adb shell dumpsys input_method | grep "mInputShown" | awk '{print $NF}')

    if [ "$KEYBOARD_STATUS" = "true" ]; then
        adb shell input keyevent KEYCODE_BACK
    else
        echo "Keyboard is not shown."
    fi

    local message="${1// /%s}"  # Replaces all spaces with '%s'
    formatted_channel="${message//\'/\'\\\'\'}"

    sleep 1;

    # Clicks on the text box
    adb shell input tap 550 2200

    sleep 1;
    # Writes your text
    adb shell input text "'$formatted_channel'"

    sleep 1;
    # Clicks on send button
    adb shell input tap 1000 1450

    sleep 1;
}

returnToHomeScreen() {
    sleep 1;
    # Returns to home screen
    adb shell input keyevent KEYCODE_HOME
    sleep 1;
}