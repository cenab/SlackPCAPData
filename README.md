+------------------+       
|     main()       |       
+------------------+       
         |                 
         v                 
+------------------+       +-------------------------+
| Initialize       |------>| Define CHANNEL_GENERAL  |
| logging          |       | and CHANNEL_RANDOM      |
+------------------+       +-------------------------+
         |                                 |
         v                                 v
+------------------+       +-------------------------------+
| Iterate through  |------>| Decide which channel to post  |
| Hamlet lines     |       | to (random choice)            |
+------------------+       +-------------------------------+
         |                                 |
         v                                 v
+--------------------------+       +-------------------------+
| Check if previous message|------>| If yes, call             |
| was part of a text       |       | return_to_home_screen()  |
| sequence                 |       +-------------------------+
+--------------------------+                   |
         |                                     v
         v                           +-----------------------+
+--------------------------+         | Wait for              |
| Decide if current message|<--------| random_wait_intervals |
| is part of a text sequence|        +-----------------------+
+--------------------------+                   |
         |                                     v
         v                           +-------------------------+
+-------------------------------+    | Open Slack channel      |
| If first message or part of a |<---| with open_general_chat_ |
| text sequence, open channel   |    | from_home_screen()     |
+-------------------------------+    +-------------------------+
         |                                     |
         v                                     v
+--------------------------+       +-------------------------+
| Post message with        |       | Update previous sequence|
| post_message_to_the_chat()|<------| flag                    |
+--------------------------+       +-------------------------+


adb shell netstat -lt
adb shell lsof | grep

t2 -r pcap_edited/sanitized_pcap/slack_edited.pcapng -w .

netstat
wireshark filtering and saving
tracewrangeler to override the linux cooking with ethernet
generate the flow files with t2