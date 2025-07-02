import re
import pandas as pd

def parse_chat_log(chat_log):
    pattern = r"\[(\d{1,2}/\d{1,2}/\d{2}), (\d{1,2}:\d{2}:\d{2})\u202f([AP]M)\] ?(.+?)? ?: ?(.+?)(?=\[|$)"
    matches = re.findall(pattern, chat_log, flags=re.DOTALL)
    
    parsed_log = []
    for match in matches:
        date_time_str = f"{match[0]} {match[1]} {match[2]}"
        date_time = pd.to_datetime(date_time_str, format='%d/%m/%y %I:%M:%S %p')
        message = match[4].strip().lower()
        
        if "image omitted" in message:
            message_type = "Image"
            message = ""
        elif "video omitted" in message:
            message_type = "Video"
            message = ""
        elif "sticker omitted" in message:
            message_type = "Sticker"
            message = ""
        elif "audio omitted" in message:
            message_type = "Audio"
            message = ""
        elif "document omitted" in message:
            message_type = "Document"
            message = ""
        elif "contact omitted" in message:
            message_type = "Contact"
            message = ""
        elif "location shared" in message:
            message_type = "Location"
            message = ""
        elif "missed voice call, tap to call back" in message:
            message_type = "Voice Call"
            message = "Missed"
        elif "voice call," in message:
            match_duration = re.search(r"(\d+:\d+)", message)
            if match_duration:
                message_type = "Voice Call"
                message = match_duration.group(0)
            else:
                message_type = "Voice Call"
                message = message.split(',')[-1]
        elif "missed video call, tap to call back" in message:
            message_type = "Video Call"
            message = "Missed"
        elif "video call," in message:
            match_duration = re.search(r"(\d+:\d+)", message)
            if match_duration:
                message_type = "Video Call"
                message = match_duration.group(0)
            else:
                message_type = "Video Call"
                message = message.split(',')[-1]
        else:
            message_type = "Text"
        
        parsed_log.append({
            "date_time": date_time,
            "user": match[3],
            "message_type": message_type,
            "message": message
        })
    df = pd.DataFrame(parsed_log)
    df.to_csv('chat_log.csv', index=False)
    return df