def reader(path):
    messages = ""
    try:
        with open(path, 'r',encoding="utf8") as file:
            for line in file:
                message = line.strip().replace('\u200e','')
                messages+=message
    except FileNotFoundError:
        print("The file does not exist")
    except Exception as e:
        print(f"An error occurred: {e}")
    return messages