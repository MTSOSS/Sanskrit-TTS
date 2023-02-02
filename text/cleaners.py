def sanskrit_cleaners(text):
    text = text.replace('॥', '।').replace('ॐ', 'ओम्')
    if len(text)==0 or text[-1] != '।':
        text += ' ।'
    return text
