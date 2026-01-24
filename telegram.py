import requests
from typing import Optional, Union


def send_telegram_message(bot_token: str, chat_id: str, message: str) -> Optional[dict]:
    """Send a message to a Telegram chat or channel using bot token.

    `chat_id` can be a channel username (like @yourchannel) or numeric chat id.
    Returns the Telegram API JSON response on success, or None on failure.
    """
    url = f'https://api.telegram.org/bot{bot_token}/sendMessage'
    data = {
        'chat_id': chat_id,
        'text': message,
        'parse_mode': 'HTML'
    }
    try:
        resp = requests.post(url, data=data, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def send_telegram_photo(
    bot_token: str,
    chat_id: str,
    photo: Union[str, bytes],
    caption: Optional[str] = None
) -> Optional[dict]:
    """Send a photo to a Telegram chat or channel using bot token.

    Args:
        bot_token: Telegram bot token
        chat_id: Chat ID or channel username (like @yourchannel)
        photo: Either a file path (str) or image bytes
        caption: Optional caption for the photo (supports HTML)

    Returns:
        Telegram API JSON response on success, or None on failure.
    """
    url = f'https://api.telegram.org/bot{bot_token}/sendPhoto'
    data = {
        'chat_id': chat_id,
    }
    if caption:
        data['caption'] = caption
        data['parse_mode'] = 'HTML'

    try:
        if isinstance(photo, str):
            # File path provided
            with open(photo, 'rb') as f:
                files = {'photo': f}
                resp = requests.post(url, data=data, files=files, timeout=30)
        else:
            # Bytes provided
            files = {'photo': ('chart.png', photo, 'image/png')}
            resp = requests.post(url, data=data, files=files, timeout=30)

        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def send_telegram_photo_group(
    bot_token: str,
    chat_id: str,
    photos: list,
    captions: Optional[list] = None
) -> Optional[dict]:
    """Send multiple photos as a media group to Telegram.

    Args:
        bot_token: Telegram bot token
        chat_id: Chat ID or channel username
        photos: List of file paths or bytes
        captions: Optional list of captions (only first caption is shown in group)

    Returns:
        Telegram API JSON response on success, or None on failure.
    """
    import json

    url = f'https://api.telegram.org/bot{bot_token}/sendMediaGroup'

    media = []
    files = {}

    for i, photo in enumerate(photos):
        attach_name = f'photo{i}'
        media_item = {
            'type': 'photo',
            'media': f'attach://{attach_name}'
        }
        if captions and i < len(captions) and captions[i]:
            media_item['caption'] = captions[i]
            media_item['parse_mode'] = 'HTML'

        media.append(media_item)

        if isinstance(photo, str):
            files[attach_name] = open(photo, 'rb')
        else:
            files[attach_name] = (f'{attach_name}.png', photo, 'image/png')

    data = {
        'chat_id': chat_id,
        'media': json.dumps(media)
    }

    try:
        resp = requests.post(url, data=data, files=files, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None
    finally:
        # Close any opened files
        for key, value in files.items():
            if hasattr(value, 'close'):
                value.close()


if __name__ == '__main__':
    print('telegram helper loaded')
