"""Automatic speech recognition"""
import whisper
import torch

def transcribe(file_path: str, model_name="base") -> str:
    """
    Transcribe input audio file.

    Examples
    --------
    >>> text = transcribe(".../audio.mp3")
    >>> print(text)
    'This text explains...'
    """
    # CUDA device (if available)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load the ASR model and trascribe the text
    model = whisper.load_model(model_name, device=device)
    result = model.transcribe(audio=file_path)
    return result['text']
