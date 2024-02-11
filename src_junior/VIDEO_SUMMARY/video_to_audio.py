"""Video-audio convertion"""
import os

from pytube import YouTube
from moviepy.editor import AudioFileClip

def video_title(youtube_url: str) -> str:
    """
    Retrieve the title of a YouTube video.

    Examples
    --------
    >>> title = video_title("https://www.youtube.com/watch?v=SampleVideoID")
    >>> print(title)
    'Sample Video Title'
    """
    # load a video and extract the title
    yt = YouTube(youtube_url)
    return yt.title

def download_audio(youtube_url: str, download_path: str) -> None:
    """
    Download the audio from a YouTube video.

    Examples
    --------
    >>> download_audio("https://www.youtube.com/watch?v=SampleVideoID", "path/to/save/audio.mp4")
    """
    # load a video and extract MP4 audio
    yt = YouTube(youtube_url)
    audio = yt.streams.filter(only_audio=True).first()
    # don't forget to split into the head and tail
    output_path, output_filename = os.path.split(download_path)
    audio.download(output_path, output_filename)

def convert_mp4_to_mp3(input_path: str, output_path: str) -> None:
    """
    Convert an audio file from mp4 format to mp3.

    Examples
    --------
    >>> convert_mp4_to_mp3("path/to/audio.mp4", "path/to/audio.mp3")
    """
    # load the MP4 file
    audio = AudioFileClip(input_path)
    # convert MP4 to MP3
    audio.write_audiofile(output_path, codec='mp3')
    audio.close()
    # remove the input MP4
    os.remove(input_path)


if __name__ == '__main__':
    youtube_url = 'https://www.youtube.com/watch?v=XLaMjwiRL4g'
    folder_path = 'src_junior/VIDEO_SUMMARY/'

    # Step 1: extract the video title
    title = video_title(youtube_url)
    print(title)
    # Step 2: download the audiotrack
    download_audio(youtube_url, folder_path)
    # Step 3: convert MP4 to MP3
    input_path = folder_path + title + '.mp4'
    output_path = folder_path + title + '.mp3'
    convert_mp4_to_mp3(input_path, output_path)
