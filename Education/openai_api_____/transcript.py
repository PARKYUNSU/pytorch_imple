import streamlit as st 
import os 
import logging 
from pytube import YouTube
from openai import OpenAI 

client = OpenAI(
    api_key = "API_KEY" # 단, Key는 여러분들의 Key 값을 넣으세요.
)

# mp3 파일로부터 자막파일 (srt 형식으로 문자열 반환)
def get_transcribe(file_path): 
  audio_file = open(file_path, 'rb')
  transcript = client.audio.transcriptions.create(
                model = 'whisper-1', 
                response_format = 'srt', 
                file = audio_file
            )
  return transcript

# 주소를 입력받으면 유튜브 동영상(mp4)와 유튜브 동영상의 음성(mp3) 추출하는 함수
def get_video_and_audio(url): 
  if url != '': 
    yt = YouTube(url)

    # 오디오 스트림 선택 >> 다운로드(mp3)
    audio = yt.streams.filter(only_audio=True).first()
    # 유튜브 영상에서 오디오만 추출 
    audio_file = audio.download(output_path ='.')
    # './' 현재경로 >> 오디오 출력 현재경로로 설정 
    base, ext = os.path.splitext(audio_file)
    new_audio_file = base + '.mp3'
    os.rename(audio_file, new_audio_file)
    
    # 비디오 스트림 선택 >> 다운로드(mp4)
    # 유튜브 영상 중 mp4 형식의 비디오 스트림 선택 
    video = yt.streams.filter(file_extension='mp4').get_highest_resolution()   
    video_file = video.download(output_path='.')
    
    # 파일크기 로깅(logging: 로그에 파일 크기를 기록함)
    audio_file_stats = os.stat(new_audio_file) # stat(status)
    video_file_stats = os.stat(video_file)
    logging.info(f'audio file size: {audio_file_stats.st_size}')
    logging.info(f'video file size: {video_file_stats.st_size}')
    
    return new_audio_file, video_file 
  
st.title('It is Your YouTube Video and subtitle Downloader')

url = st.text_input('Enter the YOUTUBE url')

# Download 버튼 클릭 >> True
if st.button('Download'): 
  if url:
    try: 
      audio_file, video_file = get_video_and_audio(url)
      result=get_transcribe(audio_file)
      subtitle_file ='./subtitle.srt'
      with open(subtitle_file, 'w', encoding='utf-8') as file: 
        file.write(result)
       
      st.success('Downloaded Successfully')  
      # 정상적으로 다운 되면 메시지 출력 
      
      # 화면에 영상 파일 경로 출력 
      video_file_path = os.path.abspath(video_file)
      # abspath : 주어진 경로를 절대경로(absapath) 로 변환
      st.markdown(f'비디오 파일이 저장되었어요: {video_file_path}')
      st.video(video_file)
      
      #화면에 자막파일 경로 출력 
      subtitle_file_path = os.path.abspath(subtitle_file) 
      st.markdown(f'자막 파일이 저장되었어요: {subtitle_file_path}')
      
      # 자막 파일 읽어 들여서 화면에 자막 출력 
      with open(subtitle_file, 'r', encoding='utf-8') as file: 
        subtitles = file.read()
        st.info(subtitles)
        
    except Exception as e: 
      st.error(f'Error: {e}') 
          
      
    
      
       
      