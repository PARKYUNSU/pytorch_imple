import streamlit as st 
import io 
import base64
from openai import OpenAI
from PIL import Image 

client = OpenAI(
    api_key = "API_KEY" # 단, Key는 여러분들의 Key 값을 넣으세요.
)

def get_image(prompt): 
  response = get_image_info(prompt)
  # dall-e 로부터 Base64 형태 이미지 획득 
  image_data = base64.b64decode(response)
  # base64로 쓰여진 데이터를 이미지 형태로 변환
  image = Image.open(io.BytesIO(image_data))
  # 파일처럼 만들어진 이미지 데이터를 컴퓨터에 볼 수 있도록 open 
  return image

# dall-e 가 이미지를 반환하는 함수 
def get_image_info(prompt):
  response = client.images.generate(
              model = 'dall-e-3', 
              prompt=prompt, 
              # 사용자 프롬프트 
              size = '1024x1024', 
              quality = 'standard',
              response_format='b64_json', 
              # base64형태의 이미지 전달 
              n=1   
            )
  return response.data[0].b64_json

st.title("그림 그리는 AI 화가 서비스")

st.image('https://wikidocs.net/images/page/215361/%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5%ED%99%94%EA%B0%80.png', width=200)

st.text("Tell me the picture you want")

input_text = st.text_area('원하는 이미지 설명을 영어로 적어보세요.', height=200)

# paining 버튼 클릭 시 >> True 
if st.button('painting'): 
  # 이미지 프롬프트 작성된 경우 >> True 
  if input_text: 
    try: 
      # 사용자 입력으로부터 이미지 전달 받을게요 
      dalle_image = get_image(input_text) 
      
      # st.image() >> 이미지 시각화 
      st.image(dalle_image)
    
    except: 
      st.error('요청 오류가 발생했습니다.')
  else: 
    # 이미지 프롬프트가 작성안되었는데 버튼 누른 경우
    st.warning('제발 텍스트 좀 입력해 주세요.')