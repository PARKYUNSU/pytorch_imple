import os 
import pandas as pd 
import numpy as np 
from numpy import dot # dot product : 내적 (행렬 곱)
from numpy.linalg import norm # norm : 단위를 1(unit) 통일 
import ast 
import openai
import streamlit as st
from streamlit_chat import message 
from tqdm import tqdm 

tqdm.pandas()

client = openai.OpenAI(
  api_key = "API_KEY"
)

# embedding(문자 >> 숫자)
def get_embedding(text): 
    response = client.embeddings.create(
    input = 'text', 
    model = 'text-embedding-ada-002'  
    
  )
    return response.data[0].embedding
  
# folder_path 와 file_name 결합 >> file_path = './data/embedding.csv' 
folder_path = './data'
file_name = 'embedding.csv'
file_path = os.path.join(folder_path, file_name)

# if: embedding.csv 가 이미 존재한다면 df (데이터프레임) 로드 
if os.path.isfile(file_path): 
  print(f'{file_name} 파일이 존재합니다')
  df = pd.read_csv(file_path)
  df['embedding'] = df['embedding'].progress_apply(ast.literal_eval)
  # ast.literal_eval : 문자열 >> 파이썬 리스트 변환 
  
# 그렇지 않다면 text 열과 embedding 열이 존재하는 df 생성해야 함 
else: 
  df = pd.read_csv('./data/ChatBotData.csv')
  # 데이터프레임 Q 열로부터 embedding 열 생성 
  df['embedding'] = df.progress_apply(lambda row : get_embedding(row.Q), axis=1)
  df.to_csv(file_path, index=False, encoding='utf-8')
  
# 주어진 질의로부터 유사 문서 반환하는 검색시스템 
# 함수 return_answer_candidate 내부에서 유사도 계산을 위해 cos_sim 호출 

def cos_sim(A, B): 
  return dot(A,B) /(norm(A) * norm(B))

def retrun_similar_answer(input): 
  embedding = get_embedding(input)
  df['score'] = df.progress_apply(lambda x: cos_sim(x['embedding'], embedding), axis=1)
  return df.loc[df['score'].idxmax()]['A']
  # df의 score 열 에서 최대값 갖는 행의 index 
  # 해당 행의 A(answer)열의 값을 선택  

st.title('당신의 대화 상대가 되어 드릴께요!')
st.image('images/ask_me_chatbot_logo.png', width=300)

# 화면에 보여주기 위해 챗봇 답변 저장 공간 필요 (할당)
if 'generated' not in st.session_state:
  st.session_state['generated'] = []

# 화면에 보여주기 위해 사용자 답변 저장 공간 필요(할당)
if 'past' not in st.session_state:
  st.session_state['past'] = []
  
# 사용자 입력 들어오면 >> user_input에 저장 
# send button click 하면 >> submitted 값이 True 변환 

with st.form('form', clear_on_submit=True): 
  user_input = st.text_input('대화를 시작해 보세요!', '', key='input')
  submitted = st.form_submit_button('SEND') 
  
# submitted 의 값이 True >> 챗봇이 답변 
if submitted and user_input: 
  # 사용자 입력 바탕으로 챗봇의 답변을 얻음 
  chatbot_response = retrun_similar_answer(user_input)
  # 화면에 보여주기 위해 사용자 질문과 챗봇 답변 각각 저장
  st.session_state['past'].append(user_input)
  st.session_state['generated'].append(chatbot_response)
  
# 사용자 질문과 챗봇 답변을 순차적으로 화면에 출력 
if st.session_state['generated']: 
  for i in reversed(range(len(st.session_state['generated']))): 
    # 저장된 메시지의 수만큼 반복 
    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
    message(st.session_state['generated'][i], key=str(i))
  






