# .env 방법
##########
"""
1. 루트에 .env 파일을 만든다    
2. 파일 안에 키를 넣음
 # .env 파일 내용
   OPENAI_API_KEY = ""
3. git 에 안올라가게 .gitignore 안에 .env 널기 
 # .gitignore내용
   .env

"""
##########

import langchain
import openai
print(langchain.__version__)        # 0.3.7
print(openai.__version__)           # 1.54.3


# import os 
# os.environ['OPENAI_API_KEY'] = openai_api_key

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model_name = 'gpt-3.5-turbo',
                 temperature=0,
                 # openai_api_key = openai_api_key,
                 # api_key = openai_api_key,
                 )

aaa = llm.invoke('비트캠프 윤영선에 대해 알려줘').content
print(aaa)


