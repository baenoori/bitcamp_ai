import langchain
import openai
print(langchain.__version__)        # 0.3.7
print(openai.__version__)           # 1.54.3


from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model_name = 'gpt-3.5-turbo',
                 temperature=0,
                 # openai_api_key = openai_api_key,
                 api_key = openai_api_key,
                 )

aaa = llm.invoke('비트캠프 윤영선에 대해 알려줘').content
print(aaa)

# 비트캠프 윤영선은 한국의 프로그래머이자 IT 전문가로, 비트캠프라는 교육기관에서 강사로 활동하고 있습니다. 윤영선은 소프트웨어 개발 및 데이터 분석 분야에서 다양한 경험을
#  가지고 있으며, 학생들에게 프로그래밍 및 데이터 분석 기술을 가르치는 데 능숙합니다. 또한 윤영선은 열정적이고 친절한 성격으로 많은 학생들에게 사랑을 받고 있습니다. 



