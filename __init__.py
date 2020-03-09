# __init__.py 파일의 내용
 
__all__ = ['bill']

# 패지키내 모든 모듈 import
# from 패키지명 import *
from models.account import *
bill.charge(1, 50)