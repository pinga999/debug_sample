`basic debugging`

![Alt text](DEBUG_img.png)
- 좌 $\Rightarrow$ 우 순서대로
    - `계속` : 그 다음 브레이크 포인트로 바로 실행
    - `단위 실행` : 코드를 한 줄 씩 실행, 다른 함수를 호출할 때 그냥 무시하고 건너뛰면서 확인이 된다.
    - `단계 정보` : 내가 정의한 함수 안으로 들어가 디버깅 할 수 있다.
    - `단계 출력` : 현재 디버깅 하고 있는 함수를 바로 끝내고 밖으로 나올 수 있다.
    - `다시 시작` : 진행중인 디버깅을 멈추고 새로 디버깅을 시작한다.
    - `중지` : 진행중인 디버깅을 종료한다.

`advanced? debugging`
- launch.json을 활용한 debugging!
    - `"args" : ["--config","test.yaml","--input","check.txt"]`를 추가해서 args를 파싱한다.
    
- .sh을 이용하거나... 등등 가끔 복잡한 사연을 가진일이있어서 python파일을 직접 돌릴수 없는경우의 디버깅
    - 필요한 곳 마다 print를 다 꽂아본다.
    - 어떻게든 python으로 돌리게 해본다.
        - multiprocessing popen 등등...
    - `breakpoint()`를 활용한다.(기본적으로 빨간콩을 찍는 대신 의심스러운곳에 `breakpoint()`를 미리 추가하는 방식)
        - help : 도움말
        - next : 현 위치 기준으로다음 문장으로 이동
        - print : 변수값을 standard output에 표시 (우리가 흔히 아는 print({something})와 기능이 같다)
        - list : 소스코드를 쭉 출력함, 현 위치를 화살표로 표시해줌
        - where : call스택 출력 (어느 함수/코드/스레드 등등에서 지금 이걸 실행하고있는지 표시)
        - continue : 다음 중단점까지 실행, 다음 중단점이 없다면 끝까지 실행
        - step : 함수 내부로 들어감, 위의 단계 정보와 동일
        - return : 현재함수의 리턴 직전까지 실행
        - !{var_name} = value : var_name이라는 변수에 value를 재할당
        - quit : 디버깅을 종료한다

- 디바이스에 토치가 안깔려있다면!
[링크](https://pytorch.kr/get-started/locally/?_gl=1*1fb5qd1*_ga*MTg0MDIwMTYzNC4xNjk2NjEyNzc5*_ga_LZRD6GXDLF*MTcwOTE0MTA0MC45LjEuMTcwOTE0MjU5Ni40MC4wLjA.)를 따라서 pytorch를 깔아 주세요!