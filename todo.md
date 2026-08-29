1. kp=30~300 사이에서 data generation.
   - 여러 사용자에 대해 데이터 수집
   - test env에서 사용자별 하이퍼파라미터 테스트. (mu, T, good 비율, 여러 사용자별 weight 분포(현재 fully bayesian이 정규분포의 하이퍼파라미터를 갖고 있다고 가정했으므로 우리도 그렇게 생성할 예정))
2. 이에 맞게 copl, fully bayesian rm 추정 (각 사용자별)
3. rm을 최대화하는 gain scheduling or 최적화.
   - CMA-ES를 통해 사용자가 가장 선호하는 p gain 찾기 (VMC 코드에서는 grid search)
   - RL을 통해 사용자가 선호하는 scheduling