# PX4-ROS2 자율비행 드론 프로젝트

**ROS 2**와 **PX4 Autopilot**(및 ArduPilot), **Gazebo 시뮬레이터**를 활용한 자율비행 드론 연구 및 시뮬레이션 레포지토리입니다.  
지상 관측 시스템과 드론 온보드 비전 간의 다중 폐루프(Closed-Loop) 협력 유도 제어, 3차원 표적 좌표 추정(UKF), 쿼드 테일시터(Quad-Tailsitter) 기동 제어, 정밀 착륙 등을 다룹니다.

---

## 팀원 (Team Members)

- **왕세영 (Seyeong Wang)** — [@SeyeongW](https://github.com/SeyeongW)
- **조태현 (Taehyeon Cho)** — [@mandu38](https://github.com/mandu38)
- **서영욱 (Yeonguk Seo)** — [@YeongukSeo](https://github.com/YeongukSeo)

---

## 주요 패키지 및 문서 안내

| 경로 | 설명 |
| :--- | :--- |
| **[`balloon_sim/`](balloon_sim/README.md)** | **지상 카메라 6-DOF UKF + 쿼드 테일시터(`SYS_AUTOSTART=1045`) 온보드 3D LOS/PN 비전 유도 시뮬레이션 패키지** (시뮬레이션 실행 방법 및 결과 수록) |
| **[`balloon_sim/ALGORITHMS.md`](balloon_sim/ALGORITHMS.md)** | **`balloon_sim`에 적용된 수학적 알고리즘 및 제어·추정 수식 상세 문서** (6-DOF UKF, 비주얼 서보잉, 테일시터 SO(3) 자세·추력 분리 제어, 3D 비례항법 유도 등) |
| **[`balloon_sim/HARDWARE_GUIDE.md`](balloon_sim/HARDWARE_GUIDE.md)** | **실기체(SIYI A8 mini + SiK 라디오 + 쿼드 테일시터 온보드) 전환 및 현장 운용 매뉴얼** |
| **[`balloon_sim/PX4_PARAMETERS.md`](balloon_sim/PX4_PARAMETERS.md)** | **PX4 쿼드 테일시터(`SYS_AUTOSTART=13003/1045`, `CA_AIRFRAME=4`) 전체 파라미터 표 및 QGC 설정 파일** |
| `flight/` · `camera/` · `simulation/` · `gcs/` | 드론 비행 제어, 카메라 비전 처리, 시뮬레이션 환경 및 지상 관제(GCS) 관련 모듈 |

---

## 빠른 시작 (시뮬레이션 실행)

현재 메인 시뮬레이션 패키지인 [`balloon_sim/`](balloon_sim/README.md) 실행 방법입니다:

```bash
cd balloon_sim

# 1) 지상 카메라 탐색/UKF + 테일시터 비전 유도 시뮬레이션 실행
./tools/run_tailsitter_sim.sh

# 2) 지상 카메라가 풍선을 락온하면, 카메라 뷰어 창에서 [SPACE] 또는 [ENTER] 키를 누르거나 아래 스크립트로 출격 승인
./tools/approve_launch.sh

# 3) 비행 후 PX4 ULog(.ulg) 텔레메트리 및 3D 궤적 그래프 생성
./tools/plot_tailsitter_log.py
```

자세한 실행 절차와 아키텍처는 **[`balloon_sim/README.md`](balloon_sim/README.md)** 및 **[`balloon_sim/ALGORITHMS.md`](balloon_sim/ALGORITHMS.md)**를 참고하세요.
