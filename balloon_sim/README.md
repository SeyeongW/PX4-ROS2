# 카메라 풍선 · STL/PACER 비행 워크스페이스

시뮬레이션·Qt UI·RViz·STL 기체·Bézier 경로 코드를 이 폴더에서 빌드한다. 총중량은 기체·배터리를 포함한 2 kg이며 카메라만 사용한다.

```bash
# Ubuntu 22.04 / ROS 2 Humble / Gazebo Classic 11 의존성 설치 후
./tools/build_px4.sh
./tools/build_workspace.sh
./tools/run_gazebo.sh
```

- [첫 화면 수식·시스템 소개](../README.md)
- [동역학 수식 상세](docs/dynamics.md)
- [설치·실행 명령 전체](docs/run.md)
- [검증 결과 및 제한](docs/validation.md)
- [출처와 라이선스](docs/sources.md)

`balloon_locator/`는 카메라와 현재 기체/미션, `support/`는 함께 배포하는 기존 UI·경로·Gazebo 지원 패키지다. `external/`, `build/`, `install/`은 직접 생성하는 의존성/빌드 폴더이며 Git에 포함하지 않는다. 실행 데이터는 기본적으로 임시 저장 후 종료 시 삭제한다.
