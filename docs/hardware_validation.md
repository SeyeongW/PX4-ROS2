# 실기체 · 하드웨어 검증 기록

시뮬레이터가 아니라 **진짜 기체에서 확인된 것과 아직 확인되지 않은 것**만 모은
문서. SITL 결과는 여기 쓰지 않는다 — 그건
[`정밀착륙_진행보고서.md`](../정밀착륙_진행보고서.md) 와 각 패키지 README 소관이다.

문서를 이렇게 나누는 이유는 하나다. **SITL에서 통과한 값은 실기체로 넘어오지
않는다.** 마운트가 다르고, 카메라가 다르고, 프레임 지연이 다르고, 네트워크가
있다. 아래 표의 "검증" 칸은 전부 *실제 하드웨어에서* 라는 뜻이다.

- 작성: 2026-08-23 · 대상 트리: `main` @ `017584f`
- 관련: [`worklog_2026-07-06.md`](worklog_2026-07-06.md) — 그날의 상세 로그

---

## 0. 한눈에 보는 현재 상태

| 항목 | 상태 |
|---|---|
| MAVROS ↔ FC USB 링크 | ✅ 확인 |
| 실비행 자체 (ArduPilot 시절) | ✅ 있었음 — `flight_logs/*.BIN` 7개(2026-06-25), 2026-07-07 첫 실비행 후 플로우 재작성 |
| 이륙/호버/착륙 (`naive`, PX4) | ⚠️ 이 목적으로 작성됨. **PX4 경로의 비행 기록은 확인 안 됨** |
| 카메라 → 검출 → pose 체인 | ✅ 기체 위에서 동작 (58 fps 하드웨어 디코드) |
| 짐벌 nadir 조준 + 스윕 | ⚠️ `./run_px4 bench` 로 프롭 없이 리허설 가능. 실행 기록은 확인 안 됨 |
| 마운트 부호 매핑 | ✅ 벤치 확인 (`lat_swap=false / +1 / +1`) |
| 카메라 내부 파라미터 | ✅ 3회 캘리브레이션 (현재 `fx=718.6`) |
| PC ↔ 젯슨 VPN + DDS 영상/그래프 | ✅ 무손실 확인 (18.4 Hz, +21 ms) |
| **ArUco 정밀착륙 실비행** | ❌ 미수행 |
| **트레일러 순항 → 착륙 실비행** | ❌ 미수행 (현장에서 라디오 문제로 중단) |
| **MPC 착륙 실비행** | ❌ 미수행 |
| 짐벌 자세 기준(`stabilized`) | ⚠️ 추정, 미측정 |
| 짐벌 레버암 | ⚠️ `[0,0,0]` = 미측정 |
| tf2 버퍼 지연 | ⚠️ 우회했을 뿐, 미해결 |

오프라인 테스트는 전부 통과한다 (`234 passed`, 2026-08-24 확인).
**테스트 통과는 비행 검증이 아니다.**

---

## 1. 하드웨어 구성

| 구성 | 사양 | 연결 |
|---|---|---|
| 비행 컴퓨터 | Jetson **Orin Nano Super**, JetPack 6 / L4T R36.4.7, 6 core | 호스트 `sw-desktop`, 계정 `sw` |
| FC | PX4 (초기 이력은 ArduPilot) | `/dev/ttyACM0:57600` — USB CDC, baud는 무시되지만 MAVROS가 숫자를 요구 |
| 하방 카메라 | USB MJPEG, `/dev/video0`, 1280×720@30 (60까지 가능) | `gst_camera_node` (NVJPG 하드웨어 디코드) |
| 짐벌 | **SIYI A8 mini** 3축 | 기본 시리얼 `/dev/ttyTHS1@115200`, 대안 UDP `192.168.144.25:37260` |
| 트레일러 라디오 | SiK 900 MHz (B end) | `/dev/ttyUSB0@57600` |
| 마커 | ArUco **0.18 m** | 크기 오설정 = `tvec.z` 전체가 틀어짐 |
| 배터리 | 4S 기준 (`min_battery_v = 14.0`) | 6S면 값을 올릴 것 |

### 카메라는 세 번 바뀌었다 — 캘리브레이션 이력

| 시점 | 카메라 | fx / fy | cx / cy | 왜곡 | 상태 |
|---|---|---|---|---|---|
| ~2026-07-05 | 초기 캠 | 717 / — | — | — | **폐기** |
| 2026-07-06 | icSpring `32e6:9221` 광각 (HFOV≈96°) | 578.25 / 579.59 | 666.82 / 395.65 | `-0.168189` … | **폐기** |
| 2026-07-10 | 협각 렌즈 | **718.60585 / 720.53861** | **630.49040 / 367.32221** | `[-0.057745, 0.054843, 0.003289, 0.000924, 0]` | **현재** |

- 저장 위치: 레포 `camera/aruco_landing/config/down_camera.yaml`
- 체커보드: `--size 8x6 --square 0.025` (25 mm), `gen_checkerboard.py` 기본값
- **캘리브 해상도 = 런타임 해상도**여야 한다. 해상도를 바꾸면 fx/fy/cx/cy가 전부
  스케일되고 solvePnP 거리가 조용히 틀어진다.
- 캘리브 파일을 읽는 건 `aruco_pose_node` 뿐이다 (`__init__`에서 1회).
  → 값 수정 후 필요한 건 **노드 재시작**이지 `colcon build`가 아니다.
- 카메라를 갈면 **무조건 재캘리브레이션**. 위 표가 그 증거다.

---

## 2. 네트워크 — VPN(Tailscale) + DDS

현장에서 가장 많은 시간을 먹은 영역이라 따로 크게 쓴다.

### 2.1 왜 VPN인가

젯슨의 유선 LAN(`10.81.162.150`)이 불안정하다가 2026-07-06에 아예 죽었고, 현장
Wi-Fi는 매번 대역이 바뀐다. **Tailscale IP는 망이 바뀌어도 고정**이므로 SSH도
DDS도 그 주소 하나만 보면 된다.

| 머신 | Tailscale IP | 비고 |
|---|---|---|
| PC (지상국) | `100.110.148.7` | |
| **착륙 젯슨 (이 프로젝트)** | **`100.112.65.33`** | `sw@100.112.65.33` |
| 다른 젯슨 (Cyclone 대상) | `100.110.98.82` | 팀 공용 |
| `core-1` | `100.65.225.66` | **죽은 노드.** `~/.ssh/config` 의 `jetson` 항목이 아직 여기를 가리킨다 — 그 주석은 지금 반대 방향으로 틀렸다 |

설치:

```bash
curl -fsSL https://tailscale.com/install.sh | sh && sudo tailscale up
tailscale status        # 링크 확인은 항상 여기부터
```

### 2.2 왜 Discovery Server인가

**Tailscale은 WireGuard이고, WireGuard는 유니캐스트 전용이다.** FastRTPS 기본
디스커버리는 멀티캐스트라서 VPN 너머로는 아무것도 못 찾는다. 그래서 유니캐스트
디스커버리 서버를 쓴다.

- 젯슨이 서버를 돌린다: systemd `fastdds-discovery.service` (enabled, `Restart=always`, 부팅 자동)
- 래퍼: `/usr/local/bin/fastdds_discovery_server.sh` → `exec /usr/bin/fast-discovery-server -i 0 -p 11811` (`0.0.0.0:11811`)
- systemd 아래서는 `fastdds discovery` 파이썬 래퍼가 아니라 **`fast-discovery-server` 바이너리를 직접** 불러야 한다 (전자는 "tool not found")
- 젯슨 `.bashrc`: `ROS_DISCOVERY_SERVER=127.0.0.1:11811` (서버가 자기 자신)

DDS 구현 자체도 2026-07-06에 **CycloneDDS → `rmw_fastrtps_cpp`** 로 옮겼다.
머신마다 peer XML을 손으로 고쳐야 하는 Cyclone 정적 peer 부담이 이유다. 레포의
`config/cyclonedds_*.xml` 은 팀 다른 젯슨용으로 남겨뒀다.

### 2.3 젯슨 DDS 프로파일 — `config/jetson_dds_fastdds.sh`

두 가지가 빠져 있어서 PC에서 영상이 아예 안 나왔었다. 둘 중 하나만 없어도 죽는다.

**(a) `interfaceWhiteList` — loopback을 반드시 같이 연다**

```
127.0.0.1        젯슨 안에서 도는 비행 스택. tailscale과 무관하게 항상 산다.
100.112.65.33    PC가 영상/그래프를 보는 경로. 끊겨도 비행엔 영향 없다.
```

PC 스크립트처럼 tailscale IP 하나만 화이트리스트하면 **젯슨 내부의
카메라→검출기→미션 링크까지 tailscale에 묶인다. 기체가 떠 있는 상태에서 VPN이
끊기면 비행 스택이 같이 죽는다.** 이건 느린 문제가 아니라 위험한 문제다.
빠지는 것이 핵심이기도 하다 — Wi-Fi LAN IP와 `docker0`가 광고되면 PC가 닿지도
못하는 주소로 연결을 시도한다.

**(b) `maxMessageSize = 1200`**

Tailscale MTU가 1280인데 FastDDS 기본 데이터그램은 65500 B다. 압축 프레임 하나가
IP 단에서 ~50조각으로 쪼개지고, 그중 하나만 유실돼도 프레임 전체가 버려진다.
RTPS 레벨에서 미리 1200으로 자르면 조각 하나 = 패킷 하나가 되어 재전송이 먹는다.

그 외: `SUPER_CLIENT`(그래프 전체 수신), `ROS_DOMAIN_ID=0` (**PC와 동일해야 함**).

설치·사용:

```bash
scp config/jetson_dds_fastdds.sh sw@100.112.65.33:~/dds_fastdds.sh
# 젯슨 ~/.bashrc 끝(ROS setup.bash 이후)에서:  source ~/dds_fastdds.sh
```

### 2.4 PC 쪽

| 스크립트 | 용도 |
|---|---|
| `~/dds_fastdds.sh` | 이 젯슨 (FastRTPS + DS `100.112.65.33:11811` + super_client + 자기 tailscale IP 자동감지 화이트리스트) |
| `~/dds_cyclone.sh` | 다른 젯슨 `100.110.98.82` / 팀 (CycloneDDS) |
| `~/dds_local.sh` | 로컬 멀티캐스트, 원격 설정 전부 해제 |

`.bashrc` 에 `jetson`(= `source ~/dds_fastdds.sh`) 과 `sitl`(멀티캐스트,
`ROS_DOMAIN_ID=1`) 셸 함수가 있고, **새 터미널 기본값은 SITL**이다.

> **PC에서 `dds_fastdds.sh` 를 자동 source 하지 말 것.** 디스커버리 서버 모드에는
> 멀티캐스트 폴백이 없어서, 젯슨이 꺼져 있으면 PC 로컬 SITL 노드끼리도 서로를
> 못 찾는다.

### 2.5 실측

| 항목 | 값 |
|---|---|
| `/down_camera/image/compressed` (PC에서) | **18.4 Hz / 783 KB/s** = 젯슨 로컬 레이트와 동일 → **손실 0** |
| 링크가 더하는 지연 | **약 21 ms** |
| tailscale 인터페이스 강제 상태 | 젯슨 `/chatter` PC 수신 확인 → 데이터가 실제로 VPN을 탄다 |

### 2.6 이 영역에서 시간을 태운 함정들

| 증상 | 진짜 원인 |
|---|---|
| PC에서 토픽이 2개만 보임 | **환경 불일치.** 비대화형 셸(= `.bashrc` 미실행 = 멀티캐스트)에서 노드를 띄우고 대화형 셸(= 디스커버리 서버)에서 조회했다. 게다가 `ros2 daemon`이 최초 환경을 캐시한다. **띄우는 셸과 조회하는 셸의 env를 같게 하고, 바꿀 때마다 `ros2 daemon stop`.** 리셋 직후 첫 `ros2 topic list`는 원래 ~2개다 — 한 번 더 실행할 것 |
| 서로 안 보임 | `ROS_DOMAIN_ID` 불일치 (PC 1 / 젯슨 0) |
| SSH로 `pkill -f "aruco_pose_node"` 했더니 조용히 아무 일도 안 남 | **원격 `bash -c` 명령줄 자체에 그 문자열이 들어 있어 자기 셸을 죽인다.** `[a]ruco_pose_node` 같은 브래킷 패턴을 쓰거나 PID를 먼저 잡을 것 |
| 로컬 노드끼리도 서로를 못 찾음 | 서버가 안 뜬 상태에서 셸에 `ROS_DISCOVERY_SERVER` 잔재. `unset ROS_DISCOVERY_SERVER ROS_SUPER_CLIENT` |
| 디스커버리 서버 재시작 실패 ("Server creation failed") | 실제 프로세스명은 `fast-discovery-server`(하이픈). `pkill -f 'fastdds discovery'` 는 못 잡고, 잔재가 11811을 물고 있다. `sudo pkill -9 -f fast-discovery-server` |
| 유령 `/fmu` 토픽 69개, 전부 0 Hz | PX4 SITL이 XRCE 세션을 안 닫고 죽어 `MicroXRCEAgent`가 퍼블리셔를 유지 중 (2026-07-28 정리) |
| SSH가 계속 끊김 (exit 255) | 젯슨 링크가 원래 불안정하다. 원격 명령은 짧게, 긴 건 `screen` |

---

## 3. 소프트웨어 스택 (실기체 경로)

```
gst_camera_node ──► aruco_pose_node ──► ┐
 (NVJPG 디코드)      (solvePnP)          │
siyi_gimbal_node ──► landing_tf_node ──►┤► aruco_landing_node / mpc_landing_node
 (A8 mini)           (map→base→광학)     │      (미션 상태기계)
trailer_gps_node ──► trailer_target_node┘              │
 (SiK 라디오)         (ENU 상대좌표)                    ▼
                                                    MAVROS ──► PX4
```

| 노드 | 패키지 | 역할 |
|---|---|---|
| `gst_camera_node` | `aruco_landing` | v4l2 → tee → {카메라 JPEG 그대로 압축토픽, `nvv4l2decoder`→BGR} |
| `aruco_pose_node` | `aruco_landing` | 캘리브 solvePnP(IPPE_SQUARE), 품질 게이트 → `/perception/down/marker_pose` |
| `landing_tf_node` | `aruco_landing` | `map→base_link→gimbal_mount→down_camera_optical_frame` 50 Hz |
| `siyi_gimbal_node` | `siyi_gimbal` | 시작 즉시 nadir, 자세 5 Hz 폴링, 미션 스윕 명령 수신 |
| `trailer_gps_node` | `trailer_link` | SiK 라디오 → `/trailer/fix` (`GLOBAL_POSITION_INT` 만) |
| `trailer_target_node` | `trailer_link` | 트레일러/기체 fix 차분 → `/trailer/target_local` (기체 로컬 ENU) |
| `aruco_landing_node` | `mpc_landing` | 비례 제어 착륙: PRECHECK→ARM→TAKEOFF→CRUISE→SEARCH→DESCEND→LAND |
| `mpc_landing_node` | `mpc_landing` | 동일 골격, MPC 하강 (SITL 검증된 `landing_mpc.mpc.LandingMPC` 그대로 import) |
| `naive_flight_node` | `mpc_landing` | MPC 대신 한 화면짜리 비례법으로 내리는 기준선 노드. `./run_px4 naive` 는 인식 스택 없이 띄우므로 이륙/호버/착륙만 돈다 |
| `radio_probe` | `trailer_link` | 라디오 진단 (아래 §6) |

### 실행 — 명령 하나

```bash
./run_px4              # ArUco 정밀착륙
./run_px4 trailer      # 트레일러 좌표로 순항 후 착륙
./run_px4 mpc          # MPC 착륙
./run_px4 naive        # MAVROS + naive 노드만 (인식 스택 미기동 = 이륙/착륙)
./run_px4 bench        # 프롭 뺀 지상 리허설: 짐벌 스윕 + 마커 픽스, FCU 무명령
```

| 환경변수 | 용도 |
|---|---|
| `FCU_URL=/dev/ttyACM0:57600` | FC 링크 변경 (TELEM UART면 `/dev/ttyTHS1:921600`) |
| `TRAILER_DEV=/dev/ttyUSB1` | 라디오 포트 |
| `TRAILER_LINK=0` | `trailer_gps_node` 를 내가 따로 돌릴 때 (**시리얼 포트 하나에 리더 하나**) |
| `GNSS_CHECK=on` | GNSS 프리체크 게이트 복원 |
| `GATES=all` | mpc 경로에서 3단 승인 복원 |

승인은 **비행당 1회(시동)** 다. `run_px4` 가 미션 노드를 포그라운드로 띄우므로
그 터미널에서 ENTER면 되고, 다른 터미널에서는:

```bash
ros2 run mpc_landing approve aruco_landing_node
ros2 run mpc_landing abort   aruco_landing_node
```

> **GNSS 프리체크는 기본 OFF다.** 패드에서 이 게이트가 막을 때 RC로 POSITION 모드
> 한 번 시동했다 끄면 PASS한다 — 기체 상태는 그대로인데 검사가 다른 순간을 봤을
> 뿐이다. 판정이 무관한 행동으로 뒤집히는 게이트는 안전장치가 아니라 **프리체크를
> 우회하는 습관을 가르치는 지연**이다. PX4 자신의 arm 거부가 권위 있는 판정이고
> 그건 그대로 살아 있다.

---

## 3.5 화면으로 보기 — 가상 장애물 회피

CJU 임무의 장애물은 **실제로 놓을 수 없어서 지도 데이터로만 존재**한다
(`drone_cju_route.yaml`, 0.45 × 0.35 m 기둥 25개, 높이 10 m). 그래서 회피가
실제로 일어났는지는 화면 없이는 확인할 방법이 없다.

미션 노드가 직접 발행한다 — 별도 뷰어 노드가 아니다. 장애물을 기체 로컬 ENU에
놓는 **site 원점이 런타임에 GPS 쌍에서 유도**되므로, 다른 노드가 원점을 다시
구하면 *그럴듯한데 실제로 난 것과 다른* 그림이 나온다.

| 토픽 | 타입 | 내용 |
|---|---|---|
| `/mission/obstacles` | `MarkerArray` (latched) | `obstacles` = 실제 크기, `clearance` = 플래너가 피하는 팽창 박스(반투명) |
| `/mission/route` | `Path` (latched) | 인증된 A*/SFC/B-spline 경로, 재계획마다 갱신 |
| `/mission/flown` | `Path` | 실제 지나온 자리 (0.25 m마다 표본, 1000점) |
| `/mission/clearance` | `Float32` | 지금 가장 가까운 장애물까지 여유 |
| `/mission/accel_scale` | `Float32` | 그래서 지금 쓰는 가속이 상한의 몇 배율인지 |

절대 토픽인 이유: `aruco_landing_node`든 `mpc_landing_node`든 화면은 같아야
하므로 RViz 설정 하나로 양쪽을 본다. latched라 **비행 중간에 RViz를 켜도** 장애물과
경로가 바로 뜬다.

**PC(지상국)에서 돌린다. 젯슨이 아니다.** 젯슨은 발행만 하고, 화면은 §2의 VPN/DDS
링크를 타고 PC에서 뜬다 — 그 링크가 존재하는 이유가 바로 이거다. 비행 컴퓨터에
3D 뷰어를 올리면 검출기가 쓸 코어를 빼앗는다(§4-2에서 이미 한 번 데인 곳).

```bash
jetson        # 젯슨을 보는 터미널로 전환 (도메인 0 + 디스커버리 서버)
./run_viz     # flight/mpc_landing/ — rviz2 -d rviz/cju_mission.rviz
```

PC에는 **이 저장소를 빌드할 필요가 없다.** 그리는 토픽이 전부 표준 메시지 타입이라
`.rviz` 파일 하나 말고는 레포에서 쓰는 게 없다.

기체 자체는 `landing_tf_node`의 `map → base_link` TF로 공짜로 뜬다.

**트레일러**는 미션 노드가 아니라 `trailer_target_node`가 이미 발행하는
`/trailer/target_local`(`PointStamped`, 프레임 `map`)을 그대로 그린다 —
트레일러 미션(`./run_px4 trailer`)일 때만, 그리고 **좌표가 완전히 유효할 때만**
뜬다. 그 노드는 낡거나 말이 안 되는 fix를 내보내는 대신 침묵하므로, **점이 사라지는
것이 에러 신호**지 화면 문제가 아니다.

> QoS 함정: 이 토픽은 센서 스타일 **BEST_EFFORT**다. RViz 디스플레이를 Reliable로
> 두면 호환이 안 돼 **아무것도 안 온다**(조용히). `.rviz`에 Best Effort로 박아뒀다.

> **DDS 환경은 스크립트가 안 건드린다.** 젯슨을 보려면 `jetson`(디스커버리 서버),
> SITL을 보려면 `sitl`(멀티캐스트)이고 디스커버리 서버 모드엔 멀티캐스트 폴백이
> 없다 — 스크립트가 한쪽을 고르면 다른 쪽엔 빈 화면이 뜬다. `run_viz`가 시작할 때
> 현재 RMW/도메인/서버를 한 줄 찍는 이유다.

⚠️ 이 지도로 볼 때 같이 보이는 사실: 기둥은 0.45 m, 차폐 마진 1.0 m인데 실기체
GPS 오차는 ±1–2 m다. **회피 기동이 위치 오차 안에 들어간다** — 화면에서 경로가
기둥을 피해 가더라도, 기체가 실제로 그만큼 정확히 그 자리에 있었다는 뜻은 아니다.

---

## 4. 검증 이력

### 2026-06-25 — ArduPilot 실비행
`flight_logs/00000001~7.BIN` (dataflash). 이 시기 스택은 ArduPilot GUIDED 속도
서보 방식이었다.

### 2026-07-06 — 하드웨어 스택 + 네트워크 정비
- `precland_hw_node`(ArduPilot GUIDED) + `aruco_pose_node` 신규
- 카메라 캘리브레이션 2회 (교체 때문)
- `usb_cam` 도입 — icSpring 720p는 **MJPG 전용**이고 `v4l2_camera`는 MJPG를 못 푼다
- DDS Cyclone→FastRTPS 전환, 디스커버리 서버 systemd 상시화

### 2026-07-07 — 첫 실비행 후 플로우 재작성
- 조종자 수동 인계 방식 → **시동 시 자동 GUIDED + 이륙(5 m) → 탐색 → 착륙 / 타임아웃 시 RTL** 로 변경
- 안전장치: `takeoff_latched`(시동 사이클당 자동이륙 1회), `guided_seen`(GUIDED 성립 후에만 모드전환 abort 허용), `max_land_attempts`(RTL 인터셉트 진동 차단)
- **마운트 부호 매핑 벤치 확정**: `lat_swap=false / lat_sign_fwd=+1 / lat_sign_left=+1`
  - 방법: **마커를 고정하고 드론을 움직인다.** 드론 전진 → `tvec.y` 증가, 드론 좌측 → `tvec.x` 증가
  - 마커를 움직이는 방식은 부호가 뒤집히므로, 이 규약을 반드시 명시할 것

### 2026-07-10 — 카메라 교체 + 재캘리브레이션
협각 렌즈, `fx=718.6`. 이 값이 현재 픽셀 예산 계산의 기준이다.

### 2026-07-28 — 세 개의 큰 발견

**(1) TF가 아예 없었다.** 검출은 성공하고 디버그 뷰에 사각형도 그려지는데
`_transform_outputs` 가 매 프레임 예외로 죽어 **pose가 0건 발행**됐다. 증상("마커를
못 본다")과 원인(TF 부재)이 전혀 닮지 않았다 — 이 형태를 기억해 둘 것.
→ `landing_tf_node` 추가.

**(2) 카메라가 스택 최대 지연원이었다.** 1280×720 MJPEG 실측:

| 경로 | 결과 |
|---|---|
| v4l2 → /dev/null | 58.3 fps |
| GStreamer + `nvv4l2decoder` | 58.2 fps |
| GStreamer + `jpegdec`(SW) | 29.8 fps |
| **`usb_cam` (mjpeg2rgb)** | **21 fps, 618 ms stale** |

usb_cam이 ARM 코어에서 21 fps로 푸는 동안 카메라는 58을 밀어넣으니 V4L2 큐가
영구히 차 있고 꺼내는 프레임마다 이미 낡았다. 오진 두 번을 거쳤다: 네트워크
문제로 보였지만 **젯슨 로컬 구독자도 618 ms**였고, 큐 백로그로 보였지만
**30/20/10 fps 전부 618 ms(±1)** 였다 — *레이트를 바꿔도 안 움직이는 지연은 큐가
아니다.* → `gst_camera_node`(NVJPG).

**(3) PC↔젯슨 영상 링크 복구** — §2.3 의 whitelist / maxMessageSize.

### 2026-07-29 — 하드웨어 디코드를 켰는데 2.1 fps였다

전날의 NVJPG 전환(`7f173b3`)이 끝났는데도 새 노드는 **2.1 fps**로 돌았다. 디코드는
이미 하드웨어에 있었으니 남은 곳은 발행 경로뿐이었고, 실측 결과 프레임 하나를
발행하는 데 **457 ms**가 걸리고 있었다.

| `msg.data` 에 넘긴 것 | 프레임당 발행 | 노드 처리율 |
|---|---|---|
| `bytes` | **457 ms** | **2.1 fps** |
| `array.array('B', ...)` | **4.7 ms** | **30 fps (카메라 전량)** |

원인은 rosidl이 `uint8[]` 필드에 생성하는 setter다. 값이 이미 `array.array` 면
그대로 받지만, 그 외에는 디버그 검증으로 떨어져

```python
all(isinstance(v, int) and 0 <= v < 256 for v in value)
```

를 **요소마다** 돈다. 1280×720 BGR 은 2,764,800 요소다.

- **`uint8[]` 필드에 `bytes` 를 넣지 말 것.** 압축 스트림도, 이 레포에서 `uint8[]`
  를 채우는 다른 어떤 코드도 같다.
- 이번에도 증상이 원인을 안 닮았다 — 화면으로는 §4-2의 "카메라가 느리다"와 구별이
  안 된다. 그때는 디코더였고 이번엔 직렬화였다. **카메라가 느려 보이면 디코드와
  발행을 따로 재야 한다.**

같은 날 통합: 런치 하나로 카메라+검출+짐벌+TF+미션, 착륙 엔드게임 정리(지면 게이트
disarm, 디글리치 acquire), 마커 기준 고도 측정.

### 2026-08-10~11 — PX4 경로 재구축
- `naive_flight_node`: **지루한 절반을 먼저 증명한다** — MAVROS 배선, PX4의 OFFBOARD 수락, 상승/유지/착륙. MAVROS 규율(BEST_EFFORT 센서 QoS, stream→mode→arm 순서, 게이트 중에도 셋포인트 유지, 서비스 응답이 아니라 텔레메트리로 상태 확인, FC의 착지 판정 후에만 disarm)은 여기서 확정돼 위쪽 노드로 그대로 올라간다
- `aruco_landing_node`: 그 골격에 "이륙 지점이 아니라 마커 위로 내린다" 하나만 추가
- `run_px4`: 명령 하나 + 포그라운드 승인

### 2026-08-13~18 — 트레일러
- `trailer_gps_node` / `trailer_target_node` / 미션 `CRUISE` 단계
- 승인 3회 → **1회**
- SEARCH가 기체가 아니라 **짐벌**을 스윕하도록
- **현장에서 라디오 문제로 중단** → `radio_probe` 추가 (§6)

### 2026-08-21 — `wang` → `main` 병합

---

## 5. 실기체에서 실제로 측정한 수치

### 픽셀 예산 — 이 스택의 모든 고도/각도 결정의 근거

```
유효 마커 픽셀 = fx · edge · sin²(elevation) / h
```

`sin²` 인 이유: nadir를 벗어나면 **두 번** 손해다 — 거리가 슬랜트 거리가 되고,
마커가 단축된다.

0.18 m 마커, `fx=719`, 디코드 하한 ~25 px 기준:

| 고도 | nadir | −60° 링 | −40° 링 | 탐색 반경 |
|---|---|---|---|---|
| 3 m | **43.1** | **32.3** | 17.8 | 1.5 m |
| 5 m | 25.9 (경계) | 19.4 | 10.7 | 2.5 m |
| 8 m | 16 (불가) | — | — | 4.0 m |

결론:
- **고도를 올려 넓게 보려는 건 자기모순이다** — 8 m면 마커는 화면 안에 있고 판독은 불가능하다. 넓은 탐색을 사는 건 고도가 아니라 **더 큰 마커**다 (`탐색반경 ≈ 9 × 마커변`)
- `scan_pitch_deg = [-90, -60]`. `-40°` 링은 이 마커로는 **어떤 고도에서도** 못 읽는다 (마커가 0.45 m가 되면 되살릴 것)
- 탐색 반경이 ~2.6 m로 묶이므로 **트레일러 GPS 안테나는 마커에서 1 m 이내에 달아야 한다.** 이건 소프트웨어로 못 고친다
- **이미 보이는 마커에 짐벌을 조준하지 말 것** — nadir 26 px vs 추적 시 ~19 px, 즉 추적이 마커를 잃게 만든다. `_track_marker` 는 실제로 잃은 뒤에만 재조준한다

### 착륙 제어 (`aruco_landing_node` 기본값)

| 파라미터 | 값 | 이유 |
|---|---|---|
| `takeoff_alt_m` | 5.0 | 이륙 및 순항 고도 |
| `search_alt_m` | 3.0 | 0.18 m 마커가 확실히 잡히는 고도 (위 표) |
| `center_kp` / `center_v_max_m_s` | 0.8 / 0.6 | |
| `descend_radius_m` / `descend_speed_m_s` | 0.30 / 0.30 | 반경 밖에서는 고도 유지하고 정렬부터 |
| `touchdown_alt_m` / `touchdown_xy_m` | 0.40 / 0.20 | 이 아래는 마커가 화면을 벗어나므로 FC LAND에 인계 |
| `cruise_kp` | 0.35 | 보이지 않는 좌표로 날아가는 구간이라 의도적으로 순한 P 게인 |
| 최대속도 / 가속도 / 저크 | **PX4에서 읽음** | `MPC_XY_VEL_MAX` / `MPC_ACC_HOR` / `MPC_JERK_AUTO`. 노드에 숫자를 두지 않고 MAVROS로 읽으며, 못 읽으면 시동 게이트가 막는다. **FCU 기본값 12 m/s를 그대로 두면 12 m/s로 순항한다 — 기체에서 값을 내려둘 것** |
| 실제로 쓰는 가속 | 여유거리에 비례 | PX4 값은 천장이고, **가장 가까운 장애물까지의 여유**로 깎아 쓴다 (`obstacle_accel_*`). 속도가 아니라 가속을 깎는 이유: 인증된 경로에서 옆으로 벗어나게 하는 건 가속이다 (`0.5·a·t²`). 실측 CJU 경로 680점: 여유 0.05–9.35 m → `a_max=4` 기준 1.0–4.0 m/s², 중앙값 2.56, 전개속 구간 16 % |
| `cruise_max_distance_m` | 150.0 | **거부하지 클램프하지 않는다** — 0/0 fix는 수천 km 밖의 멀쩡한 좌표다 |
| `marker_acquire_frames` | 5 | SEARCH→DESCEND는 비가역이라 연속 프레임 요구 |

> **P 제어만으로는 움직이는 데크에 중심을 못 맞춘다.** 정확히 `v_target / kp`
> 에서 정착한다 (0.3 ÷ 0.8 = 0.375 m > 0.30 m 반경) → 영원히 곧 내려갈 것처럼
> 호버링. 해법은 **게인 상향이 아니라 마커 속도 피드포워드**
> (`marker.VelocityEstimate`). 이 레포의 모든 추종 루프에 같은 처방을 쓸 것.

---

## 6. 아직 안 끝난 것 (실기체에서 반드시 확인할 것)

### ⚠️ 1. 짐벌 자세 기준 — 측정 아니고 추론
`landing_tf_node` 의 `gimbal_attitude_reference` 기본값 `'stabilized'` 는 3축 SIYI가
follow 모드에서 어떻게 동작할지에 대한 **추론**이다. 틀리면 기체 회전이 **두 번**
적용된다 — 10° 기울기, 5 m 고도에서 **약 0.9 m 오차**, 착륙 게이트는 0.35 m다.

```bash
ros2 run siyi_gimbal gimbal_monitor    # 기체 위에서 판정
```

단, 이 도구는 **pitch만** 흔든다. "EARTH-referenced" 판정은 `'stabilized'` 를
지지하는 것이지 `'earth'` 를 지지하지 않는다.

### ⚠️ 2. 짐벌 레버암 미측정
`gimbal_mount_xyz_m = [0,0,0]` 은 **0이라는 측정값이 아니라 안 쟀다는 뜻**이다.
FCU 원점 → 짐벌 회전중심의 실제 오프셋(base_link ENU, X전방/Y좌/Z상)을 잴 것.
카메라가 FCU보다 12 cm 앞에 있으면 마커를 12 cm 뒤로 보고한다 — 0.35 m 예산의 1/3.

### ⚠️ 3. tf2 버퍼 지연 — 우회했을 뿐
체인 자체는 건강하다 (`tf2_echo` 해석됨, 외부 프로브로 `gimbal_mount→optical` 50 Hz
/ +1.9 ms). 그런데 `aruco_pose_node` **내부의** tf2 버퍼 최신 샘플이 벽시계보다
**135–880 ms** 뒤처져 캡처 시각 조회가 전부 "미래로의 외삽"으로 실패했다.

해봤지만 안 된 것: `TransformListener(spin_thread=True)`, `MultiThreadedExecutor`
+ 전용 콜백그룹, 이미지 콜백 내부 레이트 제한, 소스단 발행 스로틀(검출기 CPU
111%→84%, 지연은 그대로).

**현재 상태:** `target_frame=""` (광학 프레임 그대로 내보내고 미션 노드가
`header.frame_id` 보고 변환) 로 **우회**. 이 우회는 짐벌이 nadir를 유지한다고
가정하고 MAVROS 헤딩만 쓴다. 기체에서 tf 조회 실패는 사라졌지만 **엔드투엔드는
미확인** (마커가 시야에 없었고 MAVROS가 죽어 있었다).

**다음 가설:** `/tf` QoS가 양쪽 모두 RELIABLE depth-100이라 한 번 멈추면 ~1.25 s
분량이 쌓였다가 순서대로 배달된다. → **BEST_EFFORT로 바꾸지 말고**(rviz/`tf2_echo`
같은 RELIABLE 구독자가 깨진다) KEEP_LAST depth를 줄여 상한을 걸 것.

### ⚠️ 4. 트레일러 라디오 — `SRx_POSITION`
2026-08-18 현장 중단 원인. 위성 수·fix type·HDOP는 다 들어오는데 `/trailer/fix`
가 조용했다. GPS 문제도 라디오 문제도 아니었다.

| 보이는 값 | 메시지 | 스트림 그룹 |
|---|---|---|
| 위성 수, fix type, HDOP | `GPS_RAW_INT` | `SRx_EXT_STAT` |
| **위도/경도** | `GLOBAL_POSITION_INT` | **`SRx_POSITION`** |

`trailer_gps_node` 는 `GLOBAL_POSITION_INT` 만 쓰므로 `SRx_POSITION=0` 이면 정확히
이 증상이다 — **안심되는 숫자는 전부 오고 정작 필요한 하나만 안 온다.** `x` 는
라디오가 꽂힌 시리얼 포트 번호 (SERIAL1 → `SR1_*`).

함정: 이 증상은 "아직 3D fix가 없다"와 **똑같이 생겼는데 대응이 정반대**다
(파라미터 수정 vs 그냥 기다리기). ArduPilot은 EKF에 위치가 설 때까지
`GLOBAL_POSITION_INT` 를 정당하게 보류한다.

```bash
ros2 run trailer_link radio_probe        # 다른 노드 전부 끄고 (포트 하나에 리더 하나)
ros2 run trailer_link radio_probe --device udpin:127.0.0.1:14599   # 책상에서 재현
```

10초간 메시지 종류를 세어 다섯 판정 중 하나와 다음 행동을 찍는다.
`--request-position` 은 런타임 임시 조치고, 영구 해결은 `SRx_POSITION`.

### ❌ 5. 비행 자체가 남았다
PX4 경로의 `aruco` / `trailer` / `mpc` 세 미션 모두 **성공한 실착륙 기록이 없다.**
순서는 코드가 이미 표현하고 있다:

    naive (인식 없이 이륙·착륙)
      → bench (프롭 없이 짐벌 스윕 + 마커 픽스)
      → aruco (고정 마커 위 착륙)
      → trailer / mpc

`naive` 와 `bench` 는 **이 순서를 밟기 위해 만들어진 단계**이지 통과 도장이 아니다
— 실제로 돌린 날짜를 §0 표에 적어 넣는 것부터가 다음 작업이다.

---

## 7. 현장 체크리스트

### 출발 전 (책상)

- [ ] `colcon build && source install/setup.bash` — 젯슨에서
- [ ] 테스트 통과 확인 (현재 234개)
- [ ] `down_camera.yaml` 의 fx가 **지금 달린 카메라**의 값인지
- [ ] `marker_size` 가 **실측한 마커 변 길이**와 같은지 (자로 잰 거리에서 `ros2 topic echo /perception/down/marker_pose --field pose.position` 로 검증)
- [ ] 젯슨 `tailscale status` / PC에서 `ssh sw@100.112.65.33`
- [ ] `systemctl status fastdds-discovery`
- [ ] PC·젯슨 `ROS_DOMAIN_ID` 동일 (=0)
- [ ] 배터리 임계 `min_battery_v` 가 실제 셀 수에 맞는지

### 패드에서

- [ ] `ls /dev/ttyACM* /dev/ttyUSB* /dev/video0` — FC / 라디오 / 카메라
- [ ] `./run_px4 bench` — 프롭 뺀 상태로 짐벌 스윕 + 마커 픽스 확인
- [ ] 트레일러 비행이면 `ros2 run trailer_link radio_probe` 먼저 (다른 노드 끄고)
- [ ] 조종자 RC 대기, 넓은 공터, 모드 전환으로 언제든 회수 가능한지 확인
- [ ] 저고도 호버에서 **"마커 쪽으로 움직이는지"** 눈으로 확인 — 부호 매핑은 벤치에서 확정됐지만 새 기체/새 마운트면 다시 본다

### 비행 후

- [ ] `/tmp/run_px4_mavros.log`, `/tmp/run_px4_stack.log` 회수
- [ ] FC 로그 회수 (`flight_logs/` — colcon의 `log/` 와 다른 디렉터리다)
- [ ] 이 문서의 §0 표 갱신

---

## 8. 트러블슈팅 (하드웨어 고유)

| 증상 | 원인 / 해결 |
|---|---|
| 검출은 되는데 pose가 하나도 안 나옴 | **TF 체인 부재/조회 실패.** `landing_tf_node` 가 도는지, `tf2_echo map down_camera_optical_frame` |
| 영상이 느리고 낡음 (수백 ms) | `usb_cam` 을 쓰고 있다. `gst_camera_node` 로 갈 것 (§4-2) |
| 레이트를 낮춰도 지연이 안 줄어듦 | 큐가 아니다. 디코더가 CPU에 있다 |
| 하드웨어 디코드인데도 몇 fps밖에 안 나옴 | **발행 쪽이다.** `uint8[]` 필드에 `bytes` 를 넘기면 요소마다 검증이 돈다 — 720p 한 장에 457 ms. `array.array('B', ...)` 로 넘길 것 (2026-07-29) |
| 미션이 시동을 거부 | 프리체크 로그가 항목별로 이유를 한 줄씩 찍는다 |
| `approve` 가 거부됨 | 게이트 단계가 아니다. 조기 승인은 삼키지 않고 거부한다 |
| 런치 아래서 ENTER가 안 먹음 | `ros2 launch` 는 stdin을 안 준다. `run_px4` 는 그래서 미션 노드만 포그라운드로 띄운다. 아니면 `ros2 run mpc_landing approve <node>` |
| 짐벌이 안 움직임 | `/siyi_gimbal_node/status` 의 `bad_rx`, 시리얼 `/dev/ttyTHS1` 권한, UDP면 기체 네트워크 |
| MAVROS 연결 실패 | `FCU_URL` 장치 확인, `sudo usermod -aG dialout $USER`, `/tmp/run_px4_mavros.log` |
| 트레일러 좌표가 안 옴 | §6-4. `radio_probe` 먼저 |
| 시리얼 노드가 조용히 안 뜸 | **포트 하나에 리더 하나.** 진 쪽이 로그 파일 속에서 조용히 죽는다. `TRAILER_LINK=0` |
| 캘리브 고쳤는데 안 바뀜 | `aruco_pose_node` 를 **재시작**할 것 (빌드 아님) |
| 노드 로그가 "Logger severity cannot be changed" 로 죽음 | `(info if ok else warn)(msg)` 패턴. rclpy는 로거를 **줄 단위**로 캐시한다 — 한 줄이 두 심각도로 못 찍는다. 절대 쓰지 말 것 |

---

## 9. 문서 상태 메모

`docs/worklog_2026-07-06.md` 가 참조하는 `docs/precision_landing_hw.md` 와,
당시의 `precision_landing` / `camera_detection` 패키지는 **현재 트리에 없다.**
2026-07-2x 구조 개편(`a106c52`, `82854a2`)에서 ArduPilot GUIDED 스택이 PX4
`camera/aruco_landing` + `flight/mpc_landing` 체인으로 대체됐다. 그 시절 코드는
`git show 918d314 -- precision_landing` 으로 볼 수 있다.
