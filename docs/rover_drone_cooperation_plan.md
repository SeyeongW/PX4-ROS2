# 로버-드론 협동 시스템 설계안 (Rover–Drone Cooperation Plan)

움직이는 지상 플랫폼을 **펌웨어 없는 가짜 차량(`aruco_platform`)** 에서 **실제 ArduPilot
Rover(SITL)** 로 교체하고, 그 위에서 드론(ArduCopter)이 정밀착륙/협동하도록 발전시키기 위한
설계 문서. 최종적으로는 로버·드론이 각자 컴패니언 컴퓨터를 탑재하고 ROS 2(DDS)로 네트워크
연결되는 실기 시스템을 목표로 하며, **시뮬레이션은 그 구조를 그대로 본뜨되 연결 엔드포인트
(포트/IP)만 바꾸면 실기로 이식되도록** 설계한다.

---

## 0. 설계 원칙 (전체를 관통하는 규칙)

1. **계층 분리** — Gazebo(물리) / ArduPilot SITL(FC 펌웨어) / MAVROS+ROS(컴패니언 SW)는
   서로 독립. 컴패니언 컴퓨터는 *하드웨어로 시뮬레이션하지 않고*, 그 안에서 돌 *소프트웨어
   (ROS 노드)만* 실행한다.
2. **차량별 네임스페이스 고정** — `/drone/*`, `/rover/*`. sim = 실기 동형 토픽 그래프.
3. **차량당 MAVROS 인스턴스 분리** — 실기 전환 시 `fcu_url`만 교체.
4. **차량 간 통신은 ROS 토픽으로만** — localhost 가정/공유메모리 지름길 금지(DDS 분산 대비).
5. **모든 엔드포인트 파라미터화** — 포트/IP를 코드에 하드코딩하지 않는다.

### 계층 대응 (실기 ↔ 시뮬레이션)

```
계층                       실기 시스템                    Gazebo SITL
────────────────────────────────────────────────────────────────────
5. 컴패니언 SW (ROS)   드론 Jetson / 로버 Jetson      ← 동일한 ROS 노드, 한 PC에서 실행
   (MAVROS + 인지/제어)  각자 보드에서 실행
        │ MAVLink (UDP/serial)                        │ MAVLink (UDP localhost)
4. 통신 링크          텔레메트리 / 시리얼            ← SITL이 14550/14560 UDP로 노출
        │                                             │
3. 비행제어 펌웨어     실제 FC (Pixhawk)             ← ArduCopter / ArduRover SITL (PC)
        │                                             │ ArduPilotPlugin FDM
2. FDM 경계           (없음 — 실제 물리)            ← 포트 9002 / 9012
        │                                             │
1. 물리 / 센서        현실 세계                      ← Gazebo (하나의 월드, 기체 2대)
```

실기 전환 시 바뀌는 것은 **계층 4의 연결 엔드포인트(`fcu_url`/포트)와 계층 5 노드의 물리적
분산 배치**뿐. 계층 1–3(Gazebo + SITL)은 실기에선 통째로 사라지고 실물 FC + 현실 물리로
대체된다. 이 repo에는 이미 PC↔Jetson DDS 구성(`config/cyclonedds_pc.xml`,
`config/cyclonedds_jetson.xml`)이 있어, 이를 드론보드↔로버보드용으로 확장만 하면 된다.

---

## 1. 포트 / 엔드포인트 맵 (충돌 방지의 핵심)

| 항목 | 드론 (I0) | 로버 (I1) |
|---|---|---|
| ArduPilot | ArduCopter, `-f JSON` | ArduRover, `-f JSON --frame rover-skid` |
| FDM (플러그인 ↔ SITL) | 9002 / 9003 | **9012 / 9013** |
| 모델 플러그인 `fdm_port_in` | 9002 | **9012** |
| MAVLink (SITL ↔ MAVROS) | udp 14550 | **14560** |
| ROS 네임스페이스 | `/drone` | `/rover` |

> SITL `-I1`은 FDM·MAVLink 포트를 자동으로 +10 오프셋한다. 모델의 `ArduPilotPlugin`에는
> **`<fdm_port_in>9012</fdm_port_in>` 를 반드시 명시**해야 드론(9002)과 충돌하지 않는다.

---

## 2. Gazebo 모델 설계

### 2-1. 스키드 스티어 로버 (`gazebo/models/skid_rover/`)

- `chassis` 평판 섀시 + 구동 바퀴 ×4 (좌 2 / 우 2, `revolute`, Y축 회전). **조향 조인트 없음**
  — 좌·우 바퀴 속도차로 선회(스키드 스티어).
- `imu_link` + IMU 센서(섀시에 고정) → `ArduPilotPlugin`의 `imuName`으로 참조.
- `ArduPilotPlugin`: `fdm_port_in 9012`, `lock_step 1`.
  - ArduPilot 파라미터: `SERVO1_FUNCTION=73`(ThrottleLeft), `SERVO3_FUNCTION=74`(ThrottleRight).
  - `<control channel="0">` → 좌측 두 바퀴(VELOCITY), `<control channel="2">` → 우측 두 바퀴.
  - throttle은 가역(전/후진)이므로 PWM 중립 1500 = 정지가 되도록 `offset=-multiplier/2` 매핑.
- 전방 라인추종 카메라: 견인차(로버)에 탑재(기존 `aruco_platform` 카메라 설정 재사용).
- 상판 ArUco 마커 + `DetachableJoint`(parent=chassis, child=iris base_link) — 드론 탑재 발진.

### 2-2. 트레일러 (3단계 옵션)

- 로버 `chassis` ↔ 트레일러 `bed` 사이 `revolute` 히치(Z yaw) + 댐핑.
- 캐스터 / 종동륜(ArduPilot 제어 없음, passive). ArduPilot은 견인차만 제어하고 트레일러는
  순수 물리로 끌려온다.
- 적재함 상면: ArUco 마커 + `DetachableJoint`(parent=trailer bed)로 이전.
- 트레일러는 회전·가속 시 스윙(jackknife)하므로 **착륙은 로버 정지(HOLD) 상태에서** 수행.
- ⚠️ `DetachableJoint`의 child 모델 네임 스코프 이슈는 기존 `aruco_platform/model.sdf` 주석
  규칙과 동일하게 처리.

### 2-3. 월드

- `iris_down_camera_runway.sdf`에 `skid_rover` 배치, 드론은 로버 상판 z 위에 스폰.
- ArduPilot 로버 모드일 때 기존 가짜 무버(`moving_marker_node` / `track_follower_node`)는
  비활성화(가짜 플랫폼 구동과 충돌).

---

## 3. ROS / 컴패니언 레이어 (launch 구조)

```
/drone/mavros        ← udp://127.0.0.1:14550   (실기: 드론 FC)
/rover/mavros        ← udp://127.0.0.1:14560   (실기: 로버 FC)
/drone/perception, /drone/offboard            (드론 컴패니언에서 돌 노드)
/rover/control                                 (로버 컴패니언에서 돌 노드)
협동 노드: 양 네임스페이스 구독/발행          (sim = 한 PC / 실기 = DDS LAN)
```

- DDS: sim은 한 머신·한 `ROS_DOMAIN_ID`. 실기는 두 머신 같은 도메인 + peer 설정
  (`config/cyclonedds_*.xml` 확장).
- 노드 코드·토픽 이름은 sim/실기에서 **변경 없음**. 바뀌는 것은 MAVROS `fcu_url`과 노드의
  물리적 배치뿐.

---

## 4. 실행 절차

```bash
# 터미널 1: 드론 SITL
sim_vehicle.py -v ArduCopter -f JSON -I0 --console --map
# 터미널 2: 로버 SITL (스키드 파라미터는 EEPROM에 안 남으므로 매번 파일로 강제)
sim_vehicle.py -v Rover --model JSON -I1 --console \
  --add-param-file=gazebo/config/rover_skid.parm
# 터미널 3: Gazebo (월드에 두 기체)
./gazebo/run_sim.sh
# 터미널 4: ROS bringup (차량별 MAVROS + 네임스페이스 노드)
ros2 launch <pkg> coop_bringup.launch.py
```

---

## 5. 단계별 로드맵

| Phase | 내용 | 검증 |
|---|---|---|
| **0** | 스키드 로버 SDF + 플러그인(9012) 단독 | `-v Rover -I1`로 GUIDED 주행됨 |
| **1** | 평판 로버 상판에 마커+드론 직접 탑재, 드론 SITL 동시 | **정지 상태 정밀착륙** 성공 |
| **2** | 로버 GUIDED 주행 중 드론 추종/착륙 | 협동 핵심 동작 |
| **3** | 트레일러 분리 구조 도입 | 정지 시 트레일러 착륙 |
| **4** | 주행 중 랑데부/추종 | 고급 협동 |
| **실기** | `fcu_url`만 실물 FC로 교체 + 노드 두 Jetson 분산 | 코드 변경 0 |

---

## 6. 리스크 체크리스트

- [ ] **포트 충돌** — 로버 플러그인 `fdm_port_in 9012` 필수.
- [ ] **lock_step** — 한 Gazebo 서버에 기체 2대 동기화 동작 확인.
- [ ] **스키드 바퀴 마찰(mu)** — 너무 높으면 선회 불가, 너무 낮으면 미끄러짐.
- [ ] **트레일러 스윙** — 착륙은 HOLD 상태에서.
- [ ] **DetachableJoint child 네임 스코프** — 기존 이슈와 동일.
- [ ] **엔드포인트 파라미터화** — 실기 도입 보장을 위해 포트/IP 하드코딩 금지.
</content>
</invoke>
