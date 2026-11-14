# ⚙️ PX4 쿼드 테일시터(Quad-Tailsitter) 및 시스템 파라미터 표

본 문서는 `balloon_sim`에 기록된 설정 예시입니다. **실기체에 적용하도록 검증된 파라미터 표가 아닙니다.** 2kg은 기체·배터리를 포함한 총질량 가정이며, 실제 FC 모델·펌웨어 버전·장착 방향·물성은 확인되지 않았습니다. 온보드는 라즈베리 파이로 확인됐지만 세부 모델은 미확인입니다.

> **실기체 일괄 적용 보류:** 기존 `balloon_locator/config/px4_quadtailsitter.params`에는 버전이 혼재한 항목과 시뮬레이션 가정값이 포함돼 있습니다. 파일의 존재나 빌드 성공을 실기체 적합성의 근거로 사용하지 마세요. 이번 문서 정정은 해당 파일이나 FC 설정을 변경하지 않습니다.

---

## 1. 기체 프레임 및 제어 할당 (Airframe & Control Allocation)

조종면(서보 에일러론/엘리베이터) 없이 **4개의 모터 차동 추력(Differential Thrust)과 반토크만으로** 수직 호버링과 수평 전진 비행을 모두 제어하는 쿼드 테일시터 설정입니다.

| 파라미터 이름 | 설정값 | 단위 | 설명 |
| :--- | :---: | :---: | :--- |
| `SYS_AUTOSTART` | **실기체 미확정**<br>`1045` *(legacy SITL)* | - | 기존13003은 로컬v1.17에 없음. 13200도2로터2조종면이라 단순 치환 불가 |
| `CA_AIRFRAME` | **`4`** | - | Control Allocation 기체 타입: `4 = Tailsitter VTOL` |
| `CA_ROTOR_COUNT` | **`4`** | 개 | 추력 로터(모터) 개수 = 4개 |
| `CA_SV_CS_COUNT` | **`0`** | 개 | 조종면(Control Surface) 서보 개수 = 0개 (무타면 차동 추력 방식) |
| `VT_TYPE` | **`0`** | - | VTOL 타입: `0 = Tailsitter` |
| `VT_ELEV_MC_LOCK` | **`1`** | - | 멀티콥터 모드에서 불필요한 서보 잠금 |

---

## 2. 4로터 장착 위치 및 회전 방향 (Rotor Geometry & Spin Matrix)

### 2.1 기체 좌표계(PX4 Body FRD) 기준 로터 배치도
기체를 지면에 수직으로 세워 놓았을 때(코가 하늘 방향):
- **$-\mathbf{z}_b$ (Up / Nose)**: 기체 코(노즈콘 카메라 광축) 및 4개 로터의 추력 방향 (`AZ = -1.0`)
- **$+\mathbf{x}_b$ (Belly / Forward)**: 기체 배면 방향 (전진 피치다운 시 아래를 향하는 면)
- **$+\mathbf{y}_b$ (Right Wing)**: 기체 우측 날개 방향

```
                  [-X_b : 기체 등(Dorsal) 방향]
                         Rotor 3 (CW)
                      (-0.12, 0.00, 0.015)
                              |
                              |
[-Y_b : 좌측 날개]            + (CG)            [+Y_b : 우측 날개]
  Rotor 1 (CCW) --------------+-------------- Rotor 0 (CCW)
(0.00, -0.12, 0.015)          |             (0.00, +0.12, 0.015)
                              |
                         Rotor 2 (CW)
                      (+0.12, 0.00, 0.015)
                  [+X_b : 기체 배(Belly) 방향]
```

### 2.2 로터별 Control Allocation 상세 파라미터 표

| 로터 번호 | 위치 (기체 기준) | 회전 방향 | `PX` `[m]` | `PY` `[m]` | `PZ` `[m]` | `AX` | `AY` | `AZ` | `KM` (반토크 계수) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Rotor 0** (`CA_ROTOR0_*`) | 우측 날개 포드 (`+Y_b`) | **CCW** | `0.000` | `+0.120` | `+0.015` | `0.0` | `0.0` | `-1.0` | **`+0.010`** |
| **Rotor 1** (`CA_ROTOR1_*`) | 좌측 날개 포드 (`-Y_b`) | **CCW** | `0.000` | `-0.120` | `+0.015` | `0.0` | `0.0` | `-1.0` | **`+0.010`** |
| **Rotor 2** (`CA_ROTOR2_*`) | 배면 포드 (`+X_b`) | **CW** | `+0.120` | `0.000` | `+0.015` | `0.0` | `0.0` | `-1.0` | **`-0.010`** |
| **Rotor 3** (`CA_ROTOR3_*`) | 등쪽 포드 (`-X_b`) | **CW** | `-0.120` | `0.000` | `+0.015` | `0.0` | `0.0` | `-1.0` | **`-0.010`** |

> ⚠️ **주의**: 실기체에서 모터 배선 순서(MAIN 1~4)와 회전 방향(CW/CCW)은 반드시 **QGC $\to$ Actuators $\to$ Actuator Testing**에서 프롭을 제거한 상태로 하나씩 돌려 위 표와 일치하는지 확인해야 합니다.

---

## 3. 센서, 보드 장착 방향 및 대기속도계 (Sensors & Board Rotation)

| 파라미터 이름 | 설정값 | 단위 | 설명 |
| :--- | :---: | :---: | :--- |
| `SENS_BOARD_ROT` | 보드 장착 방향에 맞춤<br>*(예: `0` 또는 `25` Pitch 90°)* | - | **필수 확인**: 기체를 이륙 패드에 **코가 하늘을 향하도록 수직으로 세웠을 때**, QGC HUD 인공수평계가 **`Roll = 0°, Pitch = 0°`** 를 가리켜야 합니다. |
| `SYS_HAS_NUM_ASPD` | **`0`** | 개 | 장착된 피토관(Airspeed Sensor) 개수 = 0 (미장착 에러 해제) |
| `FW_USE_AIRSPD` | **`0`** | - | 고정익/천이 제어에서 피토관 대기속도 미사용 |
| `CBRK_AIRSPD_CHK` | **폐기된 항목 — 적용값 미제공** | - | 고정 PX4 v1.17.0 소스에 현행 파라미터 정의가 없습니다. 과거 설정을 읽는 호환 코드만 존재하므로 현재 버전용 설정으로 취급하지 않습니다. |

---

## 4. 추력, 호버링 및 자세/속도 제어기 (Thrust & Attitude/MPC Control)

아래 값은 2 kg 기체 + T-MOTOR PACER P2207 V3 (1950KV, 6S) + P49436-3 프로펠러를 가정한 기존 모델의 값입니다. 사용자가 확인한 모터 정보는 PACER 시리즈까지이며, 실제 세부 모델·KV·배터리·프로펠러 및 실측 추력은 미확인입니다.

| 파라미터 이름 | 설정값 | 단위 | 설명 |
| :--- | :---: | :---: | :--- |
| `THR_MDL_FAC` | **`1.0`** | - | 추력-PWM 선형화 계수 |
| `MPC_THR_HOVER` | **`0.28 ~ 0.68`**<br>*(시뮬 기준 `0.283`, 온보드 기본 `0.68`)* | `[0..1]` | 수직 호버링 정규화 추력 (실기체 호버링시 실제 값으로 튜닝) |
| `MPC_THR_MIN` | **`0.05`** | `[0..1]` | 공중 최소 아이들 추력 |
| `MPC_THR_MAX` | **`1.00`** | `[0..1]` | 최대 추력 상한 |
| `COM_SPOOLUP_TIME` | **`0.5`** | `s` | ARM 직후 모터 스풀업 시간 |
| `MPC_TKO_RAMP_T` | **`0.5`** | `s` | 이륙 추력 램프 시간 |
| `MPC_LAND_SPEED` | **`0.3`** | `m/s` | RTL 복귀 후 최종 지면 접지 하강 속도 |
| `MPC_XY_VEL_MAX` | **`15.0`** | `m/s` | 최대 수평 비행 속도 |
| `MPC_ACC_HOR` | **`3.0`** | `m/s²` | 기본 수평 가속도 |
| `MPC_ACC_HOR_MAX` | **`5.0`** | `m/s²` | 최대 수평 가속도 |
| `MPC_ACC_UP_MAX` | **`6.0`** | `m/s²` | 최대 상승 가속도 |
| `MPC_ACC_DOWN_MAX` | **`5.0`** | `m/s²` | 최대 하강 가속도 |
| `MPC_JERK_MAX` | **`8.0`** | `m/s³` | 최대 저크(Jerk) 제한 |
| `MPC_XY_P` | **`0.95`** | `1/s` | 수평 위치 비례 이득 |
| `MPC_XY_VEL_P_ACC` | **`1.80`** | `m/s² / (m/s)` | 수평 속도 P 이득 |
| `MPC_XY_VEL_I_ACC` | **`0.40`** | `m/s² / m` | 수평 속도 I 이득 |
| `MPC_XY_VEL_D_ACC` | **`0.20`** | - | 수평 속도 D 이득 |
| `MPC_Z_P` | **`1.00`** | `1/s` | 수직 고도 비례 이득 |
| `MPC_Z_VEL_P_ACC` | **`4.00`** | `m/s² / (m/s)` | 수직 속도 P 이득 |
| `MPC_Z_VEL_I_ACC` | **`2.00`** | `m/s² / m` | 수직 속도 I 이득 |

---

## 5. 통신(MAVLink 포워딩) 및 안전(Failsafe) 파라미터

지상국 SiK 라디오(`TELEM1`)로 들어온 `COMMAND_INT` 패킷이 FC를 거쳐 온보드 컴퓨터(`TELEM2` 또는 `USB`)의 [`tailsitter_onboard.py`](file:///home/seyeongw/ros2_ws/PX4-ROS2/balloon_sim/balloon_locator/onboard/tailsitter_onboard.py)까지 전달되기 위해 반드시 설정해야 하는 항목입니다.

| 파라미터 이름 | 설정값 | 단위 | 설명 |
| :--- | :---: | :---: | :--- |
| `MAV_0_FORWARD` | **`1`** (`Enabled`) | - | MAVLink 인스턴스 0 (예: TELEM1 SiK 모뎀) 패킷 포워딩 활성화 |
| `MAV_1_FORWARD` | **`1`** (`Enabled`) | - | MAVLink 인스턴스 1 (예: TELEM2/USB 온보드 컴퓨터) 패킷 포워딩 활성화 |
| `SER_TEL1_BAUD` | **`57600`** | `bps` | TELEM1 포트(SiK 900MHz 라디오) 통신 속도 |
| `SER_TEL2_BAUD` | **`921600`** | `bps` | TELEM2 포트(온보드 컴퓨터 UART 연결 시) 통신 속도 |
| `COM_OF_LOSS_T` | **`1.0`** | `s` | OFFBOARD 명령 두절 시 Failsafe 전환 대기 시간 |
| `COM_OBL_RC_ACT` | **실기체 적용값 미선정** | - | PX4 v1.17.0의 값 의미는 `3 = Return`, `4 = Land`입니다. 이전 문서의 `4 = Return`은 오류였습니다. 실제 FC 버전 및 현재 설정은 확인되지 않았습니다. |
| `COM_RCL_EXCEPT` | **실기체 적용값 미선정** | - | RC 신호 상실 보호 동작의 예외를 지정하는 항목입니다. 이전 예시를 실기체 권장값으로 사용하지 마세요. |

값 의미의 근거: [PX4 v1.17 Offboard 문서](https://docs.px4.io/v1.17/en/flight_modes/offboard#offboard-parameters), [고정 PX4 소스의 정의](https://github.com/PX4/PX4-Autopilot/blob/d6f12ad1c4f70ad3230afd7d86e971421e02fef4/src/modules/commander/commander_params.c#L355-L369). 문서 대조는 실제 통신 상실·복귀·착륙 시험을 대신하지 않습니다.

---

## 6. 드론 온보드 제어기([`tailsitter_onboard.py`](file:///home/seyeongw/ros2_ws/PX4-ROS2/balloon_sim/balloon_locator/onboard/tailsitter_onboard.py)) 실행 파라미터 표

드론 탑재 컴퓨터(Jetson Orin Nano / Raspberry Pi)에서 [`run_tailsitter.sh`](file:///home/seyeongw/ros2_ws/PX4-ROS2/balloon_sim/balloon_locator/onboard/run_tailsitter.sh) 실행 시 사용하는 CLI 파라미터입니다.

| CLI 옵션 | 기본값 | 단위 | 설명 |
| :--- | :---: | :---: | :--- |
| `--url` | `udpin:0.0.0.0:14601` | - | 실기체 USB 연결 시 `/dev/ttyACM0`, UART 연결 시 `/dev/ttyTHS1` 지정 |
| `--baud` | `921600` | `bps` | FCU 시리얼 통신 속도 |
| `--takeoff-alt` | **`20.0`** | `m` | 발진 승인 수신 후 수직 이륙 목표 상대고도 (`min_alt` ~ `max_alt` 사이) |
| `--cruise-speed` | **`14.0`** | `m/s` | 피치다운 천이 후 풍선을 향해 날아가는 목표 속도 |
| `--terminal-dist` | **`40.0`** | `m` | 중기 UKF 유도에서 온보드 노즈콘 카메라 3D LOS/PN 종말 유도로 전환하는 거리 |
| `--pass-radius` | **`2.2`** | `m` | 풍선 통과 판정 Condition 1 최근접 거리 임계값 |
| `--hover-thrust` | **`0.68`** | `[0..1]` | 온보드 제어기의 기준 호버링 추력 (실기체 중량/배터리 전압에 맞춰 조정) |
| `--balloon-diameter` | **`0.30`** | `m` | 실제 풍선 직경 (종말 단계 광학 거리 $R_{\text{opt}} = f_x D / d_{\text{px}}$ 산출용) |
| `--fx`, `--fy` | **`640.0`**, **`640.0`** | `px` | 온보드 노즈콘 카메라(`/dev/video0`)의 수평/수직 초점거리 |
| `--min-samples` | **`30`** *(또는 지상국과 동일)* | 개 | 지상국 `COMMAND_INT` 패킷 수락 최소 UKF 샘플 수 `n` |
| `--max-sigma` | **`2.0`** | `m` | 지상국 `COMMAND_INT` 패킷 수락 최대 위치 불확실도 `s` |
