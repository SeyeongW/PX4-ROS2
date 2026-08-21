#!/usr/bin/env bash
# ── 착륙용 젯슨(100.112.65.33) FastDDS 환경설정 ──────────────────────────────
#
#   설치:  scp config/jetson_dds_fastdds.sh sw@100.112.65.33:~/dds_fastdds.sh
#   사용:  ~/.bashrc 끝(ROS setup.bash 이후)에서  source ~/dds_fastdds.sh
#
# 이 젯슨은 Discovery Server를 자기가 돌린다(systemd: fastdds-discovery.service,
# 0.0.0.0:11811). PC는 tailscale로 그 서버에 붙는다.
#
# PC쪽 ~/dds_fastdds.sh 와 짝이다. 양쪽 ROS_DOMAIN_ID가 같아야 한다(=0).
#
#
# 왜 whitelist에 127.0.0.1 이 같이 들어가는가  ← PC쪽 스크립트와 다른 점
# ─────────────────────────────────────────────────────────────────────
# PC 스크립트는 tailscale IP 하나만 whitelist 한다. 그걸 젯슨에 그대로 쓰면
# 젯슨의 모든 ROS 통신이 tailscale에 묶인다 — 비행 중 tailscale이 끊기면
# 카메라→검출기→미션 노드 링크까지 같이 죽는다. 기체가 떠 있는 상태에서.
#
# 그래서 여기선 loopback도 같이 연다:
#
#   127.0.0.1        젯슨 안에서 도는 비행 스택. tailscale과 무관하게 항상 산다.
#   100.112.65.33    PC가 영상/그래프를 보는 경로. 끊겨도 비행엔 영향 없다.
#
# ROS_DISCOVERY_SERVER도 같은 이유로 127.0.0.1을 쓴다. 서버는 0.0.0.0에 바인딩된
# 같은 프로세스이므로 PC가 100.112.65.33으로 붙어도 그래프는 하나로 합쳐진다.
#
# whitelist에서 빠지는 것들이 핵심이다: wifi LAN IP(10.17.117.150)와 docker0.
# 이 둘이 광고되면 PC가 닿지도 못하는 주소로 연결을 시도한다.
#
#
# 왜 maxMessageSize 1200 인가
# ───────────────────────────
# tailscale MTU가 1280이다. FastDDS 기본값은 65500바이트짜리 UDP 데이터그램이라
# 1280 터널을 지나며 IP 단에서 ~50조각으로 쪼개진다. 그 중 하나만 유실돼도
# 프레임 전체가 버려지므로, 압축 이미지처럼 큰 메시지는 사실상 도착하지 못한다.
# RTPS 레벨에서 미리 1200으로 쪼개면 조각 하나 = 패킷 하나가 되어 재전송이 먹는다.
# ─────────────────────────────────────────────────────────────────────────────

DDS_SERVER_PORT="${DDS_SERVER_PORT:-11811}"

# 부팅 직후엔 tailscale이 아직 안 올라왔을 수 있다. 대화형 셸은 기다릴 이유가
# 없으므로 기본 1회만 본다.
_TS_IP=""
for _i in $(seq 1 "${DDS_TS_WAIT:-1}"); do
  _TS_IP=$(tailscale ip -4 2>/dev/null | head -1)
  [ -n "$_TS_IP" ] && break
  sleep 1
done

_PROF="$HOME/.fastdds_jetson.gen.xml"

# tailscale이 없어도 loopback만으로 프로파일을 만든다. PC에서 못 보게 될 뿐,
# 젯슨 안의 비행 스택은 그대로 돈다 — 이게 tailscale 유무보다 중요하다.
_WL="        <interfaceWhiteList><address>127.0.0.1</address>"
[ -n "$_TS_IP" ] && _WL="${_WL}<address>${_TS_IP}</address>"
_WL="${_WL}</interfaceWhiteList>"

cat > "$_PROF" <<XML
<?xml version="1.0" encoding="UTF-8" ?>
<dds xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles">
  <profiles>
    <transport_descriptors>
      <transport_descriptor>
        <transport_id>ts_udp</transport_id>
        <type>UDPv4</type>
${_WL}
        <maxMessageSize>1200</maxMessageSize>
        <sendBufferSize>1048576</sendBufferSize>
        <receiveBufferSize>4194304</receiveBufferSize>
      </transport_descriptor>
    </transport_descriptors>

    <!-- SUPER_CLIENT: 노드를 서버에 등록하는 동시에 그래프 전체를 받는다.
         일반 CLIENT면 ros2 topic list 가 자기가 관여하는 토픽만 보여준다. -->
    <participant profile_name="jetson_ts" is_default_profile="true">
      <rtps>
        <userTransports><transport_id>ts_udp</transport_id></userTransports>
        <useBuiltinTransports>false</useBuiltinTransports>
        <builtin>
          <discovery_config>
            <discoveryProtocol>SUPER_CLIENT</discoveryProtocol>
            <discoveryServersList>
              <!-- prefix 는 ROS2가 \`-i 0\` 서버에 부여하는 표준 GUID. 서버를
                   다른 id로 띄우면 여기도 같이 바꿔야 붙는다. -->
              <RemoteServer prefix="44.53.00.5f.45.50.52.4f.53.49.4d.41">
                <metatrafficUnicastLocatorList>
                  <locator><udpv4>
                    <address>127.0.0.1</address>
                    <port>${DDS_SERVER_PORT}</port>
                  </udpv4></locator>
                </metatrafficUnicastLocatorList>
              </RemoteServer>
            </discoveryServersList>
          </discovery_config>
        </builtin>
      </rtps>
    </participant>
  </profiles>
</dds>
XML

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DISCOVERY_SERVER="127.0.0.1:${DDS_SERVER_PORT}"
export ROS_SUPER_CLIENT=true
export ROS_DOMAIN_ID=0          # PC와 반드시 동일
export FASTRTPS_DEFAULT_PROFILES_FILE="$_PROF"
unset CYCLONEDDS_URI

# systemd 유닛이 EnvironmentFile= 로 읽도록 평문으로도 남긴다(systemd는 .bashrc를
# 읽지 않는다).
{
  echo "RMW_IMPLEMENTATION=rmw_fastrtps_cpp"
  echo "ROS_DISCOVERY_SERVER=${ROS_DISCOVERY_SERVER}"
  echo "ROS_SUPER_CLIENT=true"
  echo "ROS_DOMAIN_ID=0"
  echo "FASTRTPS_DEFAULT_PROFILES_FILE=${_PROF}"
} > "$HOME/.dds_env"

# daemon 리셋은 "설정이 실제로 바뀐 경우"에만. .bashrc에서 매번 source 되는데
# 무조건 stop 하면 새 터미널을 열 때마다 다른 터미널이 쓰던 데몬을 죽인다.
_SIG="jetson-dds:${ROS_DISCOVERY_SERVER}:${ROS_DOMAIN_ID}:${_TS_IP}"
if [ "$_DDS_ACTIVE_SIG" != "$_SIG" ]; then
  ros2 daemon stop >/dev/null 2>&1
  export _DDS_ACTIVE_SIG="$_SIG"
fi

if [ -n "$_TS_IP" ]; then
  echo "[DDS] DiscoveryServer ${ROS_DISCOVERY_SERVER}  iface 127.0.0.1 + ${_TS_IP}"
else
  echo "[DDS] DiscoveryServer ${ROS_DISCOVERY_SERVER}  iface 127.0.0.1 only" \
       "(tailscale 미감지 → PC에서 안 보임, 젯슨 로컬 비행은 정상)"
fi
unset _i _SIG _PROF _TS_IP _WL
