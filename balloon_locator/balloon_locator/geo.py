"""지상국 NED(원점 = 카메라 회전중심) -> WGS84 위경도 + 드론 홈 기준 상대고도. ROS 의존성 없음.

평면 근사. 1 km 이내에서 근사 오차는 수 cm 수준으로 GPS 오차보다 훨씬 작다.
"""
import math

EARTH_R = 6378137.0


def rotate_declination(n: float, e: float, declination_deg: float):
    """자북 기준 (N, E) -> 진북 기준. 편각은 동쪽 +(한국은 약 -8~-9°)."""
    a = math.radians(declination_deg)
    c, s = math.cos(a), math.sin(a)
    return n * c - e * s, n * s + e * c


def offset_latlon(lat_deg: float, lon_deg: float, dn: float, de: float):
    lat = lat_deg + math.degrees(dn / EARTH_R)
    lon = lon_deg + math.degrees(de / (EARTH_R * math.cos(math.radians(lat_deg))))
    return lat, lon


def ned_to_global(n, e, d, cam_lat_deg, cam_lon_deg, cam_height_above_home_m, declination_deg=0.0):
    """카메라 원점 NED [m] -> (위도, 경도, 드론 홈 지면 기준 상대고도[m])."""
    n, e = rotate_declination(n, e, declination_deg)
    lat, lon = offset_latlon(cam_lat_deg, cam_lon_deg, n, e)
    return lat, lon, cam_height_above_home_m - d
