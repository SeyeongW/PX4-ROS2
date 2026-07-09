#!/usr/bin/env bash
# 컨테이너 메인 프로세스(CMD)용 진입점. docker exec 셸은 이걸 안 거치므로
# 실제 환경설정은 env.sh에 있고, 여긴 그걸 불러오기만 함
# (BASH_ENV/~/.bashrc가 exec 셸에서 env.sh를 불러옴).
set -e
source /entrypoint-env.sh
exec "$@"
