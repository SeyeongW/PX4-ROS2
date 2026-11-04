"""필수 노드가 종료되면 전체 실행도 실패로 끝내는 공통 감시기다."""
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessExit


def required_process_exited(event, context):
    if not context.is_shutdown:
        # Humble의 정상 종료와 구분하도록 예외로 실패 코드를 반환한다.
        raise RuntimeError(
            f"Required process '{event.process_name}' exited unexpectedly "
            f'(code {event.returncode})')
    return []


def watch_required_process(process):
    """프로세스를 시작하기 전에 등록하며 종료 코드 0도 예상 밖 종료로 본다."""
    return RegisterEventHandler(OnProcessExit(
        target_action=process, on_exit=required_process_exited))
