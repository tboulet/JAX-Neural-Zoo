def execute_once(func):
    def wrapper(*args, **kwargs):
        if not wrapper.has_executed:
            wrapper.has_executed = True
            return func(*args, **kwargs)

    wrapper.has_executed = False
    return wrapper


@execute_once
def warning_message(message_warning: str):
    print(f"[WARNING] {message_warning}")
