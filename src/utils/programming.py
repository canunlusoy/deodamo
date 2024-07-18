from types import LambdaType
from typing import Any, Callable, Iterable


def is_lambda_function(obj: Any) -> bool:
    # https://stackoverflow.com/questions/23852423/how-to-check-that-variable-is-a-lambda-function
    return isinstance(obj, LambdaType) and obj.__name__ == "<lambda>"


class EventHandler:

    def __init__(self, name: str):
        self.name = name
        self.events_subscribers: dict[str, list[Callable]] = {}

    def create_event(self, event_name: str):
        if event_name not in self.events_subscribers:
            self.events_subscribers[event_name] = []
        else:
            message = f'An event "{event_name}" has already been registered with handler "{self.name}"'
            raise KeyError(message)

    def create_events(self, event_names: Iterable[str]):
        for event_name in event_names:
            self.create_event(event_name)

    def subscribe(self, event_name: str, callback: Callable):
        if event_name not in self.events_subscribers:
            message = f'Handler "{self.name}" does not handle an event "{event_name}", cannot subscribe to event.'
            raise KeyError(message)

        self.events_subscribers[event_name].append(callback)

    def emit(self, event_name: str, *args, **kwargs):
        if event_name not in self.events_subscribers:
            message = f'Handler "{self.name}" does not handle an event "{event_name}", cannot emit the event signal.'
            raise KeyError(message)

        for callback in self.events_subscribers[event_name]:
            callback(*args, **kwargs)


