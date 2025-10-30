"""
Discrete event simulation engine.

Provides a priority queue-based event system for efficient simulation
of asynchronous network events.
"""

import heapq
from typing import Callable, Any
from dataclasses import dataclass, field


@dataclass(order=True)
class Event:
    """
    An event in the simulation.

    Attributes:
        time: When the event occurs (in milliseconds)
        priority: Priority for events at the same time (lower = higher priority)
        handler: Function to call when event fires
        args: Positional arguments for handler
        kwargs: Keyword arguments for handler
        description: Human-readable description
    """
    time: float
    priority: int = 0
    handler: Callable = field(compare=False)
    args: tuple = field(default_factory=tuple, compare=False)
    kwargs: dict = field(default_factory=dict, compare=False)
    description: str = field(default="", compare=False)

    def execute(self) -> Any:
        """Execute the event handler."""
        return self.handler(*self.args, **self.kwargs)


class EventQueue:
    """
    Priority queue for managing simulation events.

    Implements discrete event simulation with efficient O(log n)
    insertion and removal.
    """

    def __init__(self):
        self._queue: list[Event] = []
        self._current_time: float = 0.0
        self._event_count: int = 0

    def schedule(
        self,
        delay: float,
        handler: Callable,
        *args,
        priority: int = 0,
        description: str = "",
        **kwargs
    ) -> Event:
        """
        Schedule an event to occur after a delay.

        Args:
            delay: Milliseconds from now when event should fire
            handler: Function to call when event fires
            *args: Positional arguments for handler
            priority: Priority for events at same time (lower = higher)
            description: Human-readable description
            **kwargs: Keyword arguments for handler

        Returns:
            The created Event object
        """
        event_time = self._current_time + delay
        event = Event(
            time=event_time,
            priority=priority,
            handler=handler,
            args=args,
            kwargs=kwargs,
            description=description
        )
        heapq.heappush(self._queue, event)
        self._event_count += 1
        return event

    def schedule_at(
        self,
        time: float,
        handler: Callable,
        *args,
        priority: int = 0,
        description: str = "",
        **kwargs
    ) -> Event:
        """
        Schedule an event to occur at an absolute time.

        Args:
            time: Absolute time in milliseconds when event should fire
            handler: Function to call when event fires
            *args: Positional arguments for handler
            priority: Priority for events at same time (lower = higher)
            description: Human-readable description
            **kwargs: Keyword arguments for handler

        Returns:
            The created Event object
        """
        event = Event(
            time=time,
            priority=priority,
            handler=handler,
            args=args,
            kwargs=kwargs,
            description=description
        )
        heapq.heappush(self._queue, event)
        self._event_count += 1
        return event

    def pop(self) -> Event | None:
        """
        Get and remove the next event.

        Returns:
            The next event or None if queue is empty
        """
        if not self._queue:
            return None
        return heapq.heappop(self._queue)

    def peek(self) -> Event | None:
        """
        Get the next event without removing it.

        Returns:
            The next event or None if queue is empty
        """
        if not self._queue:
            return None
        return self._queue[0]

    def advance_to(self, time: float):
        """Advance current time without processing events."""
        self._current_time = max(self._current_time, time)

    @property
    def current_time(self) -> float:
        """Current simulation time in milliseconds."""
        return self._current_time

    @property
    def size(self) -> int:
        """Number of pending events."""
        return len(self._queue)

    @property
    def total_events(self) -> int:
        """Total number of events created."""
        return self._event_count

    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return len(self._queue) == 0

    def run_until(self, end_time: float, max_events: int | None = None) -> int:
        """
        Run simulation until a specific time or event count.

        Args:
            end_time: Stop when simulation time reaches this value
            max_events: Stop after processing this many events (optional)

        Returns:
            Number of events processed
        """
        events_processed = 0

        while not self.is_empty():
            event = self.peek()
            if event is None or event.time > end_time:
                break
            if max_events is not None and events_processed >= max_events:
                break

            # Pop and execute
            event = self.pop()
            self._current_time = event.time
            event.execute()
            events_processed += 1

        # Advance to end time if we stopped early
        if self.is_empty() or (max_events is not None and events_processed >= max_events):
            self._current_time = end_time

        return events_processed

    def clear(self):
        """Clear all pending events."""
        self._queue.clear()
