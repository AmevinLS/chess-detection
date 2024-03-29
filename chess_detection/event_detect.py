from collections import Counter
from enum import StrEnum

import numpy as np


class EventType(StrEnum):
    NoEvent = ""
    Capture = "Piece Captured"
    EnPassant = "En Passant"
    Move = "Piece Moved"
    NewPiece = "New Piece"
    LongCastle = "Long Castle"
    ShortCastle = "Short Castle"


class EventDetector:
    def __init__(self, queue_len: int = 5):
        self.detector = SingleEventDetector()
        self.detected_events = []
        self.cache = []
        self.queue_len = queue_len

    def push_new_points(self, points: np.ndarray) -> None:
        if len(self.cache) == self.queue_len:
            self.cache.pop(0)
        self.cache.append(points)

    def detect(self) -> EventType:
        self.detected_events = []
        new_points = self.cache[-1]

        for old_points in self.cache[:-1]:
            new_event = self.detector.detect(old_points, new_points)
            self.detected_events.append(new_event)

        if (evt := self._try_get_priority_event()) is not None:
            return evt
        return self._try_by_majority_rule()

    def _try_get_priority_event(self) -> EventType | None:
        for event in self.detected_events:
            if event == EventType.EnPassant:
                return EventType.EnPassant
            if event == EventType.ShortCastle:
                return EventType.ShortCastle
            if event == EventType.LongCastle:
                return EventType.LongCastle
        return None

    def _try_by_majority_rule(self) -> EventType:
        self.detected_events = tuple(
            filter(lambda x: x != EventType.NoEvent, self.detected_events)
        )
        if len(self.detected_events) == 0:
            return EventType.NoEvent

        counter = Counter(self.detected_events)
        return counter.most_common(n=1)[0][0]


class SingleEventDetector:
    def __init__(self):
        self.points_old = None
        self.points_new = None
        self.changed = None

    def detect(
        self, points_old: np.ndarray, points_new: np.ndarray
    ) -> EventType:
        sum_old = np.sum(points_old)
        sum_new = np.sum(points_new)

        if sum_old + 1 == sum_new:
            return EventType.NewPiece

        self.points_old = points_old
        self.points_new = points_new
        self.changed = self.points_old != self.points_new
        if sum_old == sum_new + 1:
            return self._case_captured()
        if sum_old == sum_new:
            return self._case_not_captured()
        return EventType.NoEvent

    def _case_captured(self) -> EventType:
        match np.sum(self.changed):
            case 3:
                return EventType.EnPassant
            case 1:
                return EventType.Capture
            case _:
                return EventType.NoEvent

    def _case_not_captured(self) -> EventType:
        match np.sum(self.changed):
            case 2:
                return EventType.Move
            case 4:
                if self._is_short_castle():
                    return EventType.ShortCastle
                elif self._is_long_castle():
                    return EventType.LongCastle
                else:
                    return EventType.NoEvent
            case _:
                return EventType.NoEvent

    def _is_short_castle(self) -> bool:
        if self._changed(1, 0) and self._changed(2, 0):
            return True
        if self._changed(1, 7) and self._changed(2, 7):
            return True
        if self._changed(5, 0) and self._changed(6, 0):
            return True
        if self._changed(5, 7) and self._changed(6, 7):
            return True
        return False

    def _is_long_castle(self) -> bool:
        if self._changed(0, 2) and self._changed(0, 3):
            return True
        if self._changed(0, 4) and self._changed(0, 5):
            return True
        if self._changed(7, 2) and self._changed(7, 3):
            return True
        if self._changed(7, 4) and self._changed(7, 5):
            return True
        return False

    def _changed(self, x: int, y: int, mirror=True):
        if mirror:
            return self.changed[x, y] is True or self.changed[y, x] is True
        return self.changed[x, y] is True
