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


# class EventDetector:
#     def __init__(self):
#         self.detector = EventDetector()
#         self.detected_events = []

#     def detect(self, cache: list[np.ndarray]) -> str:
#         new_img = cache[-1]
#         for old_img in cache[:-1]:
#             new_event = self.detector.detect(old_img, new_img)
#             self.detected_events.append(new_event)

#     def _is_priority_event(self):
#         if "En passant"


class EventDetector:
    def __init__(self):
        self.points_old = None
        self.points_new = None
        self.changed = None

    def detect(
        self, points_old: np.ndarray, points_new: np.ndarray
    ) -> EventType:
        if np.sum(points_old) + 1 == np.sum(points_new):
            return EventType.NewPiece

        self.points_old = points_old
        self.points_new = points_new
        self.changed = self.points_old != self.points_new
        if np.sum(points_old) == np.sum(points_new) + 1:
            return self._case_captured()
        return self._case_not_captured()

    def _case_captured(self) -> EventType:
        if self._is_en_passant():
            return EventType.EnPassant
        return EventType.Capture

    def _case_not_captured(self) -> EventType:
        if np.all(self.points_old == self.points_new):
            return EventType.NoEvent
        if self._is_castle():
            if self._is_short_castle():
                return EventType.ShortCastle
            else:
                return EventType.LongCastle
        return EventType.Move

    def _is_castle(self) -> bool:
        return np.sum(self.changed) == 4

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

    def _changed(self, x: int, y: int, mirror=True):
        if mirror:
            return self.changed[x, y] is True or self.changed[y, x] is True
        return self.changed[x, y] is True

    def _is_en_passant(self) -> bool:
        return np.sum(self.changed) == 3
