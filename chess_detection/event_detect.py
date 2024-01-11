import numpy as np

class EventDetector:
    def __init__(self):
        self.points_old = None
        self.points_new = None
        self.changed = None

    def detect(self, points_old: np.ndarray, points_new: np.ndarray) -> str:
        if np.sum(points_old) < np.sum(points_new):
            return "New piece"
        
        self.points_old = points_old
        self.points_new = points_new
        self.changed = self.points_old != self.points_new
        if np.sum(points_old) > np.sum(points_new):
            return self._case_captured()
        return self._case_not_captured()

    def _case_captured(self) -> str:
        if self._is_en_passant():
            return "En passant"
        return "Piece captured"

    def _case_not_captured(self) -> str:
        if np.all(self.points_old == self.points_new):
            return ""
        if self._is_castle():
            if self._is_short_castle():
                return "Short castle"
            else:
                return "Long castle"
        return "Piece moved"
    
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
        return np.sum(self.changed) == 2
