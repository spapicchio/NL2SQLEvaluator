import math


class Value:
    def __init__(self, raw, epsilon=1e-6):
        self.raw = None if isinstance(raw, float) and math.isnan(raw) else raw
        self.epsilon = epsilon

    def __eq__(self, other):
        if not isinstance(other, Value):
            return NotImplemented

        a, b = self.raw, other.raw
        if isinstance(a, float) and isinstance(b, float):
            return abs(a - b) <= self.epsilon

        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return abs(float(a) - float(b)) <= self.epsilon

        return a == b

    def __hash__(self):
        if isinstance(self.raw, (int, float)):
            precision = max(0, int(-math.log10(self.epsilon)))
            rounded = round(float(self.raw), precision)
            return hash(rounded)
        return hash(self.raw)
