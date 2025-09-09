from dataclasses import dataclass, field


@dataclass
class Bbox:
    x0: float = field(default_factory=float)
    y0: float = field(default_factory=float)
    x1: float = field(default_factory=float)
    y1: float = field(default_factory=float)

    def set_x0(self, x0):
        self.x0 = x0

    def set_x1(self, x1):
        self.x1 = x1

    def set_y0(self, y0):
        self.y0 = y0

    def set_y1(self, y1):
        self.y1 = y1

    def get_x0(self):
        return self.x0

    def get_x1(self):
        return self.x1

    def get_y0(self):
        return self.y0

    def get_y1(self):
        return self.y1
