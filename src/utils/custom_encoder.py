import json


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        if hasattr(obj, "x0") and hasattr(obj, "x1") and hasattr(obj, "y0") and hasattr(obj, "y1"):
            # Likely a Rect
            return obj.x0, obj.y0, obj.x1, obj.y1
        return str(obj)