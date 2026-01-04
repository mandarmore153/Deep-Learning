class LineCounter:
    def __init__(self, line_y):
        self.line_y = line_y
        self.counted_ids = set()
        self.class_counts = {}

    def count_vehicle(self, track_id, center_y, class_id):
        """
        Count vehicle only once when it crosses the line, and update class-wise counts.
        """
        if center_y > self.line_y and track_id not in self.counted_ids:
            self.counted_ids.add(track_id)
            self.class_counts[class_id] = self.class_counts.get(class_id, 0) + 1
            return True
        return False

    def get_class_counts(self):
        """
        Return the dictionary of class-wise counts with class names.
        """
        # return self.class_counts # Original code 
        class_names = {0: "car", 1: "truck"}  # Adjusted class names for best.pt model
        return {class_names[class_id]: count for class_id, count in self.class_counts.items()}