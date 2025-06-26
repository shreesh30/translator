import fitz

class Page:
    def __init__(self, number: int):
        self.number = number
        self.spans = []
        self.lines = []
        self.paragraphs = []
        self.header = []
        self.footer = []
        self.drawings = []
        self.min_x = 0
        self.max_x = 0
        self.min_y = 0
        self.max_y = 0
        self.content_width = 0
        self.content_height = 0

    def add_span(self, span):
        self.spans.append(span)

    def add_paragraph(self, para):
        self.paragraphs.append(para)

    def add_drawings(self, drawing):
        self.drawings.extend(drawing)

    def set_min_x(self, x):
        self.min_x = x

    def set_max_x(self, x):
        self.max_x = x

    def set_min_y(self, y):
        self.min_y = y

    def set_max_y(self, y):
        self.max_y = y

    def set_content_dimensions(self, min_x, max_x, min_y, max_y):
        self.set_min_x(min_x)
        self.set_max_x(max_x)
        self.set_min_y(min_y)
        self.set_max_y(max_y)

    def get_paragraphs(self):
        return self.paragraphs

    def get_header(self):
        return self.header

    def get_footer(self):
        return self.footer

    def get_content_height(self):
        return self.content_height

    def get_content_width(self):
        return self.content_width

    def sort_spans(self):
        self.spans = sorted(
            self.spans,
            key=lambda x: (x["origin"][1])
        )

    def compute_content_dimensions(self):
        self.content_width = self.max_x - self.min_x
        self.content_height = self.max_y - self.min_y

    def detect_header_footer(self):
        """Detects footer paragraphs using either a horizontal line or final line heuristics."""
        # TODO: SEPARATE HEADER FROM PARAGRAPHS
        # TODO: TRY TO SEE IF FOOTERS CAN BE ADDED AS SPANS INSTEAD OF THE WHOLE LINE, IT WILL BE EASIER MOVING FORWARD
        footer_start = self._get_footer_line_y()

        def is_footer_by_line(paragraph):
            y0 = paragraph['para_bbox'].y0
            return footer_start is not None and y0 > footer_start

        def is_footer_by_position(paragraph):
            y1 = paragraph['para_bbox'].y1
            return (
                    int(y1) == int(self.max_y) and
                    paragraph["elements"][0]["text"].isdigit()
            )

        def is_header_by_position(paragraph):
            y1 = paragraph['para_bbox'].y1
            return y1<131

        new_paragraphs = []
        self.footer = []
        self.header = []

        for para in self.paragraphs:
            if is_footer_by_line(para) or is_footer_by_position(para):
                self.footer.append(para)
            elif is_header_by_position(para):
                self.header.append(para)
            else:
                new_paragraphs.append(para)

        self.paragraphs = new_paragraphs

    def group_by_lines(self):
        """Group text spans into lines based on shared Y position.
        Adds space only when font size increases significantly within the same line.
        Stores the result in self.lines.
        """
        lines = []
        current_line = None
        prev_span = None

        def should_insert_space(previous_span, curr_span):
            if not previous_span:
                return False

            x_gap = curr_span["bbox"].x0 - previous_span["bbox"].x1
            font_increased = (curr_span["size"] - previous_span["size"]) > 2.0
            same_line = int(previous_span["origin"][1]) == int(curr_span["origin"][1])
            return (x_gap > 1.0) or (font_increased and same_line)

        for span in self.spans:
            if not span.get("text"):
                continue

            is_new_line = (
                    current_line is None or
                    int(span["origin"][1]) != int(current_line["origin"][1])
            )

            if is_new_line:
                if current_line:
                    lines.append(current_line)
                current_line = {
                    "text": span["text"],
                    "page_num": self.number,
                    "line_bbox": span["line_bbox"],
                    "bbox": fitz.Rect(span["bbox"]),
                    "origin": span["origin"],
                    "size": span["size"]
                }
            else:
                if should_insert_space(prev_span, span):
                    current_line["text"] += " " + span["text"]
                else:
                    current_line["text"] += span["text"]

                # Expand bounding box
                curr_bbox = span["bbox"]
                x0 = min(current_line["bbox"].x0, curr_bbox.x0)
                y0 = min(current_line["bbox"].y0, curr_bbox.y0)
                x1 = max(current_line["bbox"].x1, curr_bbox.x1)
                y1 = max(current_line["bbox"].y1, curr_bbox.y1)
                current_line["bbox"] = fitz.Rect(x0, y0, x1, y1)

                # Update origin
                x = min(current_line["origin"][0], span["origin"][0])
                y = max(current_line["origin"][1], span["origin"][1])
                current_line["origin"] = (x, y)

                current_line["size"] = max(current_line["size"], span["size"])

            prev_span = span

        if current_line:
            lines.append(current_line)

        self.lines = lines

    def group_by_paragraphs(self):
        """Group text lines into paragraphs based on vertical proximity.
        Stores the result in self.paragraphs.
        """
        paragraphs = []
        current_para = []

        for i, line in enumerate(self.lines):
            is_new_para = (
                    i > 0 and (
                    line["page_num"] != self.lines[i - 1]["page_num"] or
                    (int(line["origin"][1]) - int(self.lines[i - 1]["origin"][1]) > 11)
            )
            )

            if is_new_para:
                if current_para:
                    x0s = [el["bbox"].x0 for el in current_para]
                    y0s = [el["bbox"].y0 for el in current_para]
                    x1s = [el["bbox"].x1 for el in current_para]
                    y1s = [el["bbox"].y1 for el in current_para]

                    paragraphs.append({
                        "elements": current_para,
                        "page_num": current_para[0]["page_num"],
                        "font_size": current_para[0]["size"],
                        "para_bbox": fitz.Rect(min(x0s), min(y0s), max(x1s), max(y1s))
                    })
                    current_para = []

            current_para.append(line)

        if current_para:
            x0s = [el["bbox"].x0 for el in current_para]
            y0s = [el["bbox"].y0 for el in current_para]
            x1s = [el["bbox"].x1 for el in current_para]
            y1s = [el["bbox"].y1 for el in current_para]

            paragraphs.append({
                "elements": current_para,
                "page_num": current_para[0]["page_num"],
                "font_size": current_para[0]["size"],
                "para_bbox": fitz.Rect(min(x0s), min(y0s), max(x1s), max(y1s))
            })

        self.paragraphs = paragraphs

    def process_page(self):
        self.sort_spans()
        self.group_by_lines()
        self.group_by_paragraphs()
        self.detect_header_footer()
        self.compute_content_dimensions()

    def _get_footer_line_y(self):
        """Returns the Y position of a full-width horizontal line if detected."""
        for drawing in self.drawings:
            rect = fitz.Rect(drawing['rect'])
            if (
                    int(rect.x0) == int(self.min_x) and
                    int(rect.x1) == int(self.max_x) and
                    int(rect.y0) == int(rect.y1)
            ):
                return rect.y1
        return None

    def __str__(self):
        return (
            f"Page {self.number}:\n"
            f"  Paragraphs: {self.paragraphs}\n"
            f"  Header elements: {self.header}\n"
            f"  Footer elements: {self.footer}\n"
            f"  Drawings: {self.drawings}\n"
            f"  Content Width: {self.content_width}"
            f"  Width from x={self.min_x} to x={self.max_x}) \n"
            f"  Content Height: {self.content_height}"
            f"  Height from y={self.min_y} to y={self.max_y})"
        )

