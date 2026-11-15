"""Draw the three dividing lines on the invoice image."""
import cv2


def draw_lines(img, ys, color=(0, 0, 255), thickness=3):
    out = img.copy()
    w = out.shape[1]
    for y in ys:
        cv2.line(out, (0, y), (w, y), color, thickness)
    return out


def draw_labeled_lines(img, ys, names, color=(0, 0, 255), text_color=(255, 0, 0)):
    """Draw the dividing lines and write each zone's name at its top-left corner."""
    out = draw_lines(img, ys, color)
    h, w = out.shape[:2]
    scale = max(0.6, w / 2200)
    thick = max(1, int(scale * 2))
    bounds = [0, *ys, h]
    for top, name in zip(bounds[:-1], names):
        (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        # right-aligned just left of the page-edge strip, directly under the line
        x, y = int(w * 0.9) - tw - 10, top + th + 10
        if name == "items":  # the table header sits right under its top line, so go above it
            y = top - 12
        cv2.putText(out, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, text_color, thick, cv2.LINE_AA)
    return out
