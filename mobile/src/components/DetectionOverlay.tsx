import Svg, { Rect, Text as SvgText } from "react-native-svg";

import type { Detection } from "@/types";

interface Props {
  detections: Detection[];
  imageWidth: number;   // original photo's pixel dimensions, from the API response
  imageHeight: number;
  displayWidth: number; // the size the image is actually rendered at on screen
  displayHeight: number;
}

const RICE_COLOR = "#3EBD6D";
const CONE_COLOR = "#E8760C";

/**
 * Draws bounding boxes + numeric labels over the displayed photo.
 *
 * Detections come back in the ORIGINAL image's pixel coordinates, but the
 * photo is rendered on screen at a different (scaled-to-fit) size, so every
 * box has to be scaled by (displaySize / originalSize) before drawing --
 * otherwise boxes drift off their objects on any screen that isn't exactly
 * the photo's native resolution.
 */
export default function DetectionOverlay({
  detections,
  imageWidth,
  imageHeight,
  displayWidth,
  displayHeight,
}: Props) {
  if (imageWidth === 0 || imageHeight === 0) return null;

  const scaleX = displayWidth / imageWidth;
  const scaleY = displayHeight / imageHeight;

  return (
    <Svg
      width={displayWidth}
      height={displayHeight}
      style={{ position: "absolute", top: 0, left: 0 }}
    >
      {detections.map((det) => {
        const x = det.x * scaleX;
        const y = det.y * scaleY;
        const w = det.width * scaleX;
        const h = det.height * scaleY;
        const color = det.kind === "rice" ? RICE_COLOR : CONE_COLOR;
        const fontSize = det.kind === "cone" ? 16 : 11;

        return (
          <Svg.Fragment key={`${det.kind}-${det.id}`}>
            <Rect x={x} y={y} width={w} height={h} stroke={color} strokeWidth={2} fill="none" rx={2} />
            <Rect
              x={x}
              y={Math.max(0, y - fontSize - 4)}
              width={fontSize * 1.6 + String(det.id).length * (fontSize * 0.6)}
              height={fontSize + 4}
              fill={color}
              rx={3}
            />
            <SvgText
              x={x + 3}
              y={Math.max(fontSize, y - 6)}
              fill="#fff"
              fontSize={fontSize}
              fontWeight="bold"
            >
              {det.id}
            </SvgText>
          </Svg.Fragment>
        );
      })}
    </Svg>
  );
}
